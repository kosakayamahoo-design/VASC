"""Pi3-native indoor evaluation for dense and sparse transfer experiments.

Pi3 camera-to-world poses and global point maps are evaluated with explicit
Pi3 conventions.  VGGT's scale/shift reconstruction criterion is deliberately
excluded from this adapter.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


def optional_float(value):
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return float(value)


def sparsity_list(value):
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("layer sparsity list cannot be empty")
    if any(item < 0.0 or item > 1.0 for item in values):
        raise argparse.ArgumentTypeError("layer sparsity values must be in [0, 1]")
    return values


def get_args_parser():
    parser = argparse.ArgumentParser("Dense/sparse Pi3 indoor evaluation")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pi3_root", required=True)
    parser.add_argument("--streamvggt_root", required=True)
    parser.add_argument("--dataset", choices=["7scenes", "nrgbd"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics_json", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--seq_id", default=None)
    parser.add_argument("--kf_every", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--size", type=int, default=518, choices=[518])
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--sparse_ratio", type=float, default=None)
    parser.add_argument("--layer_sparsity", type=sparsity_list, default=None)
    parser.add_argument("--cdf_threshold", type=optional_float, default=None)
    parser.add_argument("--pool_mode", choices=["avg", "max"], default="avg")
    parser.add_argument("--use_hilbert", action="store_true")
    parser.add_argument("--pose_only", action="store_true")
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--timing_warmup", type=int, default=1)
    parser.add_argument("--timing_repeats", type=int, default=3)
    parser.add_argument("--profile_attention", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_checkpoint_hash", action="store_true")
    return parser


def add_repo_paths(pi3_root, streamvggt_root):
    sparse_src = str(Path(__file__).resolve().parent.parent / "src")
    stream_src = str(Path(streamvggt_root).resolve() / "src")
    for path in (str(Path(pi3_root).resolve()), sparse_src, stream_src):
        if path not in sys.path:
            sys.path.insert(0, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_homogeneous(extrinsics):
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(extrinsics), axis=0)
    poses[:, :3, :4] = extrinsics
    return poses


def align_camera_poses_sim3(pred_c2w, gt_c2w):
    pred_centers = pred_c2w[:, :3, 3]
    gt_centers = gt_c2w[:, :3, 3]
    pred_mean = pred_centers.mean(axis=0)
    gt_mean = gt_centers.mean(axis=0)
    pred_zero = pred_centers - pred_mean
    gt_zero = gt_centers - gt_mean
    variance = np.mean(np.sum(pred_zero * pred_zero, axis=1))
    if variance < 1e-12:
        raise ValueError("Cannot align a degenerate predicted trajectory")
    covariance = (gt_zero.T @ pred_zero) / len(pred_centers)
    u, singular_values, vt = np.linalg.svd(covariance)
    reflection = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        reflection[-1, -1] = -1
    rotation = u @ reflection @ vt
    scale = float(np.sum(singular_values * np.diag(reflection)) / variance)
    translation = gt_mean - scale * (rotation @ pred_mean)
    aligned = pred_c2w.copy()
    aligned[:, :3, :3] = rotation[None] @ pred_c2w[:, :3, :3]
    aligned[:, :3, 3] = (scale * (rotation @ pred_centers.T)).T + translation
    return aligned, scale


def trajectory_metrics_from_c2w(pred_c2w, gt_c2w):
    pred_c2w = np.asarray(pred_c2w, dtype=np.float64)
    gt_c2w = np.asarray(gt_c2w, dtype=np.float64)
    aligned, scale = align_camera_poses_sim3(pred_c2w, gt_c2w)
    translation_errors = np.linalg.norm(
        aligned[:, :3, 3] - gt_c2w[:, :3, 3], axis=1
    )
    result = {
        "ate_rmse_m": float(np.sqrt(np.mean(translation_errors**2))),
        "alignment_scale": scale,
        "num_frames": int(len(gt_c2w)),
    }
    if len(gt_c2w) < 2:
        result.update({"rpe_trans_rmse_m": None, "rpe_rot_rmse_deg": None})
        return result
    trans_errors = []
    rot_errors = []
    for idx in range(len(gt_c2w) - 1):
        gt_rel = np.linalg.inv(gt_c2w[idx]) @ gt_c2w[idx + 1]
        pred_rel = np.linalg.inv(aligned[idx]) @ aligned[idx + 1]
        error = np.linalg.inv(gt_rel) @ pred_rel
        trans_errors.append(np.linalg.norm(error[:3, 3]))
        cosine = (np.trace(error[:3, :3]) - 1.0) / 2.0
        rot_errors.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    result.update(
        {
            "rpe_trans_rmse_m": float(np.sqrt(np.mean(np.square(trans_errors)))),
            "rpe_rot_rmse_deg": float(np.sqrt(np.mean(np.square(rot_errors)))),
        }
    )
    return result


class Pi3AttentionTimer:
    """CUDA-time only Pi3's 18 cross-frame decoder attentions."""

    def __init__(self, model):
        self.enabled = False
        self.event_pairs = []
        self.active_starts = []
        self.handles = []
        for raw_idx in range(1, len(model.decoder), 2):
            attention = model.decoder[raw_idx].attn
            self.handles.append(attention.register_forward_pre_hook(self._pre))
            self.handles.append(attention.register_forward_hook(self._post))

    def _pre(self, _module, _inputs):
        if self.enabled:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self.active_starts.append(start)

    def _post(self, _module, _inputs, _output):
        if self.enabled:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.event_pairs.append((self.active_starts.pop(), end))

    def begin(self):
        self.event_pairs = []
        self.active_starts = []
        self.enabled = True

    def finish(self):
        self.enabled = False
        if self.active_starts:
            raise RuntimeError("Unbalanced Pi3 attention timing hooks")
        return float(sum(start.elapsed_time(end) for start, end in self.event_pairs))

    def close(self):
        for handle in self.handles:
            handle.remove()


def build_model(args, device):
    from pi3.models.pi3 import Pi3
    from safetensors.torch import load_file

    model = Pi3()
    state = load_file(args.weights, device="cpu")
    model.load_state_dict(state, strict=True)
    if not args.dense:
        from sparse_vggt.models.pi3 import sparse_model_from_pi3

        model, aux_store = sparse_model_from_pi3(
            model,
            sparse_ratio=args.sparse_ratio,
            layer_sparsity_ratios=args.layer_sparsity,
            cdf_threshold=args.cdf_threshold,
            pool_mode=args.pool_mode,
            use_hilbert=args.use_hilbert,
            aux_output=True,
            aux_sparsity_only=True,
            verbose=True,
        )
        model.sparse_aux_output_store = aux_store
    model.eval()
    return model.to(device)


def collect_sparse_metrics(model):
    stores = getattr(model, "sparse_aux_output_store", None)
    if not isinstance(stores, dict):
        return {}, []
    scalar_values = {}
    per_layer = []
    for layer_idx, store in stores.items():
        if not isinstance(store, dict):
            continue
        row = {"layer": int(layer_idx)}
        for name, value in store.items():
            if torch.is_tensor(value) and value.numel() == 1:
                value = float(value.detach().float().cpu().item())
            elif isinstance(value, (int, float, bool)):
                value = float(value)
            else:
                continue
            row[name] = value
            scalar_values.setdefault(name, []).append(value)
        if len(row) > 1:
            per_layer.append(row)
    summary = {
        name: float(np.mean(values)) for name, values in scalar_values.items()
    }
    return summary, per_layer


def reconstruction_metrics(batch, predictions, args, data_idx):
    import open3d as o3d
    from eval.mv_recon.utils import accuracy, completion
    from sparse_vggt.evaluation.pi3_reconstruction import (
        PROTOCOL_NAME,
        estimate_umeyama_similarity,
        sample_aligned_point_clouds,
    )

    predicted = predictions["points"]
    if predicted.ndim != 5 or predicted.shape[0] != 1:
        raise ValueError("Pi3 points must have shape [1,N,H,W,3]")
    if predicted.shape[1] != len(batch):
        raise ValueError("Pi3 prediction/view counts differ")
    pred_maps = []
    gt_maps = []
    valid_masks = []
    for idx, view in enumerate(batch):
        pred_maps.append(predicted[0, idx].detach().float().cpu().numpy())
        gt_maps.append(view["pts3d"][0].detach().float().cpu().numpy())
        valid_masks.append(view["valid_mask"][0].detach().cpu().numpy())

    similarity = estimate_umeyama_similarity(
        pred_maps, gt_maps, valid_masks
    )
    pred_points, gt_points, total_points = sample_aligned_point_clouds(
        pred_maps,
        gt_maps,
        valid_masks,
        similarity,
        max_points=args.max_points,
        seed=args.seed + int(data_idx),
    )
    pred_cloud = o3d.geometry.PointCloud()
    pred_cloud.points = o3d.utility.Vector3dVector(pred_points)
    gt_cloud = o3d.geometry.PointCloud()
    gt_cloud.points = o3d.utility.Vector3dVector(gt_points)
    registration = o3d.pipelines.registration.registration_icp(
        pred_cloud,
        gt_cloud,
        0.1,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pred_cloud.transform(registration.transformation)
    pred_cloud.estimate_normals()
    gt_cloud.estimate_normals()
    pred_normals = np.asarray(pred_cloud.normals)
    gt_normals = np.asarray(gt_cloud.normals)
    acc, acc_med, nc1, nc1_med = accuracy(
        gt_cloud.points, pred_cloud.points, gt_normals, pred_normals
    )
    comp, comp_med, nc2, nc2_med = completion(
        gt_cloud.points, pred_cloud.points, gt_normals, pred_normals
    )
    return {
        "reconstruction_protocol": PROTOCOL_NAME,
        "reconstruction_umeyama_scale": float(similarity["scale"]),
        "reconstruction_alignment_correspondences": int(
            similarity["correspondences"]
        ),
        "reconstruction_valid_points": int(total_points),
        "reconstruction_icp_fitness": float(registration.fitness),
        "reconstruction_icp_inlier_rmse": float(registration.inlier_rmse),
        "acc": float(acc),
        "comp": float(comp),
        "nc1": float(nc1),
        "nc2": float(nc2),
        "nc": float((nc1 + nc2) / 2),
        "acc_med": float(acc_med),
        "comp_med": float(comp_med),
        "nc1_med": float(nc1_med),
        "nc2_med": float(nc2_med),
        "nc_med": float((nc1_med + nc2_med) / 2),
        "pred_points": int(len(pred_points)),
        "gt_points": int(len(gt_points)),
    }


def summarize(records):
    names = sorted({name for record in records for name in record})
    summary = {"num_sequences": len(records)}
    for name in names:
        values = [record.get(name) for record in records]
        values = [value for value in values if isinstance(value, (int, float))]
        if values:
            summary[f"{name}_mean"] = float(np.mean(values))
            summary[f"{name}_median"] = float(np.median(values))
    return summary


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main(args):
    add_repo_paths(args.pi3_root, args.streamvggt_root)
    set_random_seeds(args.seed)
    if args.dense and (args.sparse_ratio is not None or args.cdf_threshold is not None):
        raise ValueError("--dense cannot be combined with sparse selection")
    if not args.dense and args.sparse_ratio is None and args.cdf_threshold is None:
        raise ValueError("Sparse Pi3 requires --sparse_ratio or --cdf_threshold")
    if args.layer_sparsity is not None and len(args.layer_sparsity) != 18:
        raise ValueError("Pi3 layer sparsity must contain 18 global-attention values")
    if args.num_frames is not None and args.num_frames < 2:
        raise ValueError("--num_frames must be at least 2 for trajectory metrics")
    if args.timing_warmup < 0 or args.timing_repeats < 1:
        raise ValueError("Invalid timing warmup/repeat counts")

    from eval.mv_recon.data import NRGBD, SevenScenes
    from sparse_vggt.evaluation.pi3_reconstruction import PROTOCOL_NAME

    resolution = (518, 392)
    common = {
        "split": "test",
        "ROOT": args.data_root,
        "resolution": resolution,
        "num_seq": 1,
        "full_video": True,
        "test_id": args.scene,
        "kf_every": args.kf_every,
    }
    if args.dataset == "7scenes":
        dataset = SevenScenes(seq_id=args.seq_id, **common)
    else:
        if args.seq_id is not None:
            raise ValueError("--seq_id is only valid for 7Scenes")
        dataset = NRGBD(**common)
        if hasattr(dataset, "scene_list"):
            dataset.scene_list = sorted(dataset.scene_list)

    if not torch.cuda.is_available():
        raise RuntimeError("Pi3 evaluation requires CUDA")
    device = torch.device("cuda")
    model = build_model(args, device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    records = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_json or output_dir / "metrics.json")
    checkpoint_path = Path(args.weights)
    provenance = {
        "model": "Pi3",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": (
            None if args.skip_checkpoint_hash else sha256_file(checkpoint_path)
        ),
        "camera_pose_convention": "Pi3 c2w; evaluated directly after Sim(3)",
        "reconstruction_protocol": PROTOCOL_NAME,
        "reconstruction_alignment": "full-valid-mask sequence Umeyama Sim(3), then ICP(0.1m)",
        "reconstruction_crop": "none beyond dataset principal-point resize",
        "reconstruction_confidence_filter": "none",
        "reconstruction_metric_sampling": "paired deterministic uniform sample",
        "input_resolution": list(resolution),
        "global_decoder_layers": 18,
    }

    with torch.inference_mode():
        for data_idx in range(len(dataset)):
            sequence_views = dataset[data_idx]
            if args.num_frames is not None:
                if len(sequence_views) < args.num_frames:
                    raise ValueError(
                        f"Sequence {data_idx} has {len(sequence_views)} sampled frames, "
                        f"fewer than requested {args.num_frames}"
                    )
                sequence_views = sequence_views[: args.num_frames]
            batch = default_collate([sequence_views])
            for view in batch:
                for name, value in list(view.items()):
                    if name in {"dataset", "label", "instance", "idx", "true_shape", "rng"}:
                        continue
                    if isinstance(value, (tuple, list)):
                        view[name] = [item.to(device, non_blocking=True) for item in value]
                    elif torch.is_tensor(value):
                        view[name] = value.to(device, non_blocking=True)
            images = torch.cat(
                [((view["img"] + 1.0) / 2.0) for view in batch], dim=0
            ).unsqueeze(0)

            for _ in range(args.timing_warmup):
                with torch.amp.autocast("cuda", dtype=dtype):
                    model(images)
            torch.cuda.synchronize(device)

            attention_times = []
            if args.profile_attention:
                timer = Pi3AttentionTimer(model)
                for _ in range(args.timing_repeats):
                    timer.begin()
                    with torch.amp.autocast("cuda", dtype=dtype):
                        model(images)
                    torch.cuda.synchronize(device)
                    attention_times.append(timer.finish())
                timer.close()

            torch.cuda.reset_peak_memory_stats(device)
            inference_times = []
            predictions = None
            for _ in range(args.timing_repeats):
                start = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=dtype):
                    predictions = model(images)
                torch.cuda.synchronize(device)
                inference_times.append((time.perf_counter() - start) * 1000.0)

            pred_c2w = predictions["camera_poses"][0].detach().cpu().numpy()
            gt_c2w = np.stack(
                [view["camera_pose"][0].detach().cpu().numpy() for view in batch]
            )
            record = trajectory_metrics_from_c2w(pred_c2w, gt_c2w)
            scene_id = batch[-1]["label"][0].rsplit("/", 1)[0]
            sparse_summary, per_layer = collect_sparse_metrics(model)
            record.update(sparse_summary)
            record.update(
                {
                    "scene_id": scene_id,
                    "inference_ms": float(np.median(inference_times)),
                    "inference_repeats_ms": inference_times,
                    "attention_ms": (
                        float(np.median(attention_times)) if attention_times else None
                    ),
                    "attention_repeats_ms": attention_times,
                    "peak_memory_mb": float(
                        torch.cuda.max_memory_allocated(device) / (1024**2)
                    ),
                    "sparse_per_layer": per_layer,
                }
            )
            if not args.pose_only:
                record.update(
                    reconstruction_metrics(batch, predictions, args, data_idx)
                )
            records.append(record)
            payload = {
                "config": vars(args),
                "provenance": provenance,
                "records": records,
                "summary": summarize(records),
            }
            write_json(metrics_path, payload)
            print(
                f"{scene_id}: ATE={record['ate_rmse_m']:.6f}, "
                f"RPE-t={record['rpe_trans_rmse_m']:.6f}, "
                f"RPE-r={record['rpe_rot_rmse_deg']:.6f}, "
                f"time={record['inference_ms']:.2f} ms"
            )
            del predictions
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
