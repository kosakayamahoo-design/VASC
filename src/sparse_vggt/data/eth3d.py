"""Minimal ETH3D COLMAP-text adapter.

ETH3D pre-undistorted multi-view images use COLMAP calibration files. This
module intentionally handles metadata only; downloading data and producing
benchmark submissions remain explicit evaluation steps.
"""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    def pinhole_intrinsics(self) -> np.ndarray:
        if self.model == "PINHOLE":
            fx, fy, cx, cy = self.params
        elif self.model == "SIMPLE_PINHOLE":
            focal, cx, cy = self.params
            fx = fy = focal
        else:
            raise ValueError(
                "Use ETH3D pre-undistorted PINHOLE images; "
                f"camera {self.camera_id} uses {self.model}"
            )
        return np.array(
            ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )

    def resized_intrinsics(
        self, target_width: int, target_height: int
    ) -> np.ndarray:
        if target_width < 1 or target_height < 1:
            raise ValueError("target image dimensions must be positive")
        intrinsics = self.pinhole_intrinsics()
        intrinsics[0] *= target_width / self.width
        intrinsics[1] *= target_height / self.height
        return intrinsics


def crop_preprocessed_intrinsics(
    camera: ColmapCamera,
    *,
    final_height: int,
    target_width: int = 518,
    patch_size: int = 14,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Match VGGT crop-mode resize, center crop, and batch padding."""
    if final_height < 1 or target_width < 1 or patch_size < 1:
        raise ValueError("preprocessed image dimensions must be positive")
    resized_height = (
        round(camera.height * (target_width / camera.width) / patch_size)
        * patch_size
    )
    resized_height = max(patch_size, resized_height)
    visible_height = min(resized_height, target_width)
    if final_height < visible_height:
        raise ValueError("final_height is smaller than the visible image")

    crop_top = max(0, (resized_height - target_width) // 2)
    pad_top = (final_height - visible_height) // 2
    intrinsics = camera.resized_intrinsics(target_width, resized_height)
    intrinsics[1, 2] += pad_top - crop_top
    return intrinsics, (pad_top, pad_top + visible_height)


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    quaternion_wxyz: tuple[float, float, float, float]
    translation: tuple[float, float, float]
    camera_id: int
    name: str

    def world_to_camera(self) -> np.ndarray:
        qw, qx, qy, qz = self.quaternion_wxyz
        norm = np.linalg.norm((qw, qx, qy, qz))
        if norm <= 0.0:
            raise ValueError(f"image {self.image_id} has a zero quaternion")
        qw, qx, qy, qz = np.asarray(
            (qw, qx, qy, qz), dtype=np.float64
        ) / norm
        rotation = np.array(
            (
                (
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ),
                (
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qx * qw),
                ),
                (
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx * qx + qy * qy),
                ),
            ),
            dtype=np.float64,
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = self.translation
        return transform


def read_colmap_cameras(path: str | Path) -> dict[int, ColmapCamera]:
    cameras = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"invalid COLMAP camera line: {raw_line}")
        camera = ColmapCamera(
            camera_id=int(fields[0]),
            model=fields[1],
            width=int(fields[2]),
            height=int(fields[3]),
            params=tuple(float(value) for value in fields[4:]),
        )
        cameras[camera.camera_id] = camera
    return cameras


def read_colmap_images(path: str | Path) -> list[ColmapImage]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    images = []
    line_index = 0
    while line_index < len(lines):
        header = lines[line_index].strip()
        line_index += 1
        if not header or header.startswith("#"):
            continue
        fields = header.split()
        if len(fields) < 10:
            raise ValueError(f"invalid COLMAP image line: {header}")
        images.append(
            ColmapImage(
                image_id=int(fields[0]),
                quaternion_wxyz=tuple(
                    float(value) for value in fields[1:5]
                ),
                translation=tuple(float(value) for value in fields[5:8]),
                camera_id=int(fields[8]),
                name=" ".join(fields[9:]),
            )
        )
        if line_index < len(lines):
            line_index += 1
    return sorted(images, key=lambda image: (image.name, image.image_id))


def select_frames(
    images: list[ColmapImage],
    *,
    stride: int = 1,
    max_frames: int | None = None,
    uniform: bool = False,
) -> list[ColmapImage]:
    if stride < 1:
        raise ValueError("stride must be positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    selected = images[::stride]
    if max_frames is None or len(selected) <= max_frames:
        return selected
    if not uniform:
        return selected[:max_frames]
    indices = np.linspace(0, len(selected) - 1, max_frames, dtype=np.int64)
    return [selected[int(index)] for index in indices]


def order_many_view_frames(images: list[ColmapImage]) -> list[ColmapImage]:
    """Interleave synchronized rig cameras by image timestamp when possible."""
    return sorted(
        images,
        key=lambda image: (
            Path(image.name).stem,
            Path(image.name).parent.as_posix(),
            image.image_id,
        ),
    )


def select_rig_frame_groups(
    images: list[ColmapImage],
    *,
    stride: int = 1,
    max_frames: int | None = None,
) -> list[ColmapImage]:
    """Sample capture indices while preserving every camera in the rig."""
    if stride < 1:
        raise ValueError("stride must be positive")
    streams: dict[str, list[ColmapImage]] = {}
    for image in images:
        stream = Path(image.name).parent.as_posix()
        streams.setdefault(stream, []).append(image)
    if len(streams) < 2:
        return select_frames(
            order_many_view_frames(images),
            stride=stride,
            max_frames=max_frames,
            uniform=True,
        )

    ordered_streams = []
    for stream in sorted(streams):
        ordered_streams.append(
            sorted(
                streams[stream],
                key=lambda image: (Path(image.name).stem, image.image_id),
            )
        )
    capture_count = min(len(stream) for stream in ordered_streams)
    capture_indices = list(range(0, capture_count, stride))
    cameras_per_capture = len(ordered_streams)
    if max_frames is not None:
        max_captures = max_frames // cameras_per_capture
        if max_captures < 1:
            raise ValueError("max_frames cannot fit one complete rig capture")
        if len(capture_indices) > max_captures:
            selected = np.linspace(
                0, len(capture_indices) - 1, max_captures, dtype=np.int64
            )
            capture_indices = [capture_indices[int(index)] for index in selected]

    return [
        stream[capture_index]
        for capture_index in capture_indices
        for stream in ordered_streams
    ]


def load_transformed_scan_points(mlp_path: str | Path) -> np.ndarray:
    """Load all PLY vertices from a MeshLab project in aligned coordinates."""
    from plyfile import PlyData

    project_path = Path(mlp_path)
    root = ET.parse(project_path).getroot()
    point_sets = []
    for mesh in root.findall(".//MLMesh"):
        matrix_node = mesh.find("MLMatrix44")
        filename = mesh.attrib.get("filename")
        if matrix_node is None or not filename:
            continue
        values = np.fromstring(matrix_node.text or "", sep=" ", dtype=np.float64)
        if values.size != 16:
            raise ValueError(f"invalid MeshLab transform for {filename}")
        transform = values.reshape(4, 4)
        vertices = PlyData.read(project_path.parent / filename)["vertex"].data
        points = np.column_stack(
            (vertices["x"], vertices["y"], vertices["z"])
        ).astype(np.float32)
        points = (
            points @ transform[:3, :3].astype(np.float32).T
            + transform[:3, 3].astype(np.float32)
        )
        point_sets.append(points)
    if not point_sets:
        raise ValueError(f"no transformed scans found in {project_path}")
    return np.concatenate(point_sets, axis=0)


def render_scan_depth_maps(
    scan_points: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    *,
    height: int,
    width: int,
    chunk_size: int = 500_000,
) -> np.ndarray:
    """Render camera-z depth with a deterministic nearest-point z-buffer."""
    points = np.asarray(scan_points, dtype=np.float32)
    calibration = np.asarray(intrinsics, dtype=np.float64)
    extrinsics = np.asarray(world_to_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("scan_points must have shape [P, 3]")
    if calibration.ndim != 3 or calibration.shape[1:] != (3, 3):
        raise ValueError("intrinsics must have shape [N, 3, 3]")
    if extrinsics.shape == (len(calibration), 4, 4):
        extrinsics = extrinsics[:, :3]
    if extrinsics.shape != (len(calibration), 3, 4):
        raise ValueError("world_to_camera must have shape [N, 3, 4] or [N, 4, 4]")
    if height < 1 or width < 1 or chunk_size < 1:
        raise ValueError("render dimensions and chunk size must be positive")

    depth_maps = np.zeros((len(calibration), height, width), dtype=np.float32)
    for camera_index, (intrinsic, extrinsic) in enumerate(
        zip(calibration, extrinsics)
    ):
        depth = np.full(height * width, np.inf, dtype=np.float32)
        rotation = extrinsic[:3, :3].astype(np.float32)
        translation = extrinsic[:3, 3].astype(np.float32)
        for start in range(0, len(points), chunk_size):
            camera_points = (
                points[start : start + chunk_size] @ rotation.T + translation
            )
            z = camera_points[:, 2]
            visible = np.isfinite(camera_points).all(axis=1) & (z > 1e-6)
            if not np.any(visible):
                continue
            camera_points = camera_points[visible]
            z = camera_points[:, 2]
            pixel_x = np.rint(
                intrinsic[0, 0] * camera_points[:, 0] / z + intrinsic[0, 2]
            ).astype(np.int64)
            pixel_y = np.rint(
                intrinsic[1, 1] * camera_points[:, 1] / z + intrinsic[1, 2]
            ).astype(np.int64)
            inside = (
                (pixel_x >= 0)
                & (pixel_x < width)
                & (pixel_y >= 0)
                & (pixel_y < height)
            )
            flat_indices = pixel_y[inside] * width + pixel_x[inside]
            np.minimum.at(depth, flat_indices, z[inside])
        depth[~np.isfinite(depth)] = 0.0
        depth_maps[camera_index] = depth.reshape(height, width)
    return depth_maps


def qk_scale_shift_alignment(
    predicted_points: np.ndarray,
    ground_truth_points: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Reproduce QK sparse attention's joint z-shift and robust scale alignment."""
    predicted = np.asarray(predicted_points, dtype=np.float64).copy()
    ground_truth = np.asarray(ground_truth_points, dtype=np.float64).copy()
    valid = np.asarray(valid_mask, dtype=bool)
    if predicted.shape != ground_truth.shape or predicted.shape[-1] != 3:
        raise ValueError("point maps must share shape [N, H, W, 3]")
    if valid.shape != predicted.shape[:-1]:
        raise ValueError("valid_mask must match point-map spatial dimensions")
    valid &= np.isfinite(predicted).all(axis=-1)
    valid &= np.isfinite(ground_truth).all(axis=-1)
    if np.count_nonzero(valid) < 128:
        raise ValueError("not enough valid points for QK sparse attention alignment")

    gt_shift_z = float(np.median(ground_truth[..., 2][valid]))
    pred_shift_z = float(np.median(predicted[..., 2][valid]))
    ground_truth[..., 2] -= gt_shift_z
    predicted[..., 2] -= pred_shift_z

    gt_values = ground_truth[valid]
    pred_values = predicted[valid]
    gt_center = np.median(gt_values, axis=0)
    pred_center = np.median(pred_values, axis=0)
    gt_scale = float(np.median(np.linalg.norm(gt_values - gt_center, axis=1)))
    pred_scale = float(
        np.median(np.linalg.norm(pred_values - pred_center, axis=1))
    )
    pred_scale = float(np.clip(pred_scale, 1e-3, 1e3))
    if not np.isfinite(gt_scale) or gt_scale <= 0.0:
        raise ValueError("ground-truth point scale is invalid")
    scale_ratio = gt_scale / pred_scale
    predicted *= scale_ratio
    return predicted, ground_truth, {
        "gt_shift_z": gt_shift_z,
        "pred_shift_z": pred_shift_z,
        "gt_scale": gt_scale,
        "pred_scale": pred_scale,
        "scale_ratio": scale_ratio,
    }


def estimate_similarity_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate target = scale * rotation @ source + translation."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_points and target_points must both have shape [N, 3]")
    if len(source) < 3:
        raise ValueError("at least three point correspondences are required")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    variance = np.mean(np.sum(source_zero * source_zero, axis=1))
    if variance < 1e-12:
        raise ValueError("cannot align degenerate source points")

    covariance = (target_zero.T @ source_zero) / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    reflection = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        reflection[-1, -1] = -1
    rotation = u @ reflection @ vt
    scale = float(np.sum(singular_values * np.diag(reflection)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def apply_similarity_transform(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points)
    return scale * np.einsum("ij,...j->...i", rotation, points) + translation


def estimate_depth_branch_scale(
    depth_maps: np.ndarray,
    world_points: np.ndarray,
    world_to_camera: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Match depth-head camera-z values to the point-head coordinate scale."""
    depths = np.asarray(depth_maps, dtype=np.float64)
    points = np.asarray(world_points, dtype=np.float64)
    extrinsics = np.asarray(world_to_camera, dtype=np.float64)
    if depths.ndim != 3:
        raise ValueError("depth_maps must have shape [N, H, W]")
    if points.shape != depths.shape + (3,):
        raise ValueError("world_points must have shape [N, H, W, 3]")
    if extrinsics.shape != (len(depths), 3, 4):
        raise ValueError("world_to_camera must have shape [N, 3, 4]")

    camera_points = (
        np.einsum("nij,nhwj->nhwi", extrinsics[:, :3, :3], points)
        + extrinsics[:, None, None, :3, 3]
    )
    point_depth = camera_points[..., 2]
    valid = (
        np.isfinite(depths)
        & np.isfinite(point_depth)
        & (depths > 1e-8)
        & (point_depth > 1e-8)
    )
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != depths.shape:
            raise ValueError("valid_mask must match depth_maps")
        valid &= mask
    ratios = point_depth[valid] / depths[valid]
    if ratios.size < 128:
        raise ValueError("not enough valid points for depth branch scaling")
    lower, upper = np.quantile(ratios, (0.05, 0.95))
    ratios = ratios[(ratios >= lower) & (ratios <= upper)]
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("estimated depth branch scale is invalid")
    return scale


def unproject_depth_maps(
    depth_maps: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Unproject camera-z depth with calibrated cameras into world space."""
    depths = np.asarray(depth_maps, dtype=np.float32)
    calibration = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    if depths.ndim != 3:
        raise ValueError("depth_maps must have shape [N, H, W]")
    if calibration.shape != (len(depths), 3, 3):
        raise ValueError("intrinsics must have shape [N, 3, 3]")
    if poses.shape != (len(depths), 4, 4):
        raise ValueError("camera_to_world must have shape [N, 4, 4]")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")

    height, width = depths.shape[1:]
    pixel_x, pixel_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    world_points = np.empty(depths.shape + (3,), dtype=np.float32)
    for index, depth in enumerate(depths):
        fx = float(calibration[index, 0, 0])
        fy = float(calibration[index, 1, 1])
        cx = float(calibration[index, 0, 2])
        cy = float(calibration[index, 1, 2])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        camera_points = np.stack(
            (
                (pixel_x - cx) * depth / fx,
                (pixel_y - cy) * depth / fy,
                depth,
            ),
            axis=-1,
        )
        rotation = poses[index, :3, :3].astype(np.float32)
        translation = poses[index, :3, 3].astype(np.float32)
        world_points[index] = (
            scale * np.einsum("ij,hwj->hwi", rotation, camera_points)
            + translation
        )
    return world_points
