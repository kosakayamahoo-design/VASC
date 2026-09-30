"""Evaluate VASC, Dense, or Top-K attention on VGGT and Pi3."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "configs/paper.json").read_text(encoding="utf-8"))
MODES = ("dense", "topk", "vasc")


def layer_sparsity(backbone, sparsity, allocation):
    if backbone not in ("vggt", "pi3") or allocation not in ("paper", "uniform"):
        raise ValueError("unsupported backbone or allocation")
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in [0, 1)")
    policy = CONFIG["structural"]
    if allocation == "uniform" or sparsity <= policy["enable_above"]:
        return None
    shape = policy[backbone]
    selected = shape["favored_layers_zero_based"]
    delta = policy["delta"]
    offset = len(selected) * delta / (shape["layers"] - len(selected))
    values = [sparsity - delta if i in selected else sparsity + offset
              for i in range(shape["layers"])]
    if any(not 0 <= x < 1 for x in values):
        raise ValueError("structural profile leaves [0, 1); use uniform allocation")
    return ",".join(f"{v:.10f}" for v in values)


def allocation_settings(args):
    enabled = (args.allocation == "paper" if args.allocation is not None
               else args.structural_allocation)
    if enabled is None:
        enabled = CONFIG["structural"]["enabled"]
    policy = "paper" if enabled and args.mode != "dense" else "uniform"
    profile = layer_sparsity(args.backbone, args.sparsity, policy)
    ratios = (list(map(float, profile.split(","))) if profile is not None else
              [0.0 if args.mode == "dense" else args.sparsity]
              * CONFIG["structural"][args.backbone]["layers"])
    return {
        "enabled": enabled,
        "applied": profile is not None,
        "effective_policy": ("dense" if args.mode == "dense" else
                             "paper" if profile is not None else "uniform"),
        "requested_sparsity": args.sparsity,
        "layer_sparsities": ratios,
    }


def method_environment(backbone, mode):
    if backbone not in ("vggt", "pi3") or mode not in MODES:
        raise ValueError("unsupported backbone or mode")
    active = mode == "vasc"
    # Recreate the tested selection switches, independent of shell experiments.
    values = {
        "COSA_VALUE_RISK_PAIR_DEBT": int(active),
        "COSA_PRODUCTION_FAST": 1,
        "COSA_ORDERED_LUT": 0,
        "COSA_TOPK_ORDERED_LUT": 0,
        "COSA_COMPACT_ORDERED_LUT": 0,
        "COSA_HRM_ORDERED_LUT": 0,
        "COSA_FP32_PROJECTION": int(active or backbone == "vggt"),
        "COSA_DEBT_SERVICE_DOMAIN": "qk_admission" if active or backbone == "vggt" else "pv_execution",
        "COSA_KERNEL_PAIR_SERVICE": 0,
        "COSA_INDEX_SERVICE_REDUCTION": int(active),
        "COSA_CONSERVATIVE_DEBT_PROJECTION": int(active),
        "COSA_CONSERVATIVE_ARRIVAL_PROJECTION": int(active),
        "COSA_PV_COUNT_STATS": 0,
        "COLLECT_QK_PV_SERVICE": 0,
        "SKIP_ROUTING_STATS": 1,
        "QK_PV_THRESHOLD": "1e10",
    }
    if backbone == "vggt" and mode != "dense":
        values["OFFICIAL_MASK"] = 1
    return {"SPARSE_VGGT_" + k: str(v) for k, v in values.items()}


def parser(defaults=None):
    defaults = defaults or {}
    p = argparse.ArgumentParser(
        description=__doc__,
        usage="%(prog)s [--config PATH] [overrides]\n       %(prog)s --init-config [--config PATH]",
        epilog="Daily use: python evaluate.py. Configure paths once in configs/local.json; "
               "CLI overrides are optional. See docs/usage.md for advanced options.")
    p.add_argument("--config", type=Path, help="Local JSON settings; defaults to configs/local.json when present")
    p.add_argument("--init-config", action="store_true", help="Create local settings from the example without overwriting an existing file")
    p.add_argument("--backbone", choices=("vggt", "pi3"), required="backbone" not in defaults)
    p.add_argument("--mode", choices=MODES, default="vasc")
    p.add_argument("--dataset", choices=("7scenes", "nrgbd"), required="dataset" not in defaults)
    p.add_argument("--data-root", type=Path, required="data_root" not in defaults)
    p.add_argument("--weights", type=Path, required="weights" not in defaults)
    p.add_argument("--vggt-root", type=Path, default=ROOT / "external/vggt")
    p.add_argument("--pi3-root", type=Path, default=ROOT / "external/Pi3")
    p.add_argument("--streamvggt-root", type=Path, default=ROOT / "external/StreamVGGT")
    p.add_argument("--runtime-root", type=Path, default=ROOT / "third_party/spargeattn")
    p.add_argument("--output", type=Path, help="Exact output directory; otherwise create a timestamped run")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs", help="Parent directory for automatic run folders")
    p.add_argument("--gpu", default="0")
    p.add_argument("--sparsity", type=float, default=0.75)
    allocation = p.add_mutually_exclusive_group()
    allocation.add_argument(
        "--structural-allocation", action=argparse.BooleanOptionalAction, default=None,
        help="Enable/disable the fixed cross-layer budget profile above 70%% sparsity; "
             "enabled by default; disable with --no-structural-allocation or the local config")
    allocation.add_argument(
        "--allocation", choices=("paper", "uniform"), default=None,
        help="Compatibility alias: paper enables structural allocation, uniform disables it")
    p.add_argument("--stride", type=int, default=CONFIG["stride"])
    p.add_argument("--scene")
    p.add_argument("--sequence")
    p.add_argument("--frame-limit", type=int, help="Explicit short test; omitted for paper evaluation")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--pose-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(**defaults)
    return p


def load_run_config(path):
    with path.open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("run config must be a JSON object")
    path_keys = {"data_root", "weights", "vggt_root", "pi3_root", "streamvggt_root",
                 "runtime_root", "output", "output_root"}
    integer_keys = {"stride", "frame_limit", "warmup", "repeats"}
    boolean_keys = {"structural_allocation", "pose_only"}
    choices = {"backbone": ("vggt", "pi3"), "dataset": ("7scenes", "nrgbd"), "mode": MODES}
    string_keys = {"gpu", "scene", "sequence", *choices}
    known = path_keys | integer_keys | boolean_keys | string_keys | {"sparsity"}
    unknown = data.keys() - known
    if unknown:
        raise ValueError("unknown config key(s): " + ", ".join(sorted(unknown)))
    result = {}
    for key, value in data.items():
        if key in path_keys | string_keys:
            valid = isinstance(value, str) and bool(value.strip())
        elif key in boolean_keys:
            valid = type(value) is bool
        elif key in integer_keys:
            valid = type(value) is int
        else:
            valid = type(value) in (int, float)
        if not valid:
            raise ValueError(f"invalid type or empty value for config key: {key}")
        if key in choices and value not in choices[key]:
            raise ValueError(f"invalid {key}: {value}; choose from {', '.join(choices[key])}")
        if key in path_keys:
            value = Path(value).expanduser()
            value = (value if value.is_absolute() else ROOT / value).resolve()
        result[key] = value
    return result


def build_launch(args):
    if args.stride < 1 or args.warmup < 0 or args.repeats < 1:
        raise ValueError("stride/repeats must be positive and warmup nonnegative")
    if args.frame_limit is not None and args.frame_limit < 1:
        raise ValueError("frame-limit must be positive")
    if args.sequence and (args.dataset != "7scenes" or not args.scene):
        raise ValueError("sequence requires a 7scenes scene")
    if args.dataset == "7scenes" and args.scene and not args.sequence:
        raise ValueError("a single 7scenes scene requires --sequence")
    allocation = allocation_settings(args)
    script = "eval_7scenes_sparse_vggt.py" if args.backbone == "vggt" else "eval_pi3_sparse.py"
    if args.output is None:
        policy = allocation["effective_policy"]
        name = (f"{args.backbone}_{args.dataset}_{args.mode}_s{args.sparsity:g}_{policy}_"
                f"{datetime.now():%Y%m%d-%H%M%S-%f}")
        args.output = args.output_root / name
    out = args.output.resolve()
    command = [sys.executable, str(ROOT / "evaluation" / script)]
    options = {
        "weights": args.weights.resolve(), "streamvggt_root": args.streamvggt_root.resolve(),
        "dataset": args.dataset, "data_root": args.data_root.resolve(),
        "output_dir": out / "reconstruction", "metrics_json": out / "metrics.json",
        "kf_every": args.stride, "timing_warmup": args.warmup, "timing_repeats": args.repeats,
    }
    if args.backbone == "vggt":
        options["vggt_root"] = args.vggt_root.resolve()
        command.append("--aux_output")
        if args.dataset == "nrgbd":
            options["max_points"] = 999999
    else:
        options["pi3_root"] = args.pi3_root.resolve()
        options["max_points"] = 200000
    if args.mode == "dense":
        command.append("--vanilla_vggt" if args.backbone == "vggt" else "--dense")
    else:
        options.update(sparse_ratio=args.sparsity, cdf_threshold="none")
        if allocation["applied"]:
            options["layer_sparsity"] = ",".join(
                f"{v:.10f}" for v in allocation["layer_sparsities"])
    if args.scene:
        options["scene"] = args.scene
    if args.sequence:
        options["seq_id"] = args.sequence
    if args.frame_limit:
        options["num_frames"] = args.frame_limit
    if args.pose_only:
        command.append("--pose_only")
    for key, value in options.items():
        command.extend(["--" + key, str(value)])
    paths = [args.runtime_root, args.pi3_root, args.vggt_root, ROOT / "src"]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("SPARSE_VGGT_") and k != "PYTHONPATH"}
    env.update(method_environment(args.backbone, args.mode))
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(str(p.resolve()) for p in paths))
    return command, env


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--help" in argv or "-h" in argv:
        parser().parse_args(argv)
    setup = argparse.ArgumentParser(add_help=False)
    setup.add_argument("--config", type=Path)
    setup.add_argument("--init-config", action="store_true")
    settings, rest = setup.parse_known_args(argv)
    config_path = (settings.config or ROOT / "configs/local.json").expanduser().resolve()
    if settings.init_config:
        if rest:
            setup.error("--init-config only accepts --config to choose the destination")
        try:
            example = (ROOT / "configs/local.example.json").read_text(encoding="utf-8")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open("x", encoding="utf-8") as handle:
                handle.write(example)
        except OSError as exc:
            setup.error(f"cannot create config (existing files are never overwritten): {exc}")
        run_hint = "python evaluate.py"
        if settings.config is not None:
            run_hint += f' --config "{config_path}"'
        print(f"Created {config_path}. Set weights and data_root, then run {run_hint}.")
        return
    try:
        defaults = load_run_config(config_path) if config_path.exists() or settings.config else {}
    except (OSError, ValueError) as exc:
        setup.error(f"cannot load {config_path}: {exc}")
    if not argv and not defaults:
        setup.error("no local config; start with python evaluate.py --init-config")
    cli = parser(defaults)
    args = cli.parse_args(argv)
    try:
        command, env = build_launch(args)
    except ValueError as exc:
        cli.error(str(exc))
    summary = {
        "command": command,
        "environment": {k: v for k, v in env.items()
                        if k.startswith("SPARSE_VGGT_") or k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH")},
        "frame_policy": "full_sequence_after_stride" if args.frame_limit is None else "explicit_limit",
        "allocation": allocation_settings(args),
    }
    if config_path.is_file():
        summary["config"] = {"path": str(config_path),
                             "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    required = [args.weights, args.data_root, args.streamvggt_root / "src/eval/mv_recon",
                args.vggt_root / "vggt"]
    if args.backbone == "pi3":
        required.append(args.pi3_root / "pi3")
    if args.mode != "dense":
        required.append(args.runtime_root / "spas_sage_attn")
    for path in required:
        if not path.exists():
            cli.error(f"missing dependency: {path}; check configs/local.json and docs/setup.md")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        cli.error("output path is not an empty directory; choose a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "launch.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Running {args.backbone}/{args.mode} on GPU {args.gpu}; log: {args.output / 'run.log'}", flush=True)
    with (args.output / "run.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, env=env, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
    if returncode:
        raise SystemExit(f"Evaluation failed ({returncode}); see {args.output / 'run.log'}")
    metrics = args.output / "metrics.json"
    if not metrics.is_file():
        raise SystemExit("Evaluator did not produce metrics.json")
    print(f"Results: {metrics.resolve()}")


if __name__ == "__main__":
    main()
