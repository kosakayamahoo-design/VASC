# Advanced Usage

The normal workflow is one local config and `python evaluate.py`. All paths,
defaults, and switches are resolved into `launch.json` for each run. The config
file path and SHA-256 are recorded when a config is used.

## Config Files

`configs/local.json` is loaded automatically when present. Use
`--config configs/another.json` to select a different JSON file. Relative paths
inside any config are resolved from the VASC repository root; relative paths
passed on the command line retain the usual current-directory meaning.
CLI values override config values. Unknown config keys, incorrect types, and
invalid mode names fail immediately instead of being silently ignored.

`--init-config` creates local settings from `configs/local.example.json` and
never overwrites an existing file. A custom destination is supported:

```bash
python evaluate.py --init-config --config configs/my-run.json
```

The local example contains the common settings. Optional config keys are
`runtime_root`, `output_root`, `output`, `stride`, `scene`, `sequence`,
`frame_limit`, `warmup`, `repeats`, and `pose_only`.
Omit optional settings instead of setting them to `null`. GPU identifiers must
be strings; structural allocation and pose-only settings must be JSON booleans.

Without `output`, a timestamped directory under `output_root` (default `outputs/`)
is chosen automatically. Set `output` only when an exact path is needed; a
nonempty directory is refused. Dry runs do not create output directories.

## Modes

| Mode | Current selection | Cross-layer memory |
| --- | --- | --- |
| `dense` | Full attention | No |
| `topk` | Pooled QK | No |
| `vasc` | QK times value contrast | Yes |

Change `mode` in the config or use `--mode topk`, for example. Keep the
checkpoint, inputs, sparsity, allocation, and reconstruction settings fixed
between sparse controls. Automatic run folders keep their outputs separate.

## Structural Allocation

`--structural-allocation` enables the paper profile; `--no-structural-allocation`
disables it. These override the config's `structural_allocation` boolean. If
neither the config nor CLI specifies a switch, `structural.enabled` in
`configs/paper.json` is used (`true`, so the policy is on by default).
Existing local configs with `structural_allocation: false` remain explicit opt-outs;
set that field to `true` or remove it to use the new default.

The old `--allocation paper` / `--allocation uniform` options remain aliases
for enabling / disabling the policy. Do not combine these aliases with the new
flags. `launch.json` records requested `enabled`, effective `applied`, policy,
and the actual per-layer sparsity ratios. Selection keeps native 128-query /
64-key pooling and two-child pairing. Details are in
[the evaluation protocol](evaluation.md).

## Inputs and Timing

Use `--scene` to select an NRGBD scene. For 7Scenes, selecting one scene also
requires `--sequence`; with neither option, all exposed test sequences run.
`--frame-limit 8` is an explicit short diagnostic, not a full benchmark.

Accuracy defaults use zero warmup and one repeat. For timing, specify
`--warmup` and `--repeats`, use an idle GPU, and compare identical inputs against
dense attention. See [evaluation.md](evaluation.md).

## Legacy CLI

The previous full CLI remains available without a local config:

```bash
python evaluate.py --backbone vggt --mode vasc \
  --dataset nrgbd --data-root /path/to/nrgbd \
  --weights /path/to/vggt/model.pt --output outputs/vggt_s75 \
  --sparsity 0.75 --structural-allocation
```

Use `--help` for the full argument reference. No environment-variable setup is
needed for the selector; the launcher isolates and sets its internal switches.

## Developer Checks

```bash
python -B -m unittest discover -s tests -p "test_*.py" -v
python -B tests/check_selector.py
```

The first command requires only Python; the selector check also needs PyTorch.
After installing CUDA and model dependencies, `python -B tests/gpu_smoke.py`
checks synthetic attention without checkpoints or datasets.
