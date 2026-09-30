# VASC

**Value-Aware Sparse Attention with Cross-Layer Memory**

Training-free sparse attention for VGGT and Pi3. Value contrast guides current
selection; cross-layer memory tracks unserved demand. Structural allocation is
an independent layer-budget policy, enabled by default and optional to disable.

## Quick Start

Sparse inference requires Linux, an NVIDIA GPU, and a CUDA-enabled environment.
Complete [dependency setup](docs/setup.md) once, then initialize local settings:

```bash
python evaluate.py --init-config
```

The bundled CUDA sources are stored in `third_party/spargeattn.zip` to keep
browser uploads small. Dependency setup starts with `python prepare_runtime.py`;
it verifies and expands the exact source snapshot without downloading anything.

In `configs/local.json`, set `weights` and `data_root` to your checkpoint and
dataset. Adjust repository paths only if they are not under `external/`.
Relative config paths are resolved from the VASC directory, not your shell's
working directory. Local settings are ignored by Git.

Run the evaluation:

```bash
python evaluate.py
```

Each run gets a new folder under `outputs/` containing `metrics.json`, `run.log`,
and `launch.json`. Existing results are never overwritten. Use
`python evaluate.py --dry-run` to inspect settings without loading a model.

## Settings

Edit these fields in `configs/local.json`; CLI overrides remain optional.

| Setting | Values / meaning |
| --- | --- |
| `backbone` | `vggt` or `pi3`; use the matching checkpoint |
| `dataset` | `nrgbd` or `7scenes`; use the matching data root |
| `mode` | `vasc` (default), `dense`, or `topk` |
| `sparsity` | Fraction removed; `0.75` targets 25% retention |
| `structural_allocation` | `true` (default): paper layer profile; `false`: uniform layers |
| `gpu` | GPU index as a string, e.g. `"0"` |

Structural allocation is enabled by default and takes effect only above 70%
sparsity, using the paper layer profile. This leaves
value selection, memory, and special-token protection unchanged. Dense mode
always remains dense. Stride defaults to 10; there is no implicit frame cap.

To disable structural allocation for one run without editing the file:

```bash
python evaluate.py --no-structural-allocation
```

## Details

- [Advanced usage](docs/usage.md)
- [Metric, allocation, and timing protocol](docs/evaluation.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

The public entry point is `evaluate.py`; `evaluation/` and `src/sparse_vggt/`
contain the inference backends. Historical research scripts, paper sources, model
weights, and datasets are not bundled. DA3/DINOv2 transfer diagnostics remain
separate experiments and are not exposed as supported backbones here.

Built on [VGGT](https://github.com/facebookresearch/vggt),
[Pi3](https://github.com/yyfz/Pi3),
[SpargeAttn](https://github.com/brianwang00001/SpargeAttn), and
[StreamVGGT](https://github.com/wzzheng/StreamVGGT).
