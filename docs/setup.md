# Dependency Setup

The model backbones and StreamVGGT evaluation dependencies are installed
separately; their weights and datasets retain their original terms.

Expected layout:

```text
external/vggt/vggt/
external/Pi3/pi3/
external/StreamVGGT/src/eval/mv_recon/
external/StreamVGGT/src/dust3r/
external/StreamVGGT/src/croco/
```

Obtain the repositories from the upstream links in the README. Follow their
installation instructions, including StreamVGGT submodules and dataset
preparation. The evaluation adapters use StreamVGGT's `SevenScenes`, `NRGBD`,
reconstruction criterion, and point-cloud metric utilities. Install those
dependencies in the same Python environment as VASC.

Install CUDA-enabled PyTorch before building `third_party/spargeattn`.
The research environment used Python 3.10 and PyTorch 2.3.1+cu121.
From the VASC directory, install the package and runtime:

```bash
python prepare_runtime.py
python -m pip install -e .
TORCH_CUDA_ARCH_LIST="8.6" python -m pip install --no-build-isolation -e third_party/spargeattn
```

`prepare_runtime.py` expands `third_party/spargeattn.zip` and checks every file
against `third_party/runtime_manifest.json`, including the frozen hashes in
`source_manifest.json`. It is safe to rerun and refuses to overwrite
modified runtime sources. The expanded directory is ignored by Git; retain the
ZIP in the repository. CUDA compilation still happens during the pip install.

Use architecture `8.6` for A6000 or `8.9` for RTX 4090. Checkpoints and datasets
are not included. Initialize local settings with `python evaluate.py --init-config`,
then set checkpoint, data, and repository paths in `configs/local.json` once.
Daily use requires only `python evaluate.py`.

The bundled source comes from the runtime used by the frozen experiments, not
an arbitrary current upstream kernel. Both `_qattn` and `_fused` extensions are
needed. Use `MAX_JOBS=2` to limit build concurrency when required.

The build selects SM86/SM89 FP16 attention instantiations. FP8 and Hopper paths
are not enabled by this build. Python/CUDA version mismatches require rebuilding
the extensions, not copying `.so` files from another machine.

Smoke-check the installed runtime:

```bash
python -c "import spas_sage_attn._qattn; import spas_sage_attn._fused"
python -c "import sparse_vggt; print(sparse_vggt.__file__)"
```

Upstream revisions from the original run machines were exported without Git
metadata. Exact external commit IDs are not yet established. Source hashes for
the bundled backend are recorded in `source_manifest.json`; this does not pin
the separately installed model/data dependencies.
