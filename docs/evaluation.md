# Evaluation Protocol

- Backbones: VGGT and Pi3, using their respective pretrained checkpoints.
- Benchmarks in this entry point: 7Scenes and NeuralRGBD.
- Inputs: stride 10 by default; no implicit frame cap. Use `--scene` and
  `--sequence` for a particular 7Scenes sequence. With neither, evaluate all
  test sequences exposed by the dataset loader.
- Main sparsities: 65%, 75%, 85%; 55% is also supported.
- Metrics: ATE, translational RPE, rotational RPE, reconstruction accuracy,
  completeness, and normal consistency. Pi3 keeps its native global-point
  alignment protocol; it does not use the VGGT reconstruction criterion.
- Point-cloud caps match the frozen launchers: Pi3 200,000; VGGT NeuralRGBD
  999,999; VGGT 7Scenes has no explicit point cap.
- Full VASC combines value-aware demand with cross-layer execution feedback.
  Dense and pooled-QK Top-K are the available reference modes.

## Layer Allocation

Structural allocation is optional and independent of the selection mode.
Use `--structural-allocation` to enable it or `--no-structural-allocation` to
disable it. The run config's `structural_allocation` field supplies the default;
if omitted, `structural.enabled` in `configs/paper.json` is used. Both shipped
defaults are `true`, so sparse runs use the paper layer profile when applicable.
Dense runs ignore the profile and stay dense.

When enabled at sparsity above 0.70, six favored global layers receive ratio `s - 0.05`.
Other layers receive `s + 6 * 0.05 / (L - 6)`.
VGGT uses 24 global layers and indices 11--16; Pi3 uses 18 global layers and
indices 12--17, both zero-based. The nominal mean ratio remains `s`; integer
rounding determines the realized block counts.

When disabled, every global layer receives sparsity `s`. At or below 0.70,
allocation is uniform even with the switch enabled. This changes neither the
value/memory switches nor special-token protection. `launch.json` records both
`enabled` (requested) and `applied` (effective), the effective policy, and all
per-layer sparsities; the same information is available with `--dry-run`.

For baseline comparisons, keep the allocation, checkpoint, input stride,
sparsity, and reconstruction settings fixed. Legacy
`--allocation paper` and `--allocation uniform` are equivalent aliases; they
cannot be combined with the new options.

## Timing

Defaults (`--warmup 0 --repeats 1`) are for accuracy evaluation. For timing,
specify warmup and repeats explicitly, use an otherwise idle GPU, and compare
identical inputs and execution settings against `--mode dense`. Do not treat
single-pass accuracy-run latency as the paper speedup measurement.

## Scope

This directory packages the main VGGT/Pi3 implementation and reference
modes. ScanNet-specific preprocessing and the FastVGGT/AVGGT adaptation
experiments are not exposed by this first release entry point.
