# Results

Figures from the experiments described in the report.

## Synthetic Tensor — Convergence Study

**`residual_error.png`** — residual error vs. number of GRADTENSOR iterations, on the synthetic `100×100×100` tensor experiment (`γ = 0.3`, `ε₁ = 10⁻⁶`, `ε₂ = 10⁻⁴`). Confirms the alternating Tensor-OMP / Gradient-Descent scheme converges.

**`residual_errortmax.png`** — residual error vs. `tmax` (the sparsity budget on the core tensor), showing how reconstruction accuracy improves as more non-zero core entries are allowed.

## 3D Human Abdomen CT Reconstruction

**`Reconstructed_images/`** — four example reconstructions from the CT-volume experiment (`I₁=151, I₂=125, I₃=141`, `Mₙ=1.5·Iₙ`), at the sparsity level `μ₁ = 1/2.2` discussed in the report (additional sparsity levels `μ₂` are shown in the report PDF but not all are saved as separate image files here):

- `advil.png`
- `macchinina2.png`
- `pera.png`
- `pomodori.png`

> Note: despite the object-like filenames (a holdover from the original experiment naming), these are abdomen CT slice reconstructions, not photographs of the named objects.

## COIL-100 — 4D Generalization

**`All_COIL100_Images.jpg`** — a montage of the COIL-100 dataset images used for the 4-way tensor generalization experiment (`180×180×3×60` tensor).

The original-vs-reconstructed comparison figures (at ~6.25% and ~20% non-zero core entries) referenced in the report are embedded directly in `docs/Report_Tensor_Dictionary_Learning.pdf` and are not provided as separate image files in this folder.

## Interpreting the Results

- **Residual error** is computed as `‖Y − reconstruction‖_F`, normalized by `‖Y‖_F`
- Lower sparsity (fewer non-zero core entries) → higher compression, lower reconstruction fidelity
- The CT and COIL-100 experiments both demonstrate that even a **highly sparse** core tensor (as low as ~6.25% non-zero entries in the COIL-100 case) can still produce visually recognizable reconstructions — the central claim motivating the sparse-Tucker approach

## References

For the full set of figures (including all sparsity-level comparisons for the CT volume and additional COIL-100 examples), see `docs/Report_Tensor_Dictionary_Learning.pdf`.
