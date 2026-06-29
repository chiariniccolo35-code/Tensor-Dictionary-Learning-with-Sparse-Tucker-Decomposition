# Documentation

## Report_Tensor_Dictionary_Learning.pdf

A presentation (slide deck), co-authored with Alberto Cozzani, covering:

1. **Prerequisites** — mode-n unfolding, the n-mode product, Tucker decomposition, the HOSVD theorem
2. **Purpose of the paper** — motivation for sparse Tucker decomposition (compression, dictionary-relationship clarity); overview of the two-stage TOMP + Gradient Descent learning scheme
3. **Problem formulation** — the constrained Frobenius-norm minimization defining the tensor dictionary learning problem
4. **Tensor-OMP** — derivation via the vectorized Kronecker-product form, and the full algorithm (Algorithm 1)
5. **Gradient Descent** — the mode-wise unfolded objectives `F_A, F_B, F_C` and the per-dictionary update algorithm
6. **GRADTENSOR** — the full alternating algorithm (Algorithm 2) combining Tensor-OMP and Gradient Descent
7. **Experiments and results**
   - Synthetic tensor: convergence behavior (residual vs. iterations, residual vs. `tmax`)
   - 3D Human Abdomen CT: image reconstruction at multiple sparsity levels
   - 4D generalization: COIL-100 dataset, reconstructed images at ~6.25% and ~20% core sparsity

## Reading Guide

| If you want to understand... | Read... |
|---|---|
| Tensor algebra preliminaries (unfolding, n-mode product, Tucker, HOSVD) | Report, "Prerequisites" section |
| Why sparsity is imposed on the core tensor | Report, "Purpose of the paper" |
| The Tensor-OMP algorithm in detail | Report, "Tensor OMP" + "Algorithm 1" |
| The dictionary-update (Gradient Descent) step | Report, "Gradient Descent" section |
| How the two stages combine | Report, "Algorithm 2: GRADTENSOR" |
| The synthetic convergence results | Report, "Experiments and results" (synthetic tensor) |
| The CT reconstruction results | Report, "Image reconstruction" section |
| The 4D / COIL-100 generalization | Report, "Generalization to 4 dimensions" |

## Code ↔ Report Mapping

| Report Section | Code |
|---|---|
| Algorithm 1 (Tensor-OMP) | `src/TOMP.m`, `src/efficientTOMP.m`, `src/efficientTOMP2.m` |
| Gradient Descent (dictionary update) | `src/GradTensor.m`, `src/GradTensorEff.m`, `src/GRAD_TENSOR_NICO.m` |
| Algorithm 2 (GRADTENSOR, full) | `src/GRAD_TENSOR.m` |
| Synthetic tensor experiment | `src/synthetic.m` + `src/GRAD_TENSOR.m` |
| CT reconstruction experiment | `src/crea_data.m` / `crea_data2.m` / `crea_data3.m` + `src/GRAD_TENSOR.m` |
| 4D / COIL-100 generalization | `src/trycoil100.m` + `src/efficientTOMP4d.m` + `src/GRAD_TENSOR4D.m` |

## Authors

**Niccolò Chiari**, Alberto Cozzani — University of Bologna.
