# Tensor Dictionary Learning with Sparse Tucker Decomposition

A MATLAB implementation of **Tensor Dictionary Learning** based on the **Tucker decomposition**, with sparsity imposed on the core tensor, following Zhang et al. (2013), *"Tensor Dictionary Learning with Sparse Tucker Decomposition."*

> Group project — co-authored with Alberto Cozzani (University of Bologna).

## Project Overview

Given a 3-way (or higher-order) tensor **Y**, the goal is to learn a **Tucker decomposition**

```
Y ≈ X ×₁ A ×₂ B ×₃ C
```

where **A, B, C** are mode-wise dictionaries and **X** is a **sparse core tensor**. Imposing sparsity on the core tensor serves two purposes: **compressing** the data (the core needs far fewer non-zero entries than the full tensor) and **clarifying the relationships** between the learned dictionaries.

The learning problem is solved as a **two-stage alternating process**:

1. **Sparse coding** — given the current dictionaries, find the sparse core tensor **X** via **Tensor Orthogonal Matching Pursuit (TOMP)**
2. **Dictionary update** — given the current core tensor, update the dictionaries **A, B, C** via **Gradient Descent**

These two stages alternate until convergence, in the algorithm referred to as **GRADTENSOR**.

## Mathematical Background

### Problem Formulation

```
min_{X,A,B,C} ‖Y − X ×₁ A ×₂ B ×₃ C‖²_F
```

subject to **X** having at most `tmax = s₁·s₂·s₃` non-zero entries, where `sₙ` is the sparsity level along mode `n`.

### Tensor-OMP (Sparse Coding Stage)

Using the equivalence `y = (C ⊗ B ⊗ A)x` (vectorized Tucker form via Kronecker products), Tensor-OMP greedily selects, at each iteration, the mode-1/2/3 dictionary atoms `(j₁, j₂, j₃)` that maximize the projection of the current residual:

```
[j₁, j₂, j₃] = argmax_{j₁,j₂,j₃} ‖ R ×₁ A(:,j₁)ᵗ ×₂ B(:,j₂)ᵗ ×₃ C(:,j₃)ᵗ ‖
```

adds these indices to the active sets `M₁, M₂, M₃`, solves a least-squares problem for the corresponding core entries, and updates the residual `R = Y − E ×₁ A(:,M₁) ×₂ B(:,M₂) ×₃ C(:,M₃)`. This repeats until either the sparsity budget `tmax` or a maximum iteration count is reached.

### Gradient Descent (Dictionary Update Stage)

With the core tensor **X** fixed, the objective unfolds mode-wise into three separate least-squares problems:

```
F_A = ‖Y₍₁₎ − A·X₍₁₎·(C⊗B)ᵗ‖²_F     F_B = ‖Y₍₂₎ − B·X₍₂₎·(C⊗A)ᵗ‖²_F     F_C = ‖Y₍₃₎ − C·X₍₃₎·(B⊗A)ᵗ‖²_F
```

Each dictionary is updated by Gradient Descent on its corresponding objective, holding the other two dictionaries fixed, until the update step falls below a tolerance.

### GRADTENSOR (Full Algorithm)

Alternates Tensor-OMP (sparse coding) and Gradient Descent (dictionary update) until the overall reconstruction error falls below a tolerance `ε₂`.

## Experiments

### 1. Synthetic Tensor (Convergence Study)

A synthetic tensor `Y ∈ R^(100×100×100)` is generated from randomly-initialized dictionaries `A, B, C ∈ R^(In×Mn)` (`Mₙ = 150`, `Iₙ = 100`) and a Gaussian-distributed sparse core tensor (mode sparsity `μ = 1/6`). Step size `γ = 0.3`, tolerances `ε₁ = 10⁻⁶`, `ε₂ = 10⁻⁴`. Used to verify the convergence behavior of GRADTENSOR (residual error vs. iterations, and vs. `tmax`).

### 2. 3D Human Abdomen CT Reconstruction

GRADTENSOR is applied to a 3-mode CT volume tensor (`I₁=151, I₂=125, I₃=141`, `Mₙ = 1.5·Iₙ`), at multiple sparsity levels on the core tensor, to assess practical image-reconstruction performance on real (clinical) medical imaging data.

### 3. Generalization to 4D — COIL-100 Dataset

The algorithm is generalized to a **4-way** tensor (`efficientTOMP4d`, `GRAD_TENSOR4D`) and applied to the public **COIL-100** object-image dataset, reshaped into a tensor of size `180×180×3×60` (height × width × RGB channels × object views). Reconstructions are shown at different core sparsity levels (e.g. ~6.25% and ~20% non-zero core entries), comparing the original vs. reconstructed images.

## Project Structure

```
.
├── README.md                          # This file
├── docs/
│   ├── README.md
│   └── Report_Tensor_Dictionary_Learning.pdf   # Presentation slides (theory, algorithms, results)
├── src/
│   ├── README.md
│   ├── TOMP.m                          # Tensor-OMP (3-way)
│   ├── efficientTOMP.m / 2 / 3 / 4d.m   # Optimized / generalized Tensor-OMP variants
│   ├── GradTensor.m                    # Gradient Descent dictionary update (3-way)
│   ├── GRAD_TENSOR.m                   # Full GRADTENSOR algorithm (3-way)
│   ├── GRAD_TENSOR4D.m                 # Full GRADTENSOR algorithm (4-way)
│   ├── GRAD_TENSOR_NICO.m              # Alternative dictionary-update implementation
│   ├── GradTensorEff.m                 # Optimized Gradient Descent variant
│   ├── overcomplete_dict.m             # Utility: pads a dictionary to an overcomplete size
│   ├── ttm.m / check_tensor_operations.m  # Tensor-times-matrix utilities / sanity checks
│   ├── crea_data.m / crea_data2.m / crea_data3.m  # CT volume loading & preprocessing
│   ├── synthetic.m                     # Synthetic tensor generator (Experiment 1)
│   ├── trycoil100.m                    # COIL-100 data loading (Experiment 3)
│   ├── Dicom_images.m / MedicalImages.m / viewData.m  # DICOM volume loading & 3D visualization
│   ├── mandelbrotto.m / fract.m / sierpisnki_tensor.m  # Exploratory fractal-tensor generators
│   └── (see src/README.md for full descriptions)
│
├── results/
│   ├── README.md
│   ├── residual_error.png              # Convergence plot: residual vs. iterations
│   ├── residual_errortmax.png          # Convergence plot: residual vs. tmax
│   ├── All_COIL100_Images.jpg          # COIL-100 dataset montage
│   └── Reconstructed_images/           # CT reconstruction results (4 examples)
│
└── data/
    └── README.md                       # Notes on the (not included) raw datasets
```

> **Note on data:** the raw 3D Human Abdomen CT volumes (DICOM, clinical data) and the full COIL-100 image dataset (~7,200 images, ~140 MB) are **not included** in this repository — see `data/README.md` for details and how to obtain/regenerate them.

## How to Use

### Prerequisites

- MATLAB R2019a or later
- **MATLAB Tensor Toolbox** (provides `tensor`, `sptensor`, `ttm`, `unfold`/mode-n unfolding — required by essentially all scripts in this repository)
- Image Processing Toolbox (`dicomreadVolume`, `imresize3`, `medicalVolume`) for the CT-reconstruction experiment

### Running the Synthetic Convergence Experiment

```matlab
synthetic        % generates the synthetic tensor and dictionaries
% then call GRAD_TENSOR with the generated Y, A_0, B_0, C_0
```

### Running the CT Reconstruction Experiment

```matlab
crea_data        % loads and preprocesses the abdomen CT volume (requires local DICOM data)
% then call GRAD_TENSOR on the resulting tensor Y
```

### Running the 4D COIL-100 Experiment

```matlab
trycoil100       % loads and reshapes the COIL-100 images into a 4-way tensor
% then call GRAD_TENSOR4D (using efficientTOMP4d internally)
```

## Author

**Niccolò Chiari**, with Alberto Cozzani  
University of Bologna

## References

- Zhang, T., et al. (2013). "Tensor Dictionary Learning with Sparse TUCKER Decomposition." *17th International Conference on Digital Signal Processing (DSP)*.

## License

Educational project. Available for academic use.

---

For the full theoretical background (tensor preliminaries, algorithm derivations, all experimental figures), see `docs/Report_Tensor_Dictionary_Learning.pdf`.
