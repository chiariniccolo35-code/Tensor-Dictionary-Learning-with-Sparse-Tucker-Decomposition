# Source Code

All MATLAB functions/scripts implementing the Tensor Dictionary Learning algorithm and its experiments. All tensor operations rely on the **MATLAB Tensor Toolbox** (`tensor`, `sptensor`, `ttm`, mode-n `unfold`).

## Core Algorithm — Sparse Coding (Tensor-OMP)

### TOMP.m
```matlab
[E, M1, M2, M3, r] = TOMP(A, B, C, Y, tmax, eps, itermax)
```
The baseline 3-way Tensor Orthogonal Matching Pursuit implementation, as described in the report's Algorithm 1. Greedily selects mode-1/2/3 dictionary indices that maximize the residual's projection, solves a small least-squares problem for the corresponding sparse core entries, and iterates until the sparsity budget `tmax` or `itermax` is reached, or the residual drops below `eps`.

### efficientTOMP.m / efficientTOMP2.m
Optimized variants of `TOMP.m`, reducing redundant tensor operations inside the main loop (e.g. computing `max` along single dimensions instead of over the full flattened tensor) for better performance on larger tensors.

### efficientTOMP3.m
A variant that runs **two parallel TOMP searches** simultaneously (with two independent dictionary sets `A,B,C` and `A2,B2,C2`), used for comparison/ablation experiments between two dictionary initializations.

### efficientTOMP4d.m
The **4-way generalization** of Tensor-OMP, adding a fourth mode/dictionary `D`. Used for the COIL-100 experiment (tensor of size `180×180×3×60`).

## Core Algorithm — Dictionary Update (Gradient Descent)

### GradTensor.m
The basic Gradient Descent routine that updates a single mode's dictionary, holding the core tensor and the other dictionaries fixed (Algorithm in report's "Gradient Descent" slide).

### GRAD_TENSOR.m
```matlab
[X, A, B, C, residual] = GRAD_TENSOR(Y, tmax, gamma, eps_1, eps_2, A_0, B_0, C_0, max_iter, itrs2, decay)
```
The **full GRADTENSOR algorithm** (3-way): alternates calling Tensor-OMP (sparse coding) and Gradient Descent (dictionary update for A, B, C in turn) until the reconstruction residual falls below `eps_2`. This is the main entry point for the synthetic and CT-reconstruction experiments.

### GRAD_TENSOR4D.m
The **4-way generalization** of `GRAD_TENSOR.m`, with a fourth dictionary `D` and using `efficientTOMP4d` internally. Main entry point for the COIL-100 experiment.

### GRAD_TENSOR_NICO.m
An alternative dictionary-update implementation taking the core tensor `X` as a fixed input (rather than re-computing it via TOMP internally), useful for isolating/debugging the Gradient Descent stage independently of the sparse-coding stage.

### GradTensorEff.m
An optimized Gradient Descent variant supporting a **step-size decay schedule** (`gamma` is multiplied by `decay` after 80% of `itrs_max` iterations), improving convergence stability in later iterations.

## Utilities

### overcomplete_dict.m
```matlab
U = overcomplete_dict(U, Mn)
```
Pads a dictionary `U` to have `Mn` columns (if `Mn` exceeds the current dictionary size) by duplicating randomly-selected existing columns — used to construct **overcomplete** dictionaries (`Mₙ > Iₙ`) as required by the Tucker formulation.

### ttm.m
A short script demonstrating/testing the tensor-times-matrix operation (`ttm`) and the index-selection logic used inside TOMP, useful as a minimal reference/debugging snippet.

### check_tensor_operations.m
```matlab
check_tensor_operations(Y, X, A0, B0, C0, itrs_max)
```
A sanity-check utility comparing two equivalent ways of computing a mode-1 unfolding-based reconstruction (`ttm`-based vs. direct matrix multiplication), used to verify the tensor algebra identities relied upon elsewhere in the codebase.

## Data Loading & Preprocessing

### crea_data.m / crea_data2.m / crea_data3.m
Load a 3D Human Abdomen CT volume via `dicomreadVolume`, resize it to `[151, 125, 141]` via `imresize3`, and rescale intensities to `[0, 1]`, preparing the tensor `Y` for the CT-reconstruction experiment. The three variants differ in minor downstream processing/dictionary-initialization choices used across different experimental runs.

### Dicom_images.m
An interactive script (prompts the user for a volume index 0–9) for browsing/visualizing individual slices from the DICOM dataset.

### MedicalImages.m
Loads and initializes dictionaries directly from a specific DICOM volume (`'40'`), used as the entry point for one of the CT-reconstruction experimental runs.

### viewData.m
Visualizes a DICOM volume as a 3D rendered volume (via `medicalVolume`, with a custom alpha/colormap transfer function tuned for abdominal CT Hounsfield-unit ranges).

### trycoil100.m
Loads and reshapes the COIL-100 dataset images into the 4-way tensor (`180×180×3×60`) used in the 4D generalization experiment.

## Exploratory / Auxiliary Scripts

These were used for exploratory testing of the tensor machinery on procedurally-generated data, rather than being part of the main reported experiments:

- **synthetic.m** — generates the synthetic random tensor and dictionaries for the convergence-study experiment (Experiment 1 in the main README)
- **mandelbrotto.m** — generates a 3D tensor by stacking Mandelbrot-set renderings at different zoom levels (exploratory test data with rich multi-scale structure)
- **fract.m** — a fractal-geometry helper function (computes determinants/areas from a coordinate set), used in conjunction with the fractal tensor generators
- **sierpisnki_tensor.m** — generates a 3D tensor by stacking Sierpinski-triangle renderings at different recursion/zoom levels (similar exploratory purpose to `mandelbrotto.m`)

## Dependencies Summary

| Function | Requires |
|---|---|
| All TOMP / GRAD_TENSOR variants | MATLAB Tensor Toolbox (`tensor`, `sptensor`, `ttm`, `unfold`) |
| `crea_data*.m`, `MedicalImages.m`, `Dicom_images.m` | Image Processing Toolbox (`dicomreadVolume`, `imresize3`) |
| `viewData.m` | Image Processing Toolbox (`medicalVolume`) |
| `trycoil100.m` | Local copy of the COIL-100 dataset (see `data/README.md`) |
