# Data

This project uses two external datasets, **neither of which is included in this repository**:

## 3D Human Abdomen CT Volume (DICOM)

A 4D-CT scan (10 phases, each phase a stack of ~141 axial slices) of a human abdomen, in DICOM format, organized into folders named `00`, `10`, `20`, ..., `90` (one per respiratory/temporal phase).

**Not included because:**
- It is **clinical medical imaging data**, which should not be redistributed without explicit authorization regardless of any de-identification, and
- The raw volume is large (~990 DICOM slice files across all phases)

**To reproduce this experiment:** you will need your own (appropriately licensed/authorized) 4D-CT DICOM dataset, organized as one subfolder per phase. Point `crea_data.m` (or `crea_data2.m` / `crea_data3.m`) at the relevant phase folder — e.g. `dicomreadVolume('90')` loads the phase-90 subfolder from the current working directory.

## COIL-100 Dataset

The [Columbia Object Image Library (COIL-100)](https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php) — 100 objects, each photographed from 72 angles (5° increments), used here (after subsampling to 60 views and resizing to `180×180×3`) for the 4D tensor generalization experiment.

**Not included because:** it is a ~140 MB dataset of ~7,200 images — too large to include directly in this repository.

**To reproduce this experiment:**
1. Download COIL-100 from the [official Columbia University source](https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php) (public, free for research use)
2. Place the extracted `.png` images in a `coil-100/` folder alongside the scripts
3. Update the path in `trycoil100.m` to point to your local copy
4. Run `trycoil100.m` to build the `180×180×3×60` tensor used by `GRAD_TENSOR4D.m`

## Synthetic Data

No external data is needed for the synthetic-tensor convergence experiment — `synthetic.m` generates the tensor and dictionaries procedurally (random Gaussian dictionaries and core tensor). This experiment can be run immediately with no data setup.
