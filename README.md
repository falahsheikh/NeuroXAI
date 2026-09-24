# NeuroXAI: explainable early Alzheimer's detection for clinics with limited resources

[![tests](https://github.com/falahsheikh/NeuroXAI/actions/workflows/tests.yml/badge.svg)](https://github.com/falahsheikh/NeuroXAI/actions/workflows/tests.yml)
[![DOI](https://img.shields.io/badge/DOI-10.1117%2F12.3088200-blue)](https://doi.org/10.1117/12.3088200)

This repository contains the software for the paper
[NeuroXAI: Lightweight Explainable Early Alzheimer's Detection for Resource-Constrained Clinical Settings](https://doi.org/10.1117/12.3088200)
(SPIE Medical Imaging 2026: Imaging Informatics).

NeuroXAI has two desktop applications:

- The **Slice Viewer** shows MRI volumes in the axial, coronal and sagittal planes.
  It has tools for annotation and measurement.
- The **Analysis Tool** classifies one coronal MRI slice as cognitively normal (CN),
  early mild cognitive impairment (EMCI) or late mild cognitive impairment (LMCI).
  It shows Grad-CAM++, Guided Grad-CAM++ and consensus maps for each prediction.

![Analysis Tool](docs/screenshots/analysis_full_layout.png)

> **Note:** NeuroXAI is research software. It is not a medical device. Do not use it for diagnosis.

## Installation

1. Install Python 3.11 with Tkinter.
   The installers from python.org include Tkinter.
   On Ubuntu, install the `python3-tk` package.
2. Make a virtual environment and activate it:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the tests:

   ```bash
   pytest
   ```

## Slice Viewer

Start the Slice Viewer:

```bash
python app/slice_viewer.py
```

Click **Load Volume** and select a NIfTI (`.nii`, `.nii.gz`) or MetaImage (`.mhd`) file.

| Function | Control |
| --- | --- |
| Change the slice | slice sliders, or the mouse wheel on a view |
| Change the contrast | window and level sliders |
| Zoom and pan | Ctrl + mouse wheel; **Zoom Select** and **Pan** check boxes; **Reset Zoom** (Ctrl+0) |
| Draw on a slice | **Draw** check box; color and brush size on the **Annotations** tab |
| Measure a distance or an area | **Measure** or **Area** check box |
| Save or load a session | **Save Session** (Ctrl+Shift+S), **Load Session** (Ctrl+Shift+O) |
| Save a report with snapshots | **Save Report** (Ctrl+S) |
| Undo or redo | **Undo** (Ctrl+Z), **Redo** (Ctrl+Y) |
| Open the Analysis Tool | **Launch Analysis Tool** |

![Slice Viewer](docs/screenshots/slice_full_layout.png)

## Analysis Tool

Start the Analysis Tool:

```bash
python app/analysis/analysis_tool.py
```

You can also click **Launch Analysis Tool** in the Slice Viewer.

1. Wait until the status bar shows that the model is loaded.
2. Click **Upload Coronal MRI Slice** and select a PNG or JPEG slice.
3. Examine the prediction, the class probabilities and the explanation maps.
4. To use a different convolutional layer, click **Configure XAI Layers**.
5. To keep the results, click **Save Medical Report** and select a folder.

The tool crops the uploaded slice to the brain and puts it on a 224x224 canvas.
To use your own tri-class Keras model, click **Load Custom Model**.

### Explanation methods

- **Grad-CAM++** (Chattopadhay et al., 2018).
  The default layer is the last convolutional layer (`top_conv`).
- **Guided backpropagation** (Springenberg et al., 2015).
  A gradient goes through a rectifier only where its input and the gradient are both positive.
- **Guided Grad-CAM++** is the product of the two maps, smoothed with a Gaussian filter (sigma 0.8).
- The **consensus map** is the mean of the normalized Grad-CAM++ and Guided Grad-CAM++ maps.

Both methods use the class score before the softmax.
A mask from Otsu thresholding keeps each map inside the brain.

## Model

`app/analysis/TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras` is the model from the paper.

| Property | Value |
| --- | --- |
| Architecture | EfficientNetV2B0 (ImageNet weights, frozen), global average pooling, Dense(128, ReLU), Dense(3, softmax) |
| Parameters | 6,083,667 |
| Input | 224 x 224 RGB coronal slice |
| Classes | CN, EMCI, LMCI |
| Training data | coronal slices from ADNI |

`training/training_code.ipynb` is the training notebook for the backbones.
The [eAlz](https://github.com/falahsheikh/eAlz) repository contains a tested version of the training and evaluation code.

## Preprocessing

The ADNI data use agreement does not permit redistribution.
Thus, this repository contains no MRI data.
To use ADNI data, ask for access at [adni.loni.usc.edu](https://adni.loni.usc.edu).

1. Remove the skull with SynthStrip.
   The script uses the `freesurfer/synthstrip` Docker image, or `mri_synthstrip` from FreeSurfer.

   ```bash
   python preprocessing_tools/skull_stripping.py <input_folder> <output_folder>
   ```

2. Put the skull-stripped volumes in one folder for each class: `cn/`, `emci/` and `lmci/`.
3. Make 224x224 coronal PNG slices:

   ```bash
   python preprocessing_tools/slice_extraction.py --input-dir <volumes_folder> --output-dir <slices_folder>
   ```

## Repository contents

```
NeuroXAI/
├── app/
│   ├── slice_viewer.py            Slice Viewer
│   ├── loading_window.py          start screen
│   └── analysis/
│       ├── analysis_tool.py       Analysis Tool
│       └── TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras
├── preprocessing_tools/           skull stripping and slice extraction
├── training/training_code.ipynb   training notebook
├── benchmarks/                    system and requirements checks, timing results
├── docs/screenshots/              screenshots of the applications
└── tests/                         tests with synthetic data
```

## Changes

Version 1.1 (September 2026) corrects two parts of the explanation code:

- The Grad-CAM++ weights now use the equation from Chattopadhay et al. (2018).
- The guided backpropagation now changes the gradient at each rectifier.
  The earlier version masked only the input gradient.

## Citation

```bibtex
@inproceedings{sheikh2026neuroxai,
  title     = {{NeuroXAI}: Lightweight Explainable Early {Alzheimer's} Detection for Resource-Constrained Clinical Settings},
  author    = {Sheikh, Falah and Al Marouf, Ahmed and Alhajj, Reda and Rokne, Jon George},
  booktitle = {Medical Imaging 2026: Imaging Informatics},
  series    = {Proc. SPIE},
  volume    = {13930},
  pages     = {139301R},
  year      = {2026},
  doi       = {10.1117/12.3088200}
}
```

## License

[CC BY-NC 4.0](LICENSE)

## Acknowledgments

The data for this work came from the ADNI database (adni.loni.usc.edu).
An Alberta Innovates Summer Research Studentship supported this work.
