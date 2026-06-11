<div align="center">

# Dual-Tiered Microglia–pSyn Analysis Framework

**A Modular Framework for 3D Spatial Point Process Analysis and Explainable Deep Learning in Fluorescence Microscopy**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<img src="assets/microglia_preview.gif" width="400" alt="Microglia Analysis Framework">

</div>

---

## Overview

This repository provides a reusable computational framework for analyzing spatial relationships and cellular morphology in large-scale 3D fluorescence microscopy datasets.

The framework combines two complementary analysis tiers:

### Spatial Point Process Analysis

A collection of statistical methods for quantifying spatial organization, clustering, proximity, and interaction structure between biological objects in three-dimensional space.

### Explainable Deep Learning

A deep-learning pipeline for learning morphology-aware representations from volumetric image data and visualizing model attention through Grad-CAM–based interpretability techniques.

Although the original implementation was developed for studying microglia and phosphorylated α-synuclein (pSyn) in Parkinson's disease tissue, the underlying architecture is intentionally modular and can be adapted to other microscopy datasets, biomarkers, disease models, and spatial biology applications.

The framework is designed to serve both as:

* a complete end-to-end analysis pipeline;
* a collection of reusable modules that can be integrated into custom workflows.

---

## Key Features

### 3D Object Extraction

* Connected-component labeling for volumetric microscopy data
* Object centroid and volume extraction
* Physical-space coordinate conversion
* Per-object metadata generation
* Batch processing across large cohorts

### Spatial Statistics

* Nearest-neighbor distance analysis
* Bivariate K-function analysis
* Mark-weighted K-functions
* Monte Carlo simulation testing
* Global envelope testing
* Mixed-effects and permutation-based statistical comparisons

### Deep Learning

* Multi-channel 3D classification pipelines
* Group-aware cross-validation
* Automated intensity normalization
* Explainable AI via Grad-CAM
* Area-of-Relevance (AOR) analysis
* Patient-level performance aggregation

### Microscopy Workflow Support

* OME-Zarr support
* Large-scale dataset handling
* SSD and RAM caching
* Data augmentation pipelines
* Interactive 3D visualization with Napari

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
│
├── DL
│   ├── zarr_patch_dataset.py
│   ├── intensity_normalization.py
│   ├── train_unified_5fold.py
│   ├── train_final_and_gradcam.py
│   └── test_CAM.py
│
└── SPP
    ├── cc3d_utils.py
    ├── cc3d_production.py
    ├── cc3d_test_napari.py
    └── spp_analysis.py
```

---

## Workflow

```text
3D Microscopy Data
        │
        ▼
Connected Component Extraction
        │
        ├──────────────► Spatial Point Process Analysis
        │
        ▼
Patch Extraction
        │
        ▼
Deep Learning Models
        │
        ▼
Grad-CAM Interpretation
```

---

## Infrastructure Notes

The reference implementation uses:

* ClearML for experiment tracking and orchestration
* MinIO (S3-compatible storage) for dataset management
* OME-Zarr for volumetric image storage

The framework can be adapted to alternative infrastructure, storage backends, and deployment environments. Users running the code locally or on different servers may need to update storage configuration, experiment-management settings, credentials, and data-access paths to match their own setup.

---

## Installation

```bash
git clone https://github.com/Yameteshka/spatial-dl-cell-protein.git
cd spatial-dl-cell-protein
pip install -r requirements.txt
```

---

## Typical Workflow

### 1. Extract 3D Objects

```bash
python cc3d_production.py
```

Generate connected-component statistics, centroids, and object metadata.

### 2. Run Spatial Analysis

```bash
python spp_analysis.py
```

Compute spatial interaction metrics and statistical comparisons.

### 3. Train Deep Learning Models

```bash
python train_unified_5fold.py
```

Perform cross-validation and model evaluation.

### 4. Generate Explainability Maps

```bash
python test_CAM.py
```

Produce Grad-CAM visualizations and activation maps.

### 5. Inspect Results in 3D

```bash
python cc3d_test_napari.py
```

Visualize connected components and extracted objects interactively.

---

## Applications

The framework can be adapted for:

* Neurodegeneration research
* Spatial biology
* Histopathology
* Multiplex fluorescence microscopy
* Cellular interaction analysis
* Biomarker colocalization studies
* Explainable biomedical AI

---

## Dependencies

Core dependencies are listed in `requirements.txt`.

Main packages include:

* PyTorch
* NumPy
* Pandas
* SciPy
* Scikit-image
* Zarr
* Napari
* Statsmodels
* Matplotlib
* Seaborn
* ClearML
* S3FS
* AIOHTTP

---

## License

This project is released under the MIT License.

See the LICENSE file for details.
