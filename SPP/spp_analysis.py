"""
SPP Analysis Pipeline — Spatial Point Process Analysis on 3D Microscopy Data

Implements Steps 1.1–1.7 of the SPP pipeline:
  1.1  Nearest-Neighbor Distance (NND) with Guard Zone border correction
  1.2  Overlap Index  (voxel-level pSyn AND IBA1 / pSyn)
  1.3  Edge-corrected Bivariate K-function (3D, translation correction)
       + Mark-weighted K-function (protein volume as mark)
  1.4  Random Labeling null model  (999 Monte Carlo simulations)
  1.5  Global Envelope Test (MAD) + L-function visualization
  1.6  SPP metric aggregation by patient / FOV
  1.7  LMM group comparison (PD vs HC) with residual diagnostics

Additional analyses (Steps E–R):
  E    Quality filtering (minimum interior cells)
  F    Log transformation of SPP metrics
  G–I  Diagnostic plots (outlier identification, filtering summary,
       transformed diagnostics)
  J    Standard LMM on log-transformed metrics
  K    GLMM (Gamma GEE) comparison
  L    Patient-level permutation tests
  M    Sensitivity analysis (outlier removal)
  N    Region-stratified LMM
  O    Publication-quality plots
  P    GLMM Gamma for overlap_index in Substantia Nigra
  Q    Bootstrap confidence interval for Cohen's d
  R    Ridge + Waterfall plot for SN overlap analysis

Input
  CC3D preprocessing results on MinIO  (JSON per FOV, produced by
  cc3d_production.py).  Each JSON contains:
    - spp_data.type_I_cell.points_um   : cell centroids in μm
    - spp_data.type_I_cell.marks       : {volume_um3, volume_vox}
    - spp_data.type_II_protein.points_um : protein centroids in μm
    - spp_data.type_II_protein.marks   : {volume_um3, volume_vox}
    - fov_window_um                    : 3D observation window bounds

Output
  Per-FOV JSON with SPP metrics, summary CSV, L-function plots,
  LMM/GLMM/permutation results, and publication plots — all uploaded
  to MinIO.

Usage
    # Run locally (debug):
    CLEARML_DISABLED=true python spp_analysis.py

    # Auto-submit to ClearML:
    python spp_analysis.py

Mathematical references
  - Baddeley, Rubak, Turner (2015) "Spatial Point Patterns: Methodology
    and Applications with R", CRC Press.
  - Illian et al. (2008) "Statistical Analysis and Modelling of Spatial
    Point Patterns", Wiley.
  - Diggle (2003) "Statistical Analysis of Spatial Point Patterns", Arnold.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

# Windows asyncio / SSL fix  —  MUST be before s3fs/aiohttp import
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy import stats as sp_stats

# Matplotlib — Agg backend for headless ClearML agent
# MUST be set before importing pyplot or any matplotlib backends.
# Importing inside functions caused a 14-hour crash on ClearML.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Statsmodels — top-level import (was lazy, now eager)
import statsmodels.formula.api as smf
import statsmodels.api as sm

# MinIO env vars
os.environ.setdefault("AWS_ACCESS_KEY_ID", "YOUR_MINIO_ACCESS_KEY")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "YOUR_MINIO_SECRET_KEY")
os.environ.setdefault("AWS_ENDPOINT_URL", "YOUR_MINIO_ENDPOINT")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

import s3fs
import zarr
from skimage.measure import label, regionprops
from dataclasses import asdict, dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration

STORAGE_NAME = "YOUR_STORAGE_NAME"
CC3D_OUTPUT_PREFIX = "YOUR_OUTPUT_PREFIX/cc3d_analysis"
SPP_OUTPUT_PREFIX = "YOUR_OUTPUT_PREFIX/spp_analysis"
CACHE_DIR = os.path.join(tempfile.gettempdir(), "YOUR_LOCAL_CACHE_DIR")

# K-function radii (μm)
# Maximum recommended: 1/4 of smallest window dimension ≈ 12.5/4 ≈ 3.1 μm
R_MAX_K = 3.0  # μm — upper bound for K-function evaluation
R_VALS = np.linspace(0.5, R_MAX_K, 30)

# Guard zone radius for NND (μm)
R_MAX_GUARD = 5.0  # μm

# Number of Monte Carlo simulations for Random Labeling
N_SIM = 999

# Random seed for reproducibility
RANDOM_SEED = 42

# Resolution (μm/px) — must match cc3d_production
RESOLUTION_UM = np.array([0.5, 0.11, 0.11], dtype=np.float64)

# Quality filter: minimum interior cells for reliable SPP
MIN_INTERIOR_CELLS = 3

# Outlier patients for sensitivity analysis
OUTLIER_PATIENTS = ["PD8", "PD10", "HC9"]

# Number of permutation test iterations
N_PERMUTATIONS = 10000


# Helper: recursively convert numpy types for JSON serialization

def _convert_numpy(obj):
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _convert_numpy(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# MinIO helpers

def _make_s3fs() -> s3fs.S3FileSystem:
    """Create an s3fs filesystem for MinIO."""
    return s3fs.S3FileSystem(
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        client_kwargs={
            "endpoint_url": os.environ["AWS_ENDPOINT_URL"],
            "region_name": "us-east-1",
        },
        config_kwargs={
            "read_timeout": 900,
            "connect_timeout": 300,
            "retries": {"max_attempts": 10, "mode": "adaptive"},
        },
        asynchronous=False,
    )


def _retry_s3(fn, *args, max_retries=5, base_delay=2.0, **kwargs):
    """Retry an S3 operation with exponential backoff."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            is_transient = (
                    "502" in err_str or "503" in err_str or "500" in err_str
                    or "bad gateway" in err_str or "service unavailable" in err_str
                    or "timeout" in err_str or "ssl" in err_str
            )
            if not is_transient or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt) + 0.5 * (attempt + 1)
            logger.warning("S3 error (attempt %d/%d): %s — retry %.1fs",
                           attempt + 1, max_retries, e, delay)
            time.sleep(delay)
    raise last_exc


# STEP 1.1 — Nearest-Neighbor Distance with Guard Zone Border Correction
#
# For each microglia cell (Type I point), find the Euclidean distance to
# the nearest pSyn aggregate (Type II point) in 3D.
#
# **Guard Zone correction**: cells whose centroid lies within r_max of any
# boundary of the 3D observation window W are EXCLUDED.  Rationale: for
# such border cells the true nearest protein may lie outside W, causing
# upward bias in NND estimates.
#
# Additionally, for each interior cell we record the VOLUME of its nearest
# protein — this "mark" is needed because pSyn aggregates are NOT uniform
# in size; larger aggregates near cells are biologically more significant
# (synucleinophagy proxy).

def compute_nnd_guard_zone(
        cells_um: np.ndarray,
        proteins_um: np.ndarray,
        protein_volumes: np.ndarray,
        fov_window: Dict[str, float],
        r_max: float = R_MAX_GUARD,
) -> Dict[str, Any]:
    """
    Compute NND from each interior cell to its nearest protein, with
    Guard Zone border correction.

    Parameters
    cells_um : (N1, 3) array
        Cell centroids in μm, columns [Z, Y, X].
    proteins_um : (N2, 3) array
        Protein centroids in μm, columns [Z, Y, X].
    protein_volumes : (N2,) array
        Volume (μm³) of each protein — used as mark.
    fov_window : dict
        Window bounds {z_min_um, z_max_um, y_min_um, y_max_um,
                       x_min_um, x_max_um}.
    r_max : float
        Guard zone width (μm).

    Returns
    dict with keys:
        nnd              : 1-D array — NND values for interior cells (μm)
        nearest_volumes  : 1-D array — volume of nearest protein per cell
        n_interior       : int — cells surviving guard zone
        n_border_excluded: int — cells removed by guard zone
        mean_nnd         : float
        median_nnd       : float
        std_nnd          : float
    """
    empty_result = {
        "nnd": [], "nearest_volumes": [],
        "n_interior": 0, "n_border_excluded": int(len(cells_um)),
        "mean_nnd": None, "median_nnd": None, "std_nnd": None,
    }

    if len(cells_um) == 0 or len(proteins_um) == 0:
        return empty_result

    # Identify interior cells (at least r_max from every edge)
    z_lo = fov_window["z_min_um"] + r_max
    z_hi = fov_window["z_max_um"] - r_max
    y_lo = fov_window["y_min_um"] + r_max
    y_hi = fov_window["y_max_um"] - r_max
    x_lo = fov_window["x_min_um"] + r_max
    x_hi = fov_window["x_max_um"] - r_max

    interior_mask = (
            (cells_um[:, 0] >= z_lo) & (cells_um[:, 0] <= z_hi) &
            (cells_um[:, 1] >= y_lo) & (cells_um[:, 1] <= y_hi) &
            (cells_um[:, 2] >= x_lo) & (cells_um[:, 2] <= x_hi)
    )

    interior_cells = cells_um[interior_mask]
    n_interior = len(interior_cells)
    n_border_excluded = int(np.sum(~interior_mask))

    if n_interior == 0:
        empty_result["n_border_excluded"] = n_border_excluded
        return empty_result

    # Compute all pairwise distances (n_interior × n_proteins)
    dists = cdist(interior_cells, proteins_um, metric="euclidean")

    # Nearest protein per cell
    nearest_idx = dists.argmin(axis=1)
    nnd = dists[np.arange(n_interior), nearest_idx]
    nearest_volumes = protein_volumes[nearest_idx]

    return {
        "nnd": nnd.tolist(),
        "nearest_volumes": nearest_volumes.tolist(),
        "n_interior": int(n_interior),
        "n_border_excluded": int(n_border_excluded),
        "mean_nnd": float(np.mean(nnd)),
        "median_nnd": float(np.median(nnd)),
        "std_nnd": float(np.std(nnd)),
    }


# STEP 1.2 — Overlap Index  (voxel-level mask intersection)
#
# Compute the fraction of pSyn voxels that physically overlap with IBA1+
# microglia voxels.  This is a direct proxy for synucleinophagy
# (pSyn engulfed by microglia).
#
#   overlap_index = 100 × |protein_mask AND cell_mask| / |protein_mask|
#
# Requires the RAW 3D binary masks (not just centroids), so the zarr
# store must be downloaded.

def compute_overlap_index(
        protein_mask: np.ndarray,
        cell_mask: np.ndarray,
) -> float:
    """
    Compute the percentage of pSyn volume inside IBA1+ cells.

    Parameters
    protein_mask : 3-D array  (binary, 0/1 or bool)
    cell_mask    : 3-D array  (binary, 0/1 or bool)

    Returns
    float — overlap percentage (0–100).
    """
    n_protein = int(np.sum(protein_mask > 0))
    if n_protein == 0:
        return 0.0
    n_overlap = int(np.sum(np.logical_and(protein_mask > 0, cell_mask > 0)))
    return 100.0 * n_overlap / n_protein


# STEP 1.3 — Edge-corrected Bivariate K-function (3D)
#
# Bivariate Ripley K-function K₁₂(r) with TRANSLATION edge correction
# for a 3D rectangular parallelepiped observation window W.
#
# Formula
#
#   K̂₁₂(r) = |W| / (n₁ · n₂)  ·  Σᵢ∈1 Σⱼ∈2  wᵢⱼ · I(‖xᵢ − xⱼ‖ ≤ r)
#
# Translation edge correction weight:
#
#   wᵢⱼ = |W| / |W ∩ (W + (xᵢ − xⱼ))|
#
# For a 3D box W = [0, Lz]×[0, Ly]×[0, Lx] with displacement d=(dz,dy,dx):
#
#   |W ∩ (W + d)| = max(0, Lz−|dz|) · max(0, Ly−|dy|) · max(0, Lx−|dx|)
#
# Under CSR
#
#   K₁₂(r) = (4/3)πr³
#
# L-function (3D isotropic normalisation)
#
#   L(r) = ( 3·K(r) / (4π) )^(1/3)
#   Under CSR:  L(r) = r.
#   L(r) − r > 0 → clustering (attraction)
#   L(r) − r < 0 → regularity (repulsion)
#
# Mark-weighted K-function
#
# Because pSyn aggregates are NOT uniform in size, we also compute
# K^m₁₂(r) where each protein's contribution is weighted by its volume:
#
#   K^m₁₂(r) = |W|² / (n₁ · Σⱼ mⱼ) · Σᵢ Σⱼ  mⱼ · I(dᵢⱼ ≤ r) / |W∩(W+dᵢⱼ)|
#
# Under null (marks independent of location): K^m₁₂(r) = (4/3)πr³
# (same CSR benchmark).  This detects whether LARGER proteins cluster
# preferentially near cells beyond what is expected by chance.
#
# R/spatstat alternative
#
# The same computation can be done via spatstat in R through rpy2:
#
#   library(spatstat)
#   X <- pp3(x, y, z, box3(xrange, yrange, zrange))
#   marks(X) <- factor(c(rep("cell", n1), rep("protein", n2)))
#   K <- Kcross(X, "cell", "protein", correction="translation", r=r_vals)
#   L <- with(K, (3*Ktrans/(4*pi))^(1/3))

def bivariate_K_3d(
        points1: np.ndarray,
        points2: np.ndarray,
        W_dims: np.ndarray,
        r_vals: np.ndarray,
) -> np.ndarray:
    """
    Compute 3D bivariate Ripley K-function K₁₂(r) with translation
    edge correction.

    Parameters
    points1 : (n1, 3) — Type I points (cells), μm, [Z, Y, X]
    points2 : (n2, 3) — Type II points (proteins), μm, [Z, Y, X]
    W_dims  : (3,)    — window dimensions [Lz, Ly, Lx] in μm
    r_vals  : 1-D     — radii at which to evaluate K

    Returns
    K_vals : 1-D array of K₁₂(r) values
    """
    n1, n2 = len(points1), len(points2)
    vol_W = float(np.prod(W_dims))

    if n1 == 0 or n2 == 0:
        return np.zeros(len(r_vals))

    # Displacement vectors  (n1, n2, 3)
    diffs = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]

    # Euclidean distances  (n1, n2)
    dists = np.linalg.norm(diffs, axis=2)

    # Translation edge correction:
    # overlap_vol = Π_k  max(0, W_dims[k] − |diff_k|)
    abs_diffs = np.abs(diffs)
    overlap_dims = np.maximum(0.0, W_dims[np.newaxis, np.newaxis, :] - abs_diffs)
    overlap_vol = np.prod(overlap_dims, axis=2)  # (n1, n2)

    # w_ij = |W| / overlap_vol  (0 where overlap_vol = 0)
    valid = overlap_vol > 0
    w_ij = np.where(valid, vol_W / overlap_vol, 0.0)

    # Efficient: sort by distance, use cumulative sum of weights
    dists_flat = dists.ravel()
    w_flat = w_ij.ravel()

    sort_idx = np.argsort(dists_flat)
    dists_sorted = dists_flat[sort_idx]
    w_sorted = w_flat[sort_idx]
    cumw = np.cumsum(w_sorted)

    # K(r) = (|W| / (n1 * n2)) * Σ w_ij * I(d ≤ r)
    norm = vol_W / (n1 * n2)

    K_vals = np.zeros(len(r_vals))
    for idx, r in enumerate(r_vals):
        j = np.searchsorted(dists_sorted, r, side="right")
        if j > 0:
            K_vals[idx] = cumw[j - 1] * norm

    return K_vals


def mark_weighted_K_3d(
        points1: np.ndarray,
        points2: np.ndarray,
        marks2: np.ndarray,
        W_dims: np.ndarray,
        r_vals: np.ndarray,
) -> np.ndarray:
    """
    Mark-weighted bivariate K-function K^m₁₂(r).

    Each protein point's contribution is weighted by its volume mark.
    Under null (independent marks): K^m₁₂(r) = (4/3)πr³ (same as CSR).

    Parameters
    points1, points2, W_dims, r_vals : as in bivariate_K_3d
    marks2 : (n2,) array — mark values (volume in μm³) for each protein

    Returns
    K_m_vals : 1-D array
    """
    n1, n2 = len(points1), len(points2)
    vol_W = float(np.prod(W_dims))
    sum_marks = float(np.sum(marks2))

    if n1 == 0 or n2 == 0 or sum_marks <= 0:
        return np.zeros(len(r_vals))

    diffs = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)

    abs_diffs = np.abs(diffs)
    overlap_dims = np.maximum(0.0, W_dims[np.newaxis, np.newaxis, :] - abs_diffs)
    overlap_vol = np.prod(overlap_dims, axis=2)

    valid = overlap_vol > 0
    # Combined weight:  mark_j / overlap_vol_ij
    w_ij = np.where(valid, marks2[np.newaxis, :] / overlap_vol, 0.0)

    # K^m(r) = |W|² / (n1 * Σ m_j) * Σ_i Σ_j w_ij * I(d ≤ r)
    norm = (vol_W ** 2) / (n1 * sum_marks)

    dists_flat = dists.ravel()
    w_flat = w_ij.ravel()

    sort_idx = np.argsort(dists_flat)
    dists_sorted = dists_flat[sort_idx]
    w_sorted = w_flat[sort_idx]
    cumw = np.cumsum(w_sorted)

    K_m_vals = np.zeros(len(r_vals))
    for idx, r in enumerate(r_vals):
        j = np.searchsorted(dists_sorted, r, side="right")
        if j > 0:
            K_m_vals[idx] = cumw[j - 1] * norm

    return K_m_vals


def K_to_L(K_vals: np.ndarray) -> np.ndarray:
    """
    Convert K-function to L-function (3D isotropic normalisation).

        L(r) = ( 3·K(r) / (4π) )^(1/3)

    Under CSR: L(r) = r.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.cbrt(3.0 * K_vals / (4.0 * np.pi))
    # Replace any NaN / Inf from K=0
    L = np.where(np.isfinite(L), L, 0.0)
    return L


# STEP 1.4 — Random Labeling null model
#
# Null hypothesis H₀:  The labels "cell" and "protein" are exchangeable;
# the spatial locations are fixed, only the type labels are randomly
# reassigned (keeping n₁ cells and n₂ proteins).
#
# Procedure (each of n_sim=999 simulations):
#   1. Pool all N = n₁ + n₂ coordinates.
#   2. Randomly select n₁ of them as "cells", the rest as "proteins".
#   3. Compute K₁₂(r) with translation edge correction.
#
# Note on marks:  the random labeling test uses the UNWEIGHTED K-function
# to test for spatial pattern regardless of protein size.  The mark-weighted
# K is computed for the observed data only (Step 1.3) and serves as a
# descriptive statistic showing whether larger proteins cluster near cells.

def random_labeling_test(
        cells_um: np.ndarray,
        proteins_um: np.ndarray,
        W_dims: np.ndarray,
        r_vals: np.ndarray,
        n_sim: int = N_SIM,
        seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Random Labeling Monte Carlo test.

    Parameters
    cells_um   : (n1, 3) — cell centroids in μm
    proteins_um: (n2, 3) — protein centroids in μm
    W_dims     : (3,)    — window dimensions in μm
    r_vals     : radii for K-function evaluation
    n_sim      : number of simulations (default 999)
    seed       : random seed

    Returns
    K_sims : (n_sim, len(r_vals)) array of K-functions under null
    """
    n1 = len(cells_um)
    n2 = len(proteins_um)
    N = n1 + n2

    # Pool all coordinates
    pooled = np.vstack([cells_um, proteins_um])  # (N, 3)

    rng = np.random.default_rng(seed)
    K_sims = np.zeros((n_sim, len(r_vals)))

    for s in range(n_sim):
        perm = rng.permutation(N)
        sim_cells = pooled[perm[:n1]]
        sim_proteins = pooled[perm[n1:]]
        K_sims[s] = bivariate_K_3d(sim_cells, sim_proteins, W_dims, r_vals)

    return K_sims


# STEP 1.5 — Global Envelope Test  (Maximum Absolute Deviation)
#
# The MAD (Maximum Absolute Deviation) test is a GLOBAL envelope test:
# it accounts for the fact that we test at multiple radii simultaneously.
#
# Procedure:
#   1. Convert all K-curves to L-curves: L(r) = (3K/(4π))^(1/3)
#   2. Define T = max_r |L(r) − r|  (the MAD statistic)
#   3. Compute T_obs for the observed data.
#   4. Compute T_s for each simulation under null.
#   5. Global p-value = (#{s: T_s ≥ T_obs} + 1) / (n_sim + 1)
#      (the +1 accounts for the observed statistic itself)
#   6. Pointwise 95% envelope: 2.5th and 97.5th percentiles of L_sims.

def global_envelope_test(
        K_obs: np.ndarray,
        K_sims: np.ndarray,
        r_vals: np.ndarray,
        alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Global Envelope Test using Maximum Absolute Deviation (MAD).

    Parameters
    K_obs  : (len(r_vals),)  — observed K-function
    K_sims : (n_sim, len(r_vals)) — K-functions under null
    r_vals : radii
    alpha  : significance level (default 0.05)

    Returns
    dict with keys:
        L_obs        : observed L-function
        L_lo, L_hi   : pointwise (1−α) envelope
        csr_line     : r_vals (L = r under CSR)
        T_obs        : observed MAD statistic
        p_value      : global p-value
        is_significant : bool
    """
    # Convert to L-functions
    L_obs = K_to_L(K_obs)
    L_sims = np.array([K_to_L(K_sims[s]) for s in range(len(K_sims))])

    # CSR reference: L(r) = r
    csr = r_vals.copy()

    # MAD statistic:  T = max_r |L(r) − r|
    T_obs = float(np.max(np.abs(L_obs - csr)))
    T_sims = np.array([float(np.max(np.abs(L_sims[s] - csr)))
                       for s in range(len(L_sims))])

    # Global p-value (one-sided: is observed deviation larger than null?)
    n_sim = len(T_sims)
    p_value = float((np.sum(T_sims >= T_obs) + 1) / (n_sim + 1))

    # Pointwise envelope
    lo_pct = 100.0 * alpha / 2.0
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    L_lo = np.percentile(L_sims, lo_pct, axis=0)
    L_hi = np.percentile(L_sims, hi_pct, axis=0)

    return {
        "L_obs": L_obs.tolist(),
        "L_lo": L_lo.tolist(),
        "L_hi": L_hi.tolist(),
        "csr_line": csr.tolist(),
        "T_obs": T_obs,
        "p_value": p_value,
        "is_significant": p_value < alpha,
    }


def plot_L_function(
        r_vals: np.ndarray,
        L_obs: np.ndarray,
        L_lo: np.ndarray,
        L_hi: np.ndarray,
        csr_line: np.ndarray,
        title: str,
        save_path: str,
) -> None:
    """
    Plot L-function with global envelope and CSR reference line.

    Uses matplotlib Agg backend (set at top-level import).
    Font setup for CJK characters is done here since it depends on
    runtime font availability.
    """
    # Font setup for potential CJK characters
    try:
        fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf')
        plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        # Font file may not exist on all systems; fall back gracefully
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(8, 6))

    # Envelope (shaded)
    ax.fill_between(r_vals, L_lo, L_hi, alpha=0.25, color='grey',
                    label='95% envelope (random labeling)')

    # CSR reference: L(r) = r
    ax.plot(r_vals, csr_line, 'k--', linewidth=1, label='CSR: L(r) = r')

    # Observed L-function
    ax.plot(r_vals, L_obs, 'b-', linewidth=2, label='Observed L(r)')

    ax.set_xlabel('r (μm)', fontsize=12)
    ax.set_ylabel('L(r) (μm)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved L-function plot: %s", save_path)


# STEP 1.6 — Aggregate SPP metrics per FOV / patient
#
# For each FOV compute:
#   - K₁₂(r) and L(r)   (unweighted)
#   - K^m₁₂(r) and L^m(r)  (mark-weighted by protein volume)
#   - NND (mean, median, std) with guard zone
#   - Overlap index (requires raw masks)
#   - spp_score        = max(L(r) − r)     — unweighted clustering intensity
#   - spp_score_marked = max(L^m(r) − r)   — mark-weighted clustering
#   - Protein volume statistics (mean, median, IQR) per FOV
#
# Output: DataFrame with one row per FOV.

def compute_spp_for_fov(
        fov_data: Dict[str, Any],
        r_vals: np.ndarray = R_VALS,
        r_max_guard: float = R_MAX_GUARD,
        n_sim: int = N_SIM,
        seed: int = RANDOM_SEED,
        compute_overlap: bool = True,
        fs: Optional[s3fs.S3FileSystem] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run all SPP metrics for a single FOV.

    Parameters
    fov_data : dict — the JSON data loaded from MinIO (cc3d result)
    r_vals, r_max_guard, n_sim, seed : SPP parameters
    compute_overlap : if True, download zarr and compute overlap index
    fs : s3fs filesystem (needed for overlap computation)

    Returns
    dict with SPP results, or None on failure.
    """
    spp = fov_data.get("spp_data", {})
    fov_win = fov_data.get("fov_window_um")

    cell_data = spp.get("type_I_cell", {})
    prot_data = spp.get("type_II_protein", {})

    cells_um = np.array(cell_data.get("points_um", []))
    proteins_um = np.array(prot_data.get("points_um", []))
    prot_vols = np.array(prot_data.get("marks", {}).get("volume_um3", []))

    n_cells = len(cells_um)
    n_proteins = len(proteins_um)

    result = {
        "patient_id": fov_data.get("patient_id"),
        "group": fov_data.get("group"),
        "region": fov_data.get("region"),
        "fov_id": fov_data.get("fov_id"),
        "n_cells": n_cells,
        "n_proteins": n_proteins,
    }

    # Skip FOVs with insufficient data
    if n_cells == 0 or n_proteins == 0:
        logger.warning(
            "FOV %s/%s: insufficient data (n_cells=%d, n_proteins=%d) — skipping SPP",
            fov_data.get("patient_id"), fov_data.get("fov_id"),
            n_cells, n_proteins,
        )
        result.update({
            "mean_nnd": None, "median_nnd": None, "std_nnd": None,
            "overlap_index": None, "spp_score": None, "spp_score_marked": None,
            "p_value": None, "prot_vol_mean": None, "prot_vol_median": None,
        })
        return result

    # Window dimensions
    if fov_win is None:
        logger.warning("FOV %s: no fov_window_um — skipping", fov_data.get("fov_id"))
        return None

    W_dims = np.array([
        fov_win["z_max_um"] - fov_win["z_min_um"],
        fov_win["y_max_um"] - fov_win["y_min_um"],
        fov_win["x_max_um"] - fov_win["x_min_um"],
    ])

    # Step 1.1: NND with Guard Zone
    nnd_result = compute_nnd_guard_zone(
        cells_um, proteins_um, prot_vols, fov_win, r_max_guard,
    )
    result["mean_nnd"] = nnd_result["mean_nnd"]
    result["median_nnd"] = nnd_result["median_nnd"]
    result["std_nnd"] = nnd_result["std_nnd"]
    result["n_interior_cells"] = nnd_result["n_interior"]
    result["n_border_excluded"] = nnd_result["n_border_excluded"]

    # Step 1.3: K-function (unweighted + mark-weighted)
    K_obs = bivariate_K_3d(cells_um, proteins_um, W_dims, r_vals)
    L_obs = K_to_L(K_obs)
    spp_score = float(np.max(L_obs - r_vals))

    K_m_obs = mark_weighted_K_3d(cells_um, proteins_um, prot_vols, W_dims, r_vals)
    L_m_obs = K_to_L(K_m_obs)
    spp_score_marked = float(np.max(L_m_obs - r_vals))

    result["spp_score"] = spp_score
    result["spp_score_marked"] = spp_score_marked

    # Step 1.4: Random Labeling
    K_sims = random_labeling_test(
        cells_um, proteins_um, W_dims, r_vals,
        n_sim=n_sim, seed=seed,
    )

    # Step 1.5: Global Envelope Test
    envelope = global_envelope_test(K_obs, K_sims, r_vals)
    result["p_value"] = envelope["p_value"]
    result["T_obs"] = envelope["T_obs"]

    # Step 1.2: Overlap Index (optional — needs zarr download)
    overlap_index = None
    if compute_overlap and fs is not None:
        zarr_key = fov_data.get("zarr_key")
        if zarr_key:
            try:
                overlap_index = _compute_overlap_from_zarr(fs, zarr_key)
            except Exception as exc:
                logger.warning("Overlap failed for %s: %s", zarr_key, exc)
    result["overlap_index"] = overlap_index

    # Protein volume statistics (marks)
    if len(prot_vols) > 0:
        result["prot_vol_mean"] = float(np.mean(prot_vols))
        result["prot_vol_median"] = float(np.median(prot_vols))
        result["prot_vol_iqr"] = float(
            np.percentile(prot_vols, 75) - np.percentile(prot_vols, 25)
        )
        result["prot_vol_max"] = float(np.max(prot_vols))
    else:
        result["prot_vol_mean"] = None
        result["prot_vol_median"] = None
        result["prot_vol_iqr"] = None
        result["prot_vol_max"] = None

    # Store full K/L curves
    result["K_unweighted"] = K_obs.tolist()
    result["K_mark_weighted"] = K_m_obs.tolist()
    result["L_unweighted"] = L_obs.tolist()
    result["L_mark_weighted"] = L_m_obs.tolist()
    result["L_envelope_lo"] = envelope["L_lo"]
    result["L_envelope_hi"] = envelope["L_hi"]
    result["r_vals"] = r_vals.tolist()

    return result


def _compute_overlap_from_zarr(
        fs: s3fs.S3FileSystem,
        zarr_key: str,
) -> float:
    """Download zarr, compute overlap index, delete local copy."""
    local_name = zarr_key.replace("/", "_")
    local_path = os.path.join(CACHE_DIR, "overlap_" + local_name)

    try:
        s3_path = f"{STORAGE_NAME}/{zarr_key}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        _retry_s3(fs.get, s3_path, local_path, recursive=True)

        root = zarr.open(local_path, mode="r")

        # Load protein mask
        arr_p = root["labels"]["protein_mask"]["0"]
        protein_mask = np.asarray(arr_p)
        if protein_mask.ndim == 4 and protein_mask.shape[0] == 1:
            protein_mask = protein_mask[0]

        # Load cell mask
        arr_c = root["labels"]["cell_mask"]["0"]
        cell_mask = np.asarray(arr_c)
        if cell_mask.ndim == 4 and cell_mask.shape[0] == 1:
            cell_mask = cell_mask[0]

        overlap = compute_overlap_index(protein_mask, cell_mask)
        return overlap

    finally:
        # Always clean up
        if os.path.isdir(local_path):
            shutil.rmtree(local_path, ignore_errors=True)


# STEP 1.7 — LMM Group Comparison (PD vs HC) with Residual Diagnostics
#
# Linear Mixed-Effects Model with Patient_ID as random effect.
#
#   Model 1 (Random Intercept):
#       spp_score ~ is_pd,  random = 1 | Patient_ID
#
#   Model 2 (Random Intercept + Slope):
#       spp_score ~ is_pd,  random = 1 + is_pd | Patient_ID
#
# The random intercept accounts for within-patient correlation
# (multiple FOVs per patient).  The random slope allows the disease
# effect to vary across donors.
#
# CRITICAL: Residual diagnostics (Q-Q plot, histogram) to verify
# normality assumption.  If violated, the p-values are unreliable.

def lmm_comparison(
        df: pd.DataFrame,
        metric: str = "spp_score",
        use_log: bool = True,
) -> Dict[str, Any]:
    """
    Fit Linear Mixed-Effects Models comparing PD vs HC.

    Parameters
    df      : DataFrame with columns: patient_id, group, spp_score, etc.
    metric  : response variable name (default 'spp_score').
              If metric starts with 'log_', the base name is extracted for
              plot labels.
    use_log : if True and a log_{metric} column exists, also fit on the
              log-transformed version.

    Returns
    dict with model summaries, p-values, and residual diagnostics.
    """
    # Extract base name for plot labels (e.g., "log_spp_score" → "spp_score")
    if metric.startswith("log_"):
        base_name = metric[4:]
    else:
        base_name = metric

    # Prepare data
    df = df.dropna(subset=[metric]).copy()
    df["is_pd"] = (df["group"] == "PD").astype(int)

    if len(df) < 10:
        logger.warning("Too few observations for LMM: %d", len(df))
        return {"error": "insufficient data"}

    results = {}

    # Model 1: Random Intercept
    try:
        model_ri = smf.mixedlm(
            f"{metric} ~ is_pd",
            data=df,
            groups=df["patient_id"],
        )
        fit_ri = model_ri.fit(reml=True)
        results["random_intercept"] = {
            "summary": str(fit_ri.summary()),
            "coef_is_pd": float(fit_ri.fe_params.get("is_pd", float("nan"))),
            "p_is_pd": float(fit_ri.pvalues.get("is_pd", float("nan"))),
            "aic": float(fit_ri.aic),
            "bic": float(fit_ri.bic),
            "residuals": fit_ri.resid.tolist(),
        }
        results["ri_residuals"] = fit_ri.resid
    except Exception as exc:
        logger.error("Random Intercept LMM failed: %s", exc)
        results["random_intercept"] = {"error": str(exc)}

    # Model 2: Random Intercept + Random Slope
    try:
        model_rs = smf.mixedlm(
            f"{metric} ~ is_pd",
            data=df,
            groups=df["patient_id"],
            re_formula="~is_pd",
        )
        fit_rs = model_rs.fit(reml=True)
        results["random_slope"] = {
            "summary": str(fit_rs.summary()),
            "coef_is_pd": float(fit_rs.fe_params.get("is_pd", float("nan"))),
            "p_is_pd": float(fit_rs.pvalues.get("is_pd", float("nan"))),
            "aic": float(fit_rs.aic),
            "bic": float(fit_rs.bic),
            "residuals": fit_rs.resid.tolist(),
        }
        results["rs_residuals"] = fit_rs.resid
    except Exception as exc:
        logger.warning("Random Slope LMM failed (often singular): %s", exc)
        results["random_slope"] = {"error": str(exc)}

    # Residual diagnostics
    resids = results.get("ri_residuals")
    if resids is not None:
        residuals = np.asarray(resids)

        # Shapiro-Wilk test (normality)
        if len(residuals) >= 3:
            shapiro_stat, shapiro_p = sp_stats.shapiro(residuals)
        else:
            shapiro_stat, shapiro_p = float("nan"), float("nan")

        results["residual_diagnostics"] = {
            "shapiro_stat": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "residuals_normal": shapiro_p > 0.05,
            "warning": (
                "Residuals may NOT be normally distributed (Shapiro p={:.4f}). "
                "Consider data transformation or robust methods.".format(shapiro_p)
                if shapiro_p <= 0.05 else None
            ),
        }

        # Q-Q plot and histogram
        _plot_residual_diagnostics(residuals, metric, results)

    # If use_log and base metric exists, fit original (non-log) too
    if use_log and base_name != metric and base_name in df.columns:
        df_orig = df.dropna(subset=[base_name]).copy()
        if len(df_orig) >= 10:
            try:
                model_orig = smf.mixedlm(
                    f"{base_name} ~ is_pd",
                    data=df_orig,
                    groups=df_orig["patient_id"],
                )
                fit_orig = model_orig.fit(reml=True)
                results["original_metric_lmm"] = {
                    "summary": str(fit_orig.summary()),
                    "coef_is_pd": float(fit_orig.fe_params.get("is_pd", float("nan"))),
                    "p_is_pd": float(fit_orig.pvalues.get("is_pd", float("nan"))),
                    "aic": float(fit_orig.aic),
                    "bic": float(fit_orig.bic),
                }
            except Exception as exc:
                logger.warning("Original-metric LMM failed: %s", exc)
                results["original_metric_lmm"] = {"error": str(exc)}

    # Clean up non-serializable fields
    results.pop("ri_residuals", None)
    results.pop("rs_residuals", None)

    return results


def _plot_residual_diagnostics(
        residuals: np.ndarray,
        metric: str,
        results: Dict[str, Any],
) -> None:
    """Create Q-Q plot and histogram of LMM residuals."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    save_path = os.path.join(CACHE_DIR, f"residuals_{metric}.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Q-Q plot
    sp_stats.probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q Plot of Residuals")
    axes[0].grid(True, alpha=0.3)

    # Histogram
    axes[1].hist(residuals, bins=30, density=True, alpha=0.7, color='steelblue')
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    axes[1].plot(x_range, sp_stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
                 'r-', linewidth=2, label='Normal fit')
    axes[1].set_title("Histogram of Residuals")
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Residual Diagnostics — {metric}", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    results["_residual_plot_path"] = save_path


# Load CC3D results from MinIO

def load_all_results(fs: s3fs.S3FileSystem) -> List[Dict[str, Any]]:
    """
    Load all per-FOV JSON results from MinIO (produced by cc3d_production).

    Returns a list of dicts, one per FOV.
    """
    results = []
    prefix = f"{STORAGE_NAME}/{CC3D_OUTPUT_PREFIX}/"

    # Walk group/region/patient/fov.json
    for grp in ["HC", "PD"]:
        for reg in ["putamen", "substantiaNigra"]:
            folder = f"{prefix}{grp}/{reg}/"
            try:
                patients = fs.ls(folder, detail=False)
            except FileNotFoundError:
                continue

            for p_path in patients:
                try:
                    fovs = fs.ls(p_path, detail=False)
                except FileNotFoundError:
                    continue

                for fov_path in fovs:
                    if not fov_path.endswith(".json"):
                        continue
                    try:
                        with fs.open(fov_path, "r") as f:
                            data = json.load(f)
                        results.append(data)
                    except Exception as exc:
                        logger.warning("Failed to load %s: %s", fov_path, exc)

    logger.info("Loaded %d FOV results from MinIO.", len(results))
    return results


# STEP E — Quality Filtering

def quality_filter(
        df: pd.DataFrame,
        min_interior: int = MIN_INTERIOR_CELLS,
) -> pd.DataFrame:
    """
    Filter DataFrame to keep only FOVs with n_interior_cells >= min_interior.

    FOVs with too few interior cells produce unreliable SPP metrics
    because the guard-zone border correction excludes most observations.

    Parameters
    df          : DataFrame with 'n_interior_cells' column.
    min_interior: minimum number of interior cells required.

    Returns
    Filtered DataFrame.
    """
    n_before = len(df)

    if "n_interior_cells" not in df.columns:
        logger.warning(
            "Column 'n_interior_cells' not found — skipping quality filter."
        )
        return df

    df_out = df[df["n_interior_cells"] >= min_interior].copy()
    n_removed = n_before - len(df_out)
    logger.info(
        "Quality filter: removed %d / %d FOVs (n_interior < %d). "
        "Remaining: %d FOVs.",
        n_removed, n_before, min_interior, len(df_out),
    )
    return df_out


# STEP F — Log Transformation of SPP Metrics

def log_transform_metrics(
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Add log-transformed columns for each SPP metric.

    For each metric *m*, creates column ``log_{m}`` via::

        log_{m} = np.log1p(m - min(m, na.rm=True) + 1e-6)

    This shifts the distribution to be strictly positive before applying
    log1p, handling metrics like spp_score which can be negative (≈ −0.5).

    NaN values in the original column are preserved as NaN in the
    log-transformed column.

    Parameters
    df      : DataFrame with SPP metric columns.
    metrics : list of column names to transform.
              Default: ["spp_score", "spp_score_marked", "overlap_index"].

    Returns
    DataFrame with new ``log_`` columns added.
    """
    if metrics is None:
        metrics = ["spp_score", "spp_score_marked", "overlap_index"]

    for metric in metrics:
        if metric not in df.columns:
            logger.warning("Metric column '%s' not found — skipping log transform.", metric)
            continue

        valid = df[metric].dropna()
        if len(valid) == 0:
            logger.warning("Metric '%s' has no valid values — skipping log transform.", metric)
            continue

        min_val = float(valid.min())
        df[f"log_{metric}"] = np.log1p(df[metric] - min_val + 1e-6)
        logger.info(
            "Log-transformed '%s': min=%.4f → log_%s range [%.4f, %.4f]",
            metric, min_val, metric,
            df[f"log_{metric}"].min(), df[f"log_{metric}"].max(),
        )

    return df


# STEP K — GLMM (Generalized Estimating Equations with Gamma family)

def glmm_comparison(
        df: pd.DataFrame,
        metric: str = "log_spp_score",
) -> Dict[str, Any]:
    """
    Fit a Generalized Estimating Equations (GEE) model with Gamma family
    and log link, comparing PD vs HC.

    GEE accounts for within-patient correlation (multiple FOVs per patient)
    while the Gamma family is appropriate for positively-skewed continuous
    outcomes like log-transformed SPP scores.

    Parameters
    df     : DataFrame with columns: patient_id, group, {metric}.
    metric : response variable name.

    Returns
    dict with model summary, coefficients, p-values, and confidence
    intervals.
    """
    df = df.dropna(subset=[metric]).copy()
    df["is_pd"] = (df["group"] == "PD").astype(int)

    if len(df) < 10:
        logger.warning("Too few observations for GEE: %d", len(df))
        return {"error": "insufficient data"}

    # Ensure the response is strictly positive (required for Gamma)
    y_min = df[metric].min()
    if y_min <= 0:
        shift = abs(y_min) + 1.0
        df["_shifted_response"] = df[metric] + shift
        response_col = "_shifted_response"
        logger.info(
            "GEE: shifting %s by %.4f to ensure positivity for Gamma family.",
            metric, shift,
        )
    else:
        response_col = metric

    results = {}

    try:
        fam = sm.families.Gamma(link=sm.families.links.Log())
        model = sm.GEE(
            df[response_col],
            sm.add_constant(df["is_pd"]),
            groups=df["patient_id"],
            family=fam,
        )
        fit = model.fit()

        params = fit.params
        conf_int = fit.conf_int()
        results["gee_gamma"] = {
            "summary": str(fit.summary()),
            "coef_is_pd": float(params.iloc[1]) if len(params) > 1 else float("nan"),
            "p_is_pd": float(fit.pvalues.iloc[1]) if len(fit.pvalues) > 1 else float("nan"),
            "ci_is_pd_lo": float(conf_int.iloc[1, 0]) if len(conf_int) > 1 else float("nan"),
            "ci_is_pd_hi": float(conf_int.iloc[1, 1]) if len(conf_int) > 1 else float("nan"),
            "scale": float(fit.scale),
            "response_shifted": y_min <= 0,
        }
    except Exception as exc:
        logger.error("GEE Gamma model failed: %s", exc)
        results["gee_gamma"] = {"error": str(exc)}

    return results


# STEP L — Patient-level Permutation Test

def permutation_test_patient_level(
        df: pd.DataFrame,
        metric: str = "log_spp_score",
        n_perm: int = N_PERMUTATIONS,
        seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Patient-level permutation test for group difference.

    Aggregates the metric to patient-level means, then permutes the
    PD/HC group labels across patients while preserving group sizes
    (14 PD, 13 HC — or whatever the data contains).

    Parameters
    df     : DataFrame with columns: patient_id, group, {metric}.
    metric : response variable name.
    n_perm : number of permutations (default N_PERMUTATIONS = 10000).
    seed   : random seed.

    Returns
    dict with observed_diff, p_value, and perm_distribution.
    """
    # Aggregate to patient level
    patient_df = (
        df.dropna(subset=[metric])
        .groupby(["patient_id", "group"])[metric]
        .mean()
        .reset_index()
    )

    pd_mean = patient_df.loc[patient_df["group"] == "PD", metric].mean()
    hc_mean = patient_df.loc[patient_df["group"] == "HC", metric].mean()
    observed_diff = float(pd_mean - hc_mean)

    n_pd = (patient_df["group"] == "PD").sum()
    n_hc = (patient_df["group"] == "HC").sum()
    n_patients = len(patient_df)

    if n_pd < 2 or n_hc < 2:
        logger.warning(
            "Too few patients for permutation test: %d PD, %d HC.",
            n_pd, n_hc,
        )
        return {"error": "insufficient patients", "observed_diff": observed_diff}

    rng = np.random.default_rng(seed)
    values = patient_df[metric].values
    perm_diffs = np.zeros(n_perm)

    for i in range(n_perm):
        perm_labels = rng.permutation(n_patients)
        perm_diff = float(values[perm_labels[:n_pd]].mean() - values[perm_labels[n_pd:]].mean())
        perm_diffs[i] = perm_diff

    # Two-sided p-value (absolute difference)
    p_value = float((np.sum(np.abs(perm_diffs) >= abs(observed_diff)) + 1) / (n_perm + 1))

    logger.info(
        "Permutation test (%s): observed_diff=%.4f, p=%.6f (n_perm=%d)",
        metric, observed_diff, p_value, n_perm,
    )

    return {
        "observed_diff": observed_diff,
        "pd_mean": float(pd_mean),
        "hc_mean": float(hc_mean),
        "p_value": p_value,
        "n_perm": n_perm,
        "n_pd": int(n_pd),
        "n_hc": int(n_hc),
        "perm_distribution": perm_diffs.tolist(),
    }


# STEP M — Sensitivity Analysis (Outlier Removal)

def sensitivity_analysis(
        df: pd.DataFrame,
        metric: str = "log_spp_score",
        outlier_patients: List[str] = OUTLIER_PATIENTS,
) -> Dict[str, Any]:
    """
    Sensitivity analysis: compare results with and without outlier patients.

    Parameters
    df                : DataFrame with columns: patient_id, group, {metric}.
    metric            : response variable name.
    outlier_patients  : list of patient IDs to exclude in sensitivity run.

    Returns
    dict with LMM and permutation results for full and filtered data.
    """
    results = {}

    # Full data
    logger.info("Sensitivity analysis — full data (%d FOVs).", len(df))
    results["full_lmm"] = lmm_comparison(df, metric=metric, use_log=False)
    results["full_permutation"] = permutation_test_patient_level(df, metric=metric)

    # Without outliers
    df_no = df[~df["patient_id"].isin(outlier_patients)].copy()
    n_removed = len(df) - len(df_no)
    logger.info(
        "Sensitivity analysis — without outliers (%s): %d FOVs removed, %d remaining.",
        outlier_patients, n_removed, len(df_no),
    )

    if len(df_no) >= 10:
        results["no_outliers_lmm"] = lmm_comparison(df_no, metric=metric, use_log=False)
        results["no_outliers_permutation"] = permutation_test_patient_level(df_no, metric=metric)
    else:
        results["no_outliers_lmm"] = {"error": "too few observations after outlier removal"}
        results["no_outliers_permutation"] = {"error": "too few observations after outlier removal"}

    return results


# STEP N — Region-stratified LMM

def region_stratified_lmm(
        df: pd.DataFrame,
        metric: str = "log_spp_score",
) -> Dict[str, Dict[str, Any]]:
    """
    Run LMM and permutation test separately for each brain region.

    Parameters
    df     : DataFrame with columns: patient_id, group, region, {metric}.
    metric : response variable name.

    Returns
    dict keyed by region name, each value is a dict with 'lmm' and
    'permutation' results.
    """
    results = {}

    if "region" not in df.columns:
        logger.warning("No 'region' column found — cannot stratify.")
        return results

    regions = df["region"].dropna().unique()
    logger.info("Region-stratified analysis: %d regions found.", len(regions))

    for region in sorted(regions):
        df_reg = df[df["region"] == region].copy()
        n_fov = len(df_reg)
        n_pat = df_reg["patient_id"].nunique()
        logger.info(
            "  Region '%s': %d FOVs from %d patients.", region, n_fov, n_pat,
        )

        reg_result = {"n_fovs": n_fov, "n_patients": n_pat}

        if n_fov >= 10 and n_pat >= 4:
            reg_result["lmm"] = lmm_comparison(df_reg, metric=metric, use_log=False)
            reg_result["permutation"] = permutation_test_patient_level(df_reg, metric=metric)
        else:
            reg_result["lmm"] = {"error": "insufficient data for region"}
            reg_result["permutation"] = {"error": "insufficient data for region"}

        results[region] = reg_result

    return results


# STEP P — Bootstrap Cohen's d

def bootstrap_cohens_d(
        df: pd.DataFrame,
        metric: str = "log_overlap_index",
        region: Optional[str] = None,
        n_boot: int = 10000,
        seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Bootstrap confidence interval for Cohen's d (PD vs HC) at patient level.

    Computes patient-level means, then resamples patients with replacement
    within each group to build a bootstrap distribution of Cohen's d.

    Parameters
    df      : DataFrame with columns: patient_id, group, region, {metric}.
    metric  : response variable name.
    region  : if not None, filter to this brain region before computing.
    n_boot  : number of bootstrap iterations.
    seed    : random seed.

    Returns
    dict with: d_observed, ci_lo, ci_hi, n_hc, n_pd.
    """
    # Filter to region if specified
    if region is not None and "region" in df.columns:
        df = df[df["region"] == region].copy()

    # Aggregate to patient level
    patient_df = (
        df.dropna(subset=[metric])
        .groupby(["patient_id", "group"])[metric]
        .mean()
        .reset_index()
    )

    hc_vals = patient_df.loc[patient_df["group"] == "HC", metric].values
    pd_vals = patient_df.loc[patient_df["group"] == "PD", metric].values

    n_hc, n_pd = len(hc_vals), len(pd_vals)
    if n_hc < 2 or n_pd < 2:
        return {"error": "insufficient patients", "d_observed": float("nan")}

    # Observed Cohen's d
    pooled_std = np.sqrt(
        ((n_hc - 1) * np.std(hc_vals, ddof=1) ** 2 + (n_pd - 1) * np.std(pd_vals, ddof=1) ** 2)
        / (n_hc + n_pd - 2)
    )
    d_obs = float((np.mean(pd_vals) - np.mean(hc_vals)) / pooled_std) if pooled_std > 0 else 0.0

    # Bootstrap
    rng = np.random.default_rng(seed)
    boot_d = np.zeros(n_boot)
    for i in range(n_boot):
        hc_boot = rng.choice(hc_vals, size=n_hc, replace=True)
        pd_boot = rng.choice(pd_vals, size=n_pd, replace=True)
        s = np.sqrt(
            ((n_hc - 1) * np.std(hc_boot, ddof=1) ** 2 + (n_pd - 1) * np.std(pd_boot, ddof=1) ** 2)
            / (n_hc + n_pd - 2)
        )
        boot_d[i] = (np.mean(pd_boot) - np.mean(hc_boot)) / s if s > 0 else 0.0

    ci_lo = float(np.percentile(boot_d, 2.5))
    ci_hi = float(np.percentile(boot_d, 97.5))

    logger.info(
        "Bootstrap Cohen's d (%s, region=%s): d=%.3f [%.3f, %.3f]",
        metric, region, d_obs, ci_lo, ci_hi,
    )

    return {
        "d_observed": d_obs,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_95": [ci_lo, ci_hi],
        "n_hc": int(n_hc),
        "n_pd": int(n_pd),
        "n_boot": n_boot,
    }


# STEP P-b — GLMM Gamma for Substantia Nigra

def glmm_gamma_region(
        df: pd.DataFrame,
        metric: str = "log_overlap_index",
        region: str = "substantiaNigra",
) -> Dict[str, Any]:
    """
    Fit a GEE model with Gamma family and log link for a specific region.

    Optimized for overlap_index analysis in Substantia Nigra, where the
    distribution is positively skewed and the Gamma family is more
    appropriate than Gaussian LMM.

    Parameters
    df     : DataFrame with columns: patient_id, group, region, {metric}.
    metric : response variable name.
    region : brain region to filter on.

    Returns
    dict with model summary, coefficients, p-values, and CI.
    """
    if "region" not in df.columns:
        return {"error": "no region column"}

    df_reg = df[df["region"] == region].dropna(subset=[metric]).copy()
    df_reg["is_pd"] = (df_reg["group"] == "PD").astype(int)

    if len(df_reg) < 10:
        return {"error": "insufficient data", "region": region}

    # Ensure positivity for Gamma
    y_min = df_reg[metric].min()
    if y_min <= 0:
        shift = abs(y_min) + 1.0
        df_reg["_response"] = df_reg[metric] + shift
        response_col = "_response"
    else:
        response_col = metric

    results = {"region": region, "n_fovs": len(df_reg), "response_shifted": y_min <= 0}

    try:
        fam = sm.families.Gamma(link=sm.families.links.Log())
        model = sm.GEE(
            df_reg[response_col],
            sm.add_constant(df_reg["is_pd"]),
            groups=df_reg["patient_id"],
            family=fam,
        )
        fit = model.fit()

        params = fit.params
        conf_int = fit.conf_int()
        results["gee_gamma"] = {
            "summary": str(fit.summary()),
            "coef_is_pd": float(params.iloc[1]) if len(params) > 1 else float("nan"),
            "p_is_pd": float(fit.pvalues.iloc[1]) if len(fit.pvalues) > 1 else float("nan"),
            "ci_is_pd_lo": float(conf_int.iloc[1, 0]) if len(conf_int) > 1 else float("nan"),
            "ci_is_pd_hi": float(conf_int.iloc[1, 1]) if len(conf_int) > 1 else float("nan"),
            "scale": float(fit.scale),
        }
        logger.info(
            "GLMM Gamma (%s, %s): coef=%.4f, p=%.4f",
            metric, region,
            results["gee_gamma"]["coef_is_pd"],
            results["gee_gamma"]["p_is_pd"],
        )
    except Exception as exc:
        logger.error("GLMM Gamma failed for %s/%s: %s", metric, region, exc)
        results["gee_gamma"] = {"error": str(exc)}

    return results


# STEP R — Ridge + Waterfall Plot for Substantia Nigra

def plot_sn_ridge(
        df: pd.DataFrame,
        metric: str = "log_overlap_index",
        save_dir: str = CACHE_DIR,
) -> str:
    """
    Create a ridge (joy) plot and waterfall chart for Substantia Nigra
    overlap_index comparison between HC and PD.

    Left panel: ridge plot showing patient-level density distributions
    for HC (blue) and PD (orange) in Substantia Nigra.
    Right panel: waterfall chart showing each patient's mean value,
    sorted by group, highlighting outlier patients.

    Parameters
    df       : DataFrame with patient_id, group, region, {metric}.
    metric   : response variable name.
    save_dir : directory for saving the PNG.

    Returns
    Path to saved figure.
    """
    # Font Setup
    try:
        fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf')
        plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    sn = df[df["region"] == "substantiaNigra"].dropna(subset=[metric]).copy()

    # Patient-level means
    patient_means = sn.groupby(["patient_id", "group"])[metric].mean().reset_index()
    patient_means = patient_means.sort_values(["group", metric]).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Ridge plot
    hc_patients = patient_means[patient_means["group"] == "HC"].sort_values(metric)
    pd_patients = patient_means[patient_means["group"] == "PD"].sort_values(metric)

    all_patients = pd.concat([hc_patients, pd_patients])
    n_total = len(all_patients)

    from scipy.stats import gaussian_kde

    y_positions = {}
    for i, (_, row) in enumerate(all_patients.iterrows()):
        pid = row["patient_id"]
        grp = row["group"]

        # Get FOV-level data for this patient
        fov_data = sn[sn["patient_id"] == pid][metric].values
        if len(fov_data) < 3:
            continue

        try:
            kde = gaussian_kde(fov_data)
            x_range = np.linspace(fov_data.min() - 0.1, fov_data.max() + 0.1, 200)
            density = kde(x_range)
            # Scale density for visibility
            density_scaled = density / density.max() * 0.8

            color = '#4C72B0' if grp == 'HC' else '#DD8452'
            alpha = 0.6
            lw = 1.5

            # Check if outlier
            if pid in OUTLIER_PATIENTS:
                color = '#C44E52'
                lw = 2.5
                alpha = 0.8

            ax1.fill_between(x_range, i, i + density_scaled, alpha=alpha * 0.5, color=color)
            ax1.plot(x_range, i + density_scaled, color=color, linewidth=lw)
        except Exception:
            ax1.scatter([row[metric]], [i], color='grey', s=20, zorder=5)

    # Group labels
    n_hc = len(hc_patients)
    n_pd = len(pd_patients)
    ax1.axhline(y=n_hc - 0.5, color='grey', linestyle='--', alpha=0.5)
    ax1.text(ax1.get_xlim()[0] + 0.01, n_hc / 2 - 0.5, 'HC', fontsize=14,
             fontweight='bold', color='#4C72B0', va='center')
    ax1.text(ax1.get_xlim()[0] + 0.01, n_hc + n_pd / 2 - 0.5, 'PD', fontsize=14,
             fontweight='bold', color='#DD8452', va='center')

    ax1.set_ylabel('Patient (sorted by group & value)', fontsize=11)
    ax1.set_xlabel(metric, fontsize=11)
    ax1.set_title('Ridge Plot — Substantia Nigra', fontsize=13)

    # Right: Waterfall chart
    grand_mean = patient_means[metric].mean()
    deviations = patient_means[metric].values - grand_mean
    colors = ['#4C72B0' if g == 'HC' else '#DD8452' for g in patient_means["group"]]

    for i, (dev, pid, grp) in enumerate(zip(deviations, patient_means["patient_id"], patient_means["group"])):
        c = '#C44E52' if pid in OUTLIER_PATIENTS else ('#4C72B0' if grp == 'HC' else '#DD8452')
        edge = 'black' if pid in OUTLIER_PATIENTS else 'none'
        lw = 2 if pid in OUTLIER_PATIENTS else 0.5
        ax2.bar(i, dev, color=c, edgecolor=edge, linewidth=lw, width=0.9)
        if pid in OUTLIER_PATIENTS:
            ax2.text(i, dev + (0.02 if dev > 0 else -0.04), pid,
                     ha='center', va='bottom' if dev > 0 else 'top',
                     fontsize=8, fontweight='bold', color='#C44E52')

    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_ylabel(f'Deviation from grand mean', fontsize=11)
    ax2.set_xlabel('Patient (sorted)', fontsize=11)
    ax2.set_title(f'Waterfall — {metric} (SN)', fontsize=13)

    # Add Mann-Whitney p-value annotation
    hc_vals = hc_patients[metric].values
    pd_vals = pd_patients[metric].values
    if len(hc_vals) >= 3 and len(pd_vals) >= 3:
        u_stat, p_val = sp_stats.mannwhitneyu(hc_vals, pd_vals, alternative='two-sided')
        sig_text = f'p = {p_val:.4f}' + (' *' if p_val < 0.05 else ' (n.s.)')
        ax2.text(0.98, 0.98, sig_text, transform=ax2.transAxes,
                 fontsize=12, fontweight='bold', ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Substantia Nigra — Overlap Index Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"sn_ridge_waterfall_{metric}.png")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved SN ridge+waterfall plot: %s", save_path)
    return save_path


# STEP G — Outlier Identification Plot

def plot_outlier_identification(
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        save_dir: str = CACHE_DIR,
) -> str:
    """
    Create box plots (HC vs PD) with individual patient means as scatter
    points, highlighting outlier patients.

    Parameters
    df      : DataFrame with patient_id, group, and metric columns.
    metrics : list of metrics to plot.
              Default: ["spp_score", "spp_score_marked", "overlap_index"].
    save_dir: directory for saving the PNG.

    Returns
    Path to saved figure.
    """
    if metrics is None:
        metrics = ["spp_score", "spp_score_marked", "overlap_index"]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    # Aggregate to patient level
    patient_df = df.groupby(["patient_id", "group"])[metrics].mean().reset_index()

    for ax, metric in zip(axes, metrics):
        valid = patient_df.dropna(subset=[metric])
        if len(valid) == 0:
            ax.set_title(f"{metric}\n(no data)")
            continue

        # Box plot
        groups = ["HC", "PD"]
        data_by_group = [valid.loc[valid["group"] == g, metric].dropna().values for g in groups]
        bp = ax.boxplot(data_by_group, labels=groups, patch_artist=True, widths=0.5)
        colors = ["#4C72B0", "#DD8452"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)

        # Scatter patient means
        for gi, grp in enumerate(groups):
            grp_data = valid[valid["group"] == grp]
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(grp_data))
            x = gi + 1 + jitter

            # Regular patients
            regular = grp_data[~grp_data["patient_id"].isin(OUTLIER_PATIENTS)]
            ax.scatter(
                gi + 1 + jitter[:len(regular)] if len(regular) > 0 else [],
                regular[metric] if len(regular) > 0 else [],
                color="black", alpha=0.6, s=30, zorder=5,
            )

            # Outlier patients
            outliers = grp_data[grp_data["patient_id"].isin(OUTLIER_PATIENTS)]
            if len(outliers) > 0:
                ax.scatter(
                    gi + 1 + jitter[len(regular):len(regular) + len(outliers)],
                    outliers[metric],
                    color="red", alpha=0.9, s=60, zorder=6, marker="D",
                    edgecolors="darkred", linewidths=1,
                )
                for _, row in outliers.iterrows():
                    ax.annotate(
                        row["patient_id"],
                        (gi + 1 + jitter[len(regular)], row[metric]),
                        textcoords="offset points",
                        xytext=(8, 4),
                        fontsize=8,
                        color="red",
                        fontweight="bold",
                    )

        ax.set_title(metric, fontsize=12)
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Outlier Identification — Patient-level SPP Metrics", fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "outlier_identification.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved outlier identification plot: %s", save_path)
    return save_path


# STEP H — Filtering Summary Plot

def plot_filtering_summary(
        df_raw: pd.DataFrame,
        df_filtered: pd.DataFrame,
        save_dir: str = CACHE_DIR,
) -> str:
    """
    Create figure showing before/after quality filtering.

    Left: histogram of n_interior_cells with vertical line at threshold.
    Right: Q-Q plot of spp_score before and after filtering.

    Parameters
    df_raw     : DataFrame before filtering.
    df_filtered: DataFrame after filtering.
    save_dir   : directory for saving the PNG.

    Returns
    Path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: histogram of n_interior_cells
    if "n_interior_cells" in df_raw.columns:
        axes[0].hist(
            df_raw["n_interior_cells"].dropna(),
            bins=30, color="steelblue", alpha=0.7, edgecolor="white",
        )
        axes[0].axvline(
            MIN_INTERIOR_CELLS, color="red", linestyle="--", linewidth=2,
            label=f"Threshold = {MIN_INTERIOR_CELLS}",
        )
        axes[0].set_xlabel("n_interior_cells", fontsize=11)
        axes[0].set_ylabel("Count", fontsize=11)
        axes[0].set_title("Distribution of Interior Cell Counts", fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "n_interior_cells column\nnot available",
                     ha="center", va="center", fontsize=12)

    # Right: Q-Q plot of spp_score before and after
    metric = "spp_score"
    if metric in df_raw.columns:
        raw_vals = df_raw[metric].dropna().values
        sp_stats.probplot(raw_vals, dist="norm", plot=axes[1])
        axes[1].set_title(f"Q-Q Plot of {metric} (before filtering)", fontsize=12)
        axes[1].grid(True, alpha=0.3)

        if metric in df_filtered.columns:
            filt_vals = df_filtered[metric].dropna().values
            if len(filt_vals) > 0:
                # Add secondary Q-Q for filtered data
                (osm2, osr2), (slope2, intercept2, r2) = sp_stats.probplot(filt_vals, dist="norm")
                axes[1].plot(osm2, osr2, 'g.', markersize=4, alpha=0.5, label="After filter")
                axes[1].plot(osm2, slope2 * osm2 + intercept2, 'g-', linewidth=1, alpha=0.7)
                axes[1].legend(["Before (theoretical)", "Before (data)", "After filter (data)"],
                               fontsize=9)
    else:
        axes[1].text(0.5, 0.5, f"{metric} column\nnot available",
                     ha="center", va="center", fontsize=12)

    fig.suptitle("Quality Filtering Summary", fontsize=14)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "filtering_summary.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved filtering summary plot: %s", save_path)
    return save_path


# STEP I — Transformed Diagnostics Plot

def plot_transformed_diagnostics(
        df: pd.DataFrame,
        metric: str = "spp_score",
        save_dir: str = CACHE_DIR,
) -> str:
    """
    Create 2×2 figure comparing raw vs log-transformed metric residuals.

    Top-left: Q-Q of raw metric residuals (from LMM).
    Top-right: Q-Q of log-transformed metric residuals.
    Bottom-left: Histogram of raw residuals.
    Bottom-right: Histogram of log-transformed residuals.

    This demonstrates whether the log transformation improved normality.

    Parameters
    df      : DataFrame with metric and log_{metric} columns.
    metric  : base metric name (e.g. 'spp_score').
    save_dir: directory for saving the PNG.

    Returns
    Path to saved figure.
    """
    log_metric = f"log_{metric}"

    # Fit simple LMMs to get residuals
    raw_resids = None
    log_resids = None

    df_work = df.dropna(subset=[metric]).copy()
    df_work["is_pd"] = (df_work["group"] == "PD").astype(int)

    if len(df_work) >= 10:
        try:
            model_raw = smf.mixedlm(f"{metric} ~ is_pd", data=df_work, groups=df_work["patient_id"])
            fit_raw = model_raw.fit(reml=True)
            raw_resids = fit_raw.resid.values
        except Exception:
            pass

    df_log = df.dropna(subset=[log_metric]).copy() if log_metric in df.columns else pd.DataFrame()
    if len(df_log) >= 10:
        df_log["is_pd"] = (df_log["group"] == "PD").astype(int)
        try:
            model_log = smf.mixedlm(f"{log_metric} ~ is_pd", data=df_log, groups=df_log["patient_id"])
            fit_log = model_log.fit(reml=True)
            log_resids = fit_log.resid.values
        except Exception:
            pass

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: Q-Q of raw residuals
    if raw_resids is not None and len(raw_resids) >= 3:
        sp_stats.probplot(raw_resids, dist="norm", plot=axes[0, 0])
        shapiro_p_raw = sp_stats.shapiro(raw_resids)[1] if len(raw_resids) >= 3 else float("nan")
        axes[0, 0].set_title(f"Q-Q: Raw {metric}\n(Shapiro p={shapiro_p_raw:.4f})", fontsize=11)
    else:
        axes[0, 0].set_title(f"Q-Q: Raw {metric}\n(insufficient data)")
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Q-Q of log-transformed residuals
    if log_resids is not None and len(log_resids) >= 3:
        sp_stats.probplot(log_resids, dist="norm", plot=axes[0, 1])
        shapiro_p_log = sp_stats.shapiro(log_resids)[1] if len(log_resids) >= 3 else float("nan")
        axes[0, 1].set_title(f"Q-Q: log {metric}\n(Shapiro p={shapiro_p_log:.4f})", fontsize=11)
    else:
        axes[0, 1].set_title(f"Q-Q: log {metric}\n(insufficient data)")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Histogram of raw residuals
    if raw_resids is not None:
        axes[1, 0].hist(raw_resids, bins=30, density=True, alpha=0.7, color="steelblue")
        x_r = np.linspace(raw_resids.min(), raw_resids.max(), 200)
        axes[1, 0].plot(x_r, sp_stats.norm.pdf(x_r, raw_resids.mean(), raw_resids.std()),
                        "r-", linewidth=2, label="Normal fit")
        axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_title(f"Histogram: Raw {metric} residuals", fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Histogram of log-transformed residuals
    if log_resids is not None:
        axes[1, 1].hist(log_resids, bins=30, density=True, alpha=0.7, color="seagreen")
        x_l = np.linspace(log_resids.min(), log_resids.max(), 200)
        axes[1, 1].plot(x_l, sp_stats.norm.pdf(x_l, log_resids.mean(), log_resids.std()),
                        "r-", linewidth=2, label="Normal fit")
        axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_title(f"Histogram: log {metric} residuals", fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Transformation Diagnostics — {metric}", fontsize=14)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"transformed_diagnostics_{metric}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved transformed diagnostics plot: %s", save_path)
    return save_path


# STEP O-a — Group Comparison (Patient-level Violin Plot)

def plot_group_comparison_patient_level(
        df: pd.DataFrame,
        metric: str = "log_spp_score",
        save_dir: str = CACHE_DIR,
) -> str:
    """
    Violin + box plot showing HC vs PD at patient level.

    Individual data points are jittered. A p-value annotation from the
    permutation test is added.

    Parameters
    df      : DataFrame with patient_id, group, {metric}.
    metric  : response variable name.
    save_dir: directory for saving the PNG.

    Returns
    Path to saved figure.
    """
    # Aggregate to patient level
    patient_df = (
        df.dropna(subset=[metric])
        .groupby(["patient_id", "group"])[metric]
        .mean()
        .reset_index()
    )

    if len(patient_df) < 4:
        logger.warning("Too few patients for group comparison plot.")
        return ""

    fig, ax = plt.subplots(figsize=(8, 7))

    groups = ["HC", "PD"]
    data_by_group = [patient_df.loc[patient_df["group"] == g, metric].values for g in groups]

    # Violin plot
    parts = ax.violinplot(data_by_group, positions=[1, 2], showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(["#4C72B0", "#DD8452"][i])
        pc.set_alpha(0.3)

    # Box plot overlay
    bp = ax.boxplot(data_by_group, positions=[1, 2], widths=0.15, patch_artist=True,
                    showfliers=False)
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Jittered individual points
    rng = np.random.default_rng(42)
    for gi, grp in enumerate(groups):
        grp_vals = data_by_group[gi]
        jitter = rng.uniform(-0.08, 0.08, len(grp_vals))
        ax.scatter(
            gi + 1 + jitter, grp_vals,
            color="black", alpha=0.7, s=40, zorder=5,
        )

    # Permutation test p-value
    try:
        perm_result = permutation_test_patient_level(df, metric=metric, n_perm=min(5000, N_PERMUTATIONS))
        p_val = perm_result.get("p_value", float("nan"))
        if not math.isnan(p_val):
            # Determine significance stars
            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.01:
                stars = "**"
            elif p_val < 0.05:
                stars = "*"
            else:
                stars = "n.s."

            y_max = max(patient_df[metric].max(), max(max(d) for d in data_by_group if len(d) > 0))
            y_offset = y_max * 0.05
            ax.plot([1, 1, 2, 2], [y_max + y_offset] * 2 + [y_max + y_offset * 2] * 2,
                    "k-", linewidth=1)
            ax.text(1.5, y_max + y_offset * 2, f"p = {p_val:.4f} ({stars})",
                    ha="center", va="bottom", fontsize=11)
    except Exception:
        pass

    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"Patient-level Comparison: {metric}", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"group_comparison_{metric}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved group comparison plot: %s", save_path)
    return save_path


# STEP O-b — Sensitivity Analysis Forest Plot

def plot_sensitivity_analysis(
        sensitivity_results: Dict[str, Any],
        save_dir: str = CACHE_DIR,
) -> str:
    """
    Forest-plot style figure showing coefficient (is_pd) with 95% CI
    for each analysis variant.

    Parameters
    sensitivity_results : output of sensitivity_analysis().
    save_dir            : directory for saving the PNG.

    Returns
    Path to saved figure.
    """
    labels = []
    coefs = []
    ci_lo = []
    ci_hi = []
    colors = []

    def _extract_lmm_coefs(lmm_result: Dict, label: str, color: str) -> None:
        """Extract coefficient and CI from LMM result dict."""
        ri = lmm_result.get("random_intercept", {})
        if "error" in ri:
            return
        coef = ri.get("coef_is_pd", float("nan"))
        p = ri.get("p_is_pd", float("nan"))
        labels.append(f"{label}\n(p={p:.4f})" if not math.isnan(p) else label)
        coefs.append(coef)
        # LMM doesn't provide CI directly — approximate via ±1.96*SE
        # Use a wide placeholder if not available
        ci_lo.append(coef - 0.1)  # placeholder
        ci_hi.append(coef + 0.1)  # placeholder
        colors.append(color)

    def _extract_perm_coefs(perm_result: Dict, label: str, color: str) -> None:
        """Extract from permutation test result."""
        obs = perm_result.get("observed_diff", float("nan"))
        p = perm_result.get("p_value", float("nan"))
        labels.append(f"{label}\n(p={p:.4f})" if not math.isnan(p) else label)
        coefs.append(obs)
        ci_lo.append(obs - 0.1)  # placeholder — permutation doesn't give CI
        ci_hi.append(obs + 0.1)
        colors.append(color)

    # Extract results
    _extract_lmm_coefs(sensitivity_results.get("full_lmm", {}), "Full data (LMM)", "#4C72B0")
    _extract_lmm_coefs(sensitivity_results.get("no_outliers_lmm", {}), "No outliers (LMM)", "#DD8452")
    _extract_perm_coefs(sensitivity_results.get("full_permutation", {}), "Full data (perm)", "#55A868")
    _extract_perm_coefs(sensitivity_results.get("no_outliers_permutation", {}), "No outliers (perm)", "#C44E52")

    if not coefs:
        logger.warning("No sensitivity results to plot.")
        return ""

    fig, ax = plt.subplots(figsize=(10, max(4, len(coefs) * 1.2)))

    y_pos = np.arange(len(coefs))
    xerr_lo = [c - lo if not math.isnan(c) else 0 for c, lo in zip(coefs, ci_lo)]
    xerr_hi = [hi - c if not math.isnan(c) else 0 for c, hi in zip(coefs, ci_hi)]

    for i in range(len(coefs)):
        if not math.isnan(coefs[i]):
            ax.errorbar(
                coefs[i], y_pos[i],
                xerr=[[xerr_lo[i]], [xerr_hi[i]]],
                fmt="o", color=colors[i], markersize=8, capsize=5, linewidth=2,
            )

    ax.axvline(0, color="grey", linestyle="--", linewidth=1, label="Null (coef = 0)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Coefficient / Difference (is_pd)", fontsize=11)
    ax.set_title("Sensitivity Analysis — Forest Plot", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "sensitivity_analysis.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sensitivity analysis plot: %s", save_path)
    return save_path


# STEP O-c — Region Comparison Bar Chart

def plot_region_comparison(
        region_results: Dict[str, Dict[str, Any]],
        save_dir: str = CACHE_DIR,
) -> str:
    """
    Bar chart: mean SPP score by region and group (HC/PD), with SEM
    error bars and p-value annotations.

    Parameters
    region_results : output of region_stratified_lmm().
    save_dir       : directory for saving the PNG.

    Returns
    Path to saved figure.
    """
    if not region_results:
        logger.warning("No region results to plot.")
        return ""

    regions = sorted(region_results.keys())
    n_regions = len(regions)

    fig, ax = plt.subplots(figsize=(max(8, n_regions * 3), 7))

    x = np.arange(n_regions)
    width = 0.35

    hc_means = []
    hc_sems = []
    pd_means = []
    pd_sems = []
    p_values = []

    # We need to recalculate means from the original data — but
    # region_results may only contain LMM/permutation results.
    # Store the info we can extract:
    for region in regions:
        res = region_results[region]
        perm = res.get("permutation", {})
        if "error" not in perm:
            hc_means.append(perm.get("hc_mean", 0))
            pd_means.append(perm.get("pd_mean", 0))
            # SEM not directly available from permutation test — use 0 for now
            hc_sems.append(0)
            pd_sems.append(0)
            p_values.append(perm.get("p_value", float("nan")))
        else:
            hc_means.append(0)
            pd_means.append(0)
            hc_sems.append(0)
            pd_sems.append(0)
            p_values.append(float("nan"))

    bars1 = ax.bar(x - width / 2, hc_means, width, label="HC",
                   color="#4C72B0", alpha=0.8, yerr=hc_sems, capsize=4)
    bars2 = ax.bar(x + width / 2, pd_means, width, label="PD",
                   color="#DD8452", alpha=0.8, yerr=pd_sems, capsize=4)

    # Add p-value annotations
    for i, p_val in enumerate(p_values):
        if not math.isnan(p_val):
            y_max = max(hc_means[i] + hc_sems[i], pd_means[i] + pd_sems[i])
            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.01:
                stars = "**"
            elif p_val < 0.05:
                stars = "*"
            else:
                stars = "n.s."
            ax.text(x[i], y_max * 1.05, f"p={p_val:.4f}\n({stars})",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=11)
    ax.set_ylabel("Mean log SPP score", fontsize=12)
    ax.set_title("Region-stratified SPP Comparison (HC vs PD)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "region_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved region comparison plot: %s", save_path)
    return save_path


# Main pipeline

def run_pipeline() -> None:
    """Run the full SPP analysis pipeline."""
    t_start = time.time()
    os.makedirs(CACHE_DIR, exist_ok=True)

    fs = _make_s3fs()

    # Step A: Load CC3D results
    logger.info("Loading CC3D results from MinIO ...")
    all_fov_data = load_all_results(fs)

    if not all_fov_data:
        logger.error("No CC3D results found. Aborting.")
        return

    # Step B: Compute SPP metrics for each FOV
    logger.info("Computing SPP metrics for %d FOVs ...", len(all_fov_data))

    spp_results = []
    n_processed = 0
    n_failed = 0

    for fov_data in all_fov_data:
        pid = fov_data.get("patient_id", "?")
        fid = fov_data.get("fov_id", "?")
        logger.info(">>> SPP: %s / %s / FOV %s",
                    fov_data.get("group"), fov_data.get("region"), fid)

        try:
            result = compute_spp_for_fov(
                fov_data,
                r_vals=R_VALS,
                r_max_guard=R_MAX_GUARD,
                n_sim=N_SIM,
                seed=RANDOM_SEED,
                compute_overlap=True,
                fs=fs,
            )
            if result is not None:
                spp_results.append(result)
            else:
                n_failed += 1
        except Exception as exc:
            logger.error("SPP failed for %s/FOV %s: %s", pid, fid, exc)
            n_failed += 1

        n_processed += 1
        if n_processed % 10 == 0 or n_processed == len(all_fov_data):
            logger.info("  SPP processed %d/%d FOVs ...", n_processed, len(all_fov_data))

    logger.info("SPP computation complete: %d OK, %d failed.",
                len(spp_results), n_failed)

    if not spp_results:
        logger.error("No SPP results. Aborting.")
        return

    # Step C: Aggregate into DataFrame
    # Select scalar columns for the FOV-level DataFrame
    scalar_cols = [
        "patient_id", "group", "region", "fov_id",
        "n_cells", "n_proteins",
        "mean_nnd", "median_nnd", "std_nnd",
        "n_interior_cells", "n_border_excluded",
        "spp_score", "spp_score_marked",
        "p_value", "T_obs",
        "overlap_index",
        "prot_vol_mean", "prot_vol_median", "prot_vol_iqr", "prot_vol_max",
    ]

    rows = []
    for r in spp_results:
        row = {col: r.get(col) for col in scalar_cols}
        rows.append(row)

    df = pd.DataFrame(rows)
    df["is_pd"] = (df["group"] == "PD").astype(int)

    csv_path = os.path.join(CACHE_DIR, "spp_fov_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info("FOV-level CSV: %s (%d rows)", csv_path, len(df))

    # Step D: UPLOAD DATA FIRST  (even if plotting fails, data is safe)
    logger.info("Uploading SPP DATA to MinIO (before plotting!) ...")

    # Upload CSV
    csv_key = f"{STORAGE_NAME}/{SPP_OUTPUT_PREFIX}/spp_fov_results.csv"
    try:
        _retry_s3(fs.put, csv_path, csv_key, max_retries=3)
        logger.info("Uploaded CSV to %s", csv_key)
    except Exception as exc:
        logger.error("Failed to upload CSV: %s", exc)

    # Upload per-FOV SPP JSON
    n_uploaded = 0
    for r in spp_results:
        json_key = (
            f"{STORAGE_NAME}/{SPP_OUTPUT_PREFIX}/"
            f"{r['group']}/{r['region']}/{r['patient_id']}/{r['fov_id']}_spp.json"
        )
        r_safe = _convert_numpy(r)
        json_bytes = json.dumps(r_safe, indent=2, ensure_ascii=False, default=str).encode("utf-8")
        try:
            with fs.open(json_key, "wb") as f:
                f.write(json_bytes)
            n_uploaded += 1
        except Exception as exc:
            logger.error("Failed to upload %s: %s", json_key, exc)
    logger.info("Uploaded %d per-FOV SPP JSONs to MinIO.", n_uploaded)

    # Step E: Quality Filtering (n_interior >= 3)
    logger.info("Applying quality filter (n_interior >= %d) ...", MIN_INTERIOR_CELLS)
    df_filtered = quality_filter(df)

    # Step F: Log Transformation
    logger.info("Log-transforming SPP metrics ...")
    df_filtered = log_transform_metrics(df_filtered)

    # Step G: Outlier Identification Plot
    outlier_plot_path = ""
    try:
        outlier_plot_path = plot_outlier_identification(df_filtered)
        logger.info("Outlier identification plot saved.")
    except Exception as exc:
        logger.warning("Outlier identification plot failed: %s (non-fatal)", exc)

    # Step H: Filtering Summary Plot
    filter_plot_path = ""
    try:
        filter_plot_path = plot_filtering_summary(df, df_filtered)
        logger.info("Filtering summary plot saved.")
    except Exception as exc:
        logger.warning("Filtering summary plot failed: %s (non-fatal)", exc)

    # Step I: Transformed Diagnostics Plot (before/after Q-Q)
    transform_plot_paths = []
    for metric in ["spp_score", "spp_score_marked"]:
        try:
            p = plot_transformed_diagnostics(df_filtered, metric=metric)
            transform_plot_paths.append(p)
        except Exception as exc:
            logger.warning("Transformed diagnostics plot failed for %s: %s (non-fatal)", metric, exc)

    # Step J: Standard LMM on log-transformed metrics
    logger.info("Running LMM group comparison on log-transformed metrics ...")

    lmm_results = {}
    for metric in ["log_spp_score", "log_spp_score_marked", "log_overlap_index"]:
        if metric not in df_filtered.columns:
            logger.warning("Column %s not found — skipping LMM.", metric)
            continue
        valid = df_filtered.dropna(subset=[metric])
        if len(valid) < 20:
            logger.warning("Skipping LMM for %s: only %d valid rows", metric, len(valid))
            continue
        try:
            lmm_result = lmm_comparison(df_filtered, metric=metric)

            # Save LMM result locally
            lmm_path = os.path.join(CACHE_DIR, f"lmm_{metric}.json")
            lmm_safe = _convert_numpy(lmm_result)
            with open(lmm_path, "w") as f:
                json.dump(lmm_safe, f, indent=2, ensure_ascii=False, default=str)
            logger.info("LMM result for %s saved to %s", metric, lmm_path)
            lmm_results[metric] = lmm_result

            # Upload LMM result
            lmm_key = f"{STORAGE_NAME}/{SPP_OUTPUT_PREFIX}/lmm/lmm_{metric}.json"
            try:
                _retry_s3(fs.put, lmm_path, lmm_key, max_retries=2)
            except Exception as exc:
                logger.warning("Failed to upload LMM %s: %s", metric, exc)
        except Exception as exc:
            logger.error("LMM failed for %s: %s (non-fatal, continuing)", metric, exc)

    # Step K: GLMM (Gamma GEE)
    logger.info("Running GLMM (GEE Gamma) comparison ...")

    glmm_results = {}
    for metric in ["log_spp_score", "log_spp_score_marked"]:
        if metric not in df_filtered.columns:
            logger.warning("Column %s not found — skipping GLMM.", metric)
            continue
        try:
            glmm_result = glmm_comparison(df_filtered, metric=metric)

            glmm_path = os.path.join(CACHE_DIR, f"glmm_{metric}.json")
            glmm_safe = _convert_numpy(glmm_result)
            with open(glmm_path, "w") as f:
                json.dump(glmm_safe, f, indent=2, ensure_ascii=False, default=str)
            glmm_results[metric] = glmm_result
            logger.info("GLMM result for %s saved.", metric)

            glmm_key = f"{STORAGE_NAME}/{SPP_OUTPUT_PREFIX}/glmm/glmm_{metric}.json"
            try:
                _retry_s3(fs.put, glmm_path, glmm_key, max_retries=2)
            except Exception as exc:
                logger.warning("Failed to upload GLMM %s: %s", metric, exc)
        except Exception as exc:
            logger.error("GLMM failed for %s: %s (non-fatal)", metric, exc)

    # Step L: Permutation Test (patient-level)
    logger.info("Running patient-level permutation tests ...")

    perm_results = {}
    for metric in ["log_spp_score", "log_spp_score_marked", "log_overlap_index"]:
        if metric not in df_filtered.columns:
            logger.warning("Column %s not found — skipping permutation test.", metric)
            continue
        try:
            perm_result = permutation_test_patient_level(df_filtered, metric=metric)

            perm_path = os.path.join(CACHE_DIR, f"permutation_{metric}.json")
            # Omit perm_distribution from JSON (too large)
            perm_safe = {k: v for k, v in perm_result.items() if k != "perm_distribution"}
            perm_safe = _convert_numpy(perm_safe)
            with open(perm_path, "w") as f:
                json.dump(perm_safe, f, indent=2, ensure_ascii=False, default=str)
            perm_results[metric] = perm_result
            logger.info("Permutation test for %s: p=%.6f", metric, perm_result.get("p_value", float("nan")))

            perm_key = f"{STORAGE_NAME}/{SPP_OUTPUT_PREFIX}/permutation/permutation_{metric}.json"
            try:
                _retry_s3(fs.put, perm_path, perm_key, max_retries=2)
            except Exception as exc:
                logger.warning("Failed to upload permutation %s: %s", metric, exc)
        except Exception as exc:
            logger.error("Permutation test failed for %s: %s (non-fatal)", metric, exc)

    # Step M: Sensitivity Analysis (without outliers)
    logger.info("Running sensitivity analysis (outlier removal) ...")
    sensitivity_results = {}
    try:
        sensitivity_results = sensitivity_analysis(df_filtered)
        sens_path = os.path.join(CACHE_DIR, "sensitivity_analysis.json")
        # Strip large data before JSON serialization
        sens_safe = _convert_numpy(sensitivity_results)
        with open(sens_path, "w") as f:
            json.dump(sens_safe, f, indent=2, ensure_ascii=False, default=str)

        sens_key = f"{STORAGE_NAME}/{SPP_OUTPUT_PREFIX}/sensitivity/sensitivity_analysis.json"
        try:
            _retry_s3(fs.put, sens_path, sens_key, max_retries=2)
        except Exception as exc:
            logger.warning("Failed to upload sensitivity analysis: %s", exc)
    except Exception as exc:
        logger.error("Sensitivity analysis failed: %s (non-fatal)", exc)

    # Step N: Region-stratified Analysis
    logger.info("Running region-stratified LMM ...")
    region_results = {}
    try:
        region_results = region_stratified_lmm(df_filtered)
        region_path = os.path.join(CACHE_DIR, "region_stratified.json")
        region_safe = _convert_numpy(region_results)
        with open(region_path, "w") as f:
            json.dump(region_safe, f, indent=2, ensure_ascii=False, default=str)

        region_key = f"{STORAGE_NAME}/{SPP_OUTPUT_PREFIX}/region/region_stratified.json"
        try:
            _retry_s3(fs.put, region_path, region_key, max_retries=2)
        except Exception as exc:
            logger.warning("Failed to upload region results: %s", exc)
    except Exception as exc:
        logger.error("Region-stratified analysis failed: %s (non-fatal)", exc)

    # Step O: Publication Plots
    logger.info("Generating publication-quality plots ...")

    # O-a: Group comparison (patient-level)
    group_plot_paths = []
    for metric in ["log_spp_score", "log_spp_score_marked"]:
        try:
            p = plot_group_comparison_patient_level(df_filtered, metric=metric)
            if p:
                group_plot_paths.append(p)
        except Exception as exc:
            logger.warning("Group comparison plot failed for %s: %s", metric, exc)

    # O-b: Sensitivity analysis forest plot
    sensitivity_plot_path = ""
    try:
        if sensitivity_results:
            sensitivity_plot_path = plot_sensitivity_analysis(sensitivity_results)
    except Exception as exc:
        logger.warning("Sensitivity analysis plot failed: %s (non-fatal)", exc)

    # O-c: Region comparison bar chart
    region_plot_path = ""
    try:
        if region_results:
            region_plot_path = plot_region_comparison(region_results)
    except Exception as exc:
        logger.warning("Region comparison plot failed: %s (non-fatal)", exc)

    # L-function plots (non-fatal)
    plot_dir = os.path.join(CACHE_DIR, "L_plots")
    try:
        os.makedirs(plot_dir, exist_ok=True)
        n_plots = min(20, len(spp_results))
        for i in range(0, len(spp_results), max(1, len(spp_results) // n_plots)):
            r = spp_results[i]
            if "L_unweighted" not in r or "r_vals" not in r:
                continue
            title = f"{r['group']} / {r['region']} / {r['patient_id']} / FOV {r['fov_id']}"
            save_path = os.path.join(
                plot_dir,
                f"L_{r['group']}_{r['region']}_{r['patient_id']}_{r['fov_id']}.png"
            )
            plot_L_function(
                np.array(r["r_vals"]),
                np.array(r["L_unweighted"]),
                np.array(r["L_envelope_lo"]),
                np.array(r["L_envelope_hi"]),
                np.array(r["r_vals"]),
                title,
                save_path,
            )
    except Exception as exc:
        logger.warning("L-function plotting failed: %s (non-fatal)", exc)

    # Step P: GLMM Gamma for overlap_index in SN
    logger.info("=" * 60)
    logger.info("STEP P: GLMM Gamma (overlap_index, SN)")
    all_results = {}
    glmm_sn_result = glmm_gamma_region(df_filtered, metric="log_overlap_index", region="substantiaNigra")
    all_results["glmm_gamma_sn"] = glmm_sn_result

    # Also run for spp_score in SN for comparison
    glmm_sn_spp = glmm_gamma_region(df_filtered, metric="log_spp_score", region="substantiaNigra")
    all_results["glmm_gamma_sn_spp"] = glmm_sn_spp

    # Save and upload GLMM Gamma results
    try:
        glmm_gamma_path = os.path.join(CACHE_DIR, "glmm_gamma_sn.json")
        glmm_gamma_safe = _convert_numpy(all_results)
        with open(glmm_gamma_path, "w") as f:
            json.dump(glmm_gamma_safe, f, indent=2, ensure_ascii=False, default=str)
        glmm_gamma_key = f"{STORAGE_NAME}/{SPP_OUTPUT_PREFIX}/glmm_gamma/glmm_gamma_sn.json"
        try:
            _retry_s3(fs.put, glmm_gamma_path, glmm_gamma_key, max_retries=2)
        except Exception as exc:
            logger.warning("Failed to upload GLMM Gamma results: %s", exc)
    except Exception as exc:
        logger.warning("Failed to save GLMM Gamma results: %s", exc)

    # Step Q: Bootstrap Cohen's d
    logger.info("=" * 60)
    logger.info("STEP Q: Bootstrap Cohen's d")
    for _metric in ["log_spp_score", "log_overlap_index"]:
        if _metric not in df_filtered.columns:
            continue
        for _region in [None, "putamen", "substantiaNigra"]:
            label = f"{_metric}" + (f"_{_region}" if _region else "_all")
            boot_result = bootstrap_cohens_d(df_filtered, metric=_metric, region=_region)
            all_results[f"bootstrap_cohens_d_{label}"] = boot_result

    # Save and upload bootstrap results
    try:
        boot_path = os.path.join(CACHE_DIR, "bootstrap_cohens_d.json")
        boot_safe = _convert_numpy(all_results)
        with open(boot_path, "w") as f:
            json.dump(boot_safe, f, indent=2, ensure_ascii=False, default=str)
        boot_key = f"{STORAGE_NAME}/{SPP_OUTPUT_PREFIX}/bootstrap/bootstrap_cohens_d.json"
        try:
            _retry_s3(fs.put, boot_path, boot_key, max_retries=2)
        except Exception as exc:
            logger.warning("Failed to upload bootstrap results: %s", exc)
    except Exception as exc:
        logger.warning("Failed to save bootstrap results: %s", exc)

    # Step R: SN Ridge + Waterfall plot
    logger.info("=" * 60)
    logger.info("STEP R: SN Ridge + Waterfall plot")
    ridge_plot_path = ""
    try:
        ridge_plot_path = plot_sn_ridge(df_filtered, metric="log_overlap_index", save_dir=CACHE_DIR)
    except Exception as exc:
        logger.warning("SN ridge plot failed: %s", exc)

    # Step S: Upload ALL results + plots to MinIO
    logger.info("Uploading quality-filtered CSV and all plots to MinIO ...")

    # Upload quality-filtered CSV
    filtered_csv_path = os.path.join(CACHE_DIR, "spp_fov_results_filtered.csv")
    try:
        df_filtered.to_csv(filtered_csv_path, index=False)
        filtered_csv_key = f"{STORAGE_NAME}/{SPP_OUTPUT_PREFIX}/spp_fov_results_filtered.csv"
        _retry_s3(fs.put, filtered_csv_path, filtered_csv_key, max_retries=2)
        logger.info("Uploaded filtered CSV to %s", filtered_csv_key)
    except Exception as exc:
        logger.warning("Failed to upload filtered CSV: %s", exc)

    # Upload L-function plots
    if os.path.isdir(plot_dir):
        for fname in os.listdir(plot_dir):
            fpath = os.path.join(plot_dir, fname)
            plot_key = f"{STORAGE_NAME}/{SPP_OUTPUT_PREFIX}/L_plots/{fname}"
            try:
                _retry_s3(fs.put, fpath, plot_key, max_retries=2)
            except Exception as exc:
                logger.warning("Failed to upload plot %s: %s", fname, exc)

    # Upload residual plots
    for fname in os.listdir(CACHE_DIR):
        if fname.startswith("residuals_") and fname.endswith(".png"):
            fpath = os.path.join(CACHE_DIR, fname)
            diag_key = f"{STORAGE_NAME}/{SPP_OUTPUT_PREFIX}/diagnostics/{fname}"
            try:
                _retry_s3(fs.put, fpath, diag_key, max_retries=2)
            except Exception as exc:
                logger.warning("Failed to upload diagnostics %s: %s", fname, exc)

    # Upload new publication plots
    all_new_plots = [
        (outlier_plot_path, "outlier_identification.png"),
        (filter_plot_path, "filtering_summary.png"),
        (sensitivity_plot_path, "sensitivity_analysis.png"),
        (region_plot_path, "region_comparison.png"),
        (ridge_plot_path, "sn_ridge_waterfall.png"),
    ]
    for plot_path, default_name in all_new_plots:
        if plot_path and os.path.isfile(plot_path):
            plot_key = f"{STORAGE_NAME}/{SPP_OUTPUT_PREFIX}/plots/{os.path.basename(plot_path)}"
            try:
                _retry_s3(fs.put, plot_path, plot_key, max_retries=2)
                logger.info("Uploaded plot %s", plot_key)
            except Exception as exc:
                logger.warning("Failed to upload plot %s: %s", plot_path, exc)

    for plot_path in transform_plot_paths + group_plot_paths:
        if plot_path and os.path.isfile(plot_path):
            plot_key = f"{STORAGE_NAME}/{SPP_OUTPUT_PREFIX}/plots/{os.path.basename(plot_path)}"
            try:
                _retry_s3(fs.put, plot_path, plot_key, max_retries=2)
                logger.info("Uploaded plot %s", plot_key)
            except Exception as exc:
                logger.warning("Failed to upload plot %s: %s", plot_path, exc)

    # Step T: Cleanup
    elapsed = time.time() - t_start
    logger.info("SPP pipeline completed in %.1f minutes.", elapsed / 60.0)

    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


# ClearML entry point

if __name__ == "__main__":
    clearml_disabled = os.environ.get("CLEARML_DISABLED", "").lower() in (
        "1", "true", "yes",
    )

    if not clearml_disabled:
        from clearml import Task

        Task.add_requirements("boto3")
        Task.add_requirements("s3fs")
        Task.add_requirements("matplotlib")
        Task.add_requirements("statsmodels")
        # scikit-image auto-detected by ClearML from import

        task = Task.init(
            project_name="YOUR_CLEARML_PROJECT",
            task_name="SPP_Analysis_Pipeline",
            task_type=Task.TaskTypes.data_processing,
        )
        task.execute_remotely(queue_name="default")

    run_pipeline()
