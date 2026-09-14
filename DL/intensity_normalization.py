"""
Intensity Normalization Pipeline for Microglia 2.75D Data
==========================================================

Critical preprocessing module that addresses three key data quality
challenges in fluorescence microscopy for PD vs HC classification:

1. **Technical artifacts** — thermal noise floor and saturated pixels
   are clipped using 1st/99th percentile-based contrast limiting
   before Min-Max scaling to [0, 1].

2. **Cross-patient alignment** — systematic intensity differences
   between patients (due to sample preparation, imaging conditions,
   staining batch effects) are removed by aligning each patient's
   channel statistics to the cohort-wide median (target_stats).

3. **Biological bias detection** — a Spearman rank-correlation test
   checks whether the mean raw channel intensity correlates with
   the diagnosis label (PD vs HC).  If significant correlation is
   found, normalization is MANDATORY because the model could learn
   intensity as a confound rather than true pathology features.

Architecture
------------

The pipeline has three stages, applied in order:

    Stage 1: Cross-Patient Alignment (per-patient, applied per-patch)
             ┌──────────────────────────────────────────────┐
             │  Align RAW data to the cohort target using    │
             │  z-score normalization with patient-level     │
             │  channel statistics, then rescale to the      │
             │  target (median) statistics across all        │
             │  patients.  Applied on RAW data so that the   │
             │  mean/std are on the same scale.              │
             │                                               │
             │  patch_aligned = (patch - mu_patient) /       │
             │                    sigma_patient *             │
             │                    sigma_target + mu_target    │
             └──────────────────────────────────────────────┘

    Stage 2: Percentile Clipping + Min-Max Scaling (per-patch)
             ┌──────────────────────────────────────────────┐
             │  After alignment, clip at 1st/99th percentile │
             │  and Min-Max scale to [0, 1].  Percentiles    │
             │  come from patient-level stats (or per-patch  │
             │  as fallback).  This is done AFTER alignment  │
             │  so the clipping operates on aligned data.    │
             └──────────────────────────────────────────────┘

    Stage 3: Spearman Bias Test (diagnostic, run once before training)
             ┌──────────────────────────────────────────────┐
             │  For each channel, compute mean raw intensity │
             │  per patient.  Test whether intensity ranks   │
             │  correlate with diagnosis labels (0=HC,1=PD). │
             │  If rho is significant (p < alpha), the      │
             │  channel MUST be normalized.                  │
             └──────────────────────────────────────────────┘

Integration with ZarrPatchDataset
----------------------------------

This module is designed to be used in two ways:

A) **Pre-scan mode** (before training):
   Call ``compute_patient_intensity_stats()`` on a dataset that has
   been preloaded (``dataset.preload_data()``).  This computes the
   per-patient channel statistics needed for cross-patient alignment
   and runs the Spearman bias test.  The results can be logged to
   ClearML and saved to MinIO.

B) **Runtime mode** (during __getitem__):
   The ``IntensityNormalizer`` class stores pre-computed patient
   statistics and applies the full normalization pipeline to each
   patch as it is loaded.  It replaces the dataset's internal
   ``_normalize_patch`` method.

Usage example
-------------

    # Pre-scan: compute stats and run bias test
    from intensity_normalization import (
        compute_patient_intensity_stats,
        run_spearman_bias_test,
        IntensityNormalizer,
    )

    dataset = ZarrPatchDataset(...)
    dataset.preload_data()

    stats = compute_patient_intensity_stats(dataset)
    bias_report = run_spearman_bias_test(stats)

    # Create normalizer with pre-computed stats
    normalizer = IntensityNormalizer(
        patient_stats=stats,
        model_type=dataset.model_type,
    )

    # Replace dataset's normalization
    dataset._normalizer = normalizer
    dataset.normalization = "intensity_pipeline"

References
----------
- Reinhold et al. (2019): "Whole-slide image color normalization
  and stain separation" — percentile-based clipping for digital
  pathology.
- Zanjani et al. (2023): "Cross-stain normalization in histopathology"
  — cross-patient alignment for microscopy.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# =====================================================================
#  1. Data classes for patient statistics
# =====================================================================

@dataclass
class ChannelStats:
    """Statistics for a single channel of a single patient."""

    patient_id: str
    channel_idx: int      # 0 = pSyn, 1 = IBA1
    mean: float           # mean raw intensity across all patches
    std: float            # std of raw intensity across all patches
    p1: float             # 1st percentile across all patches
    p5: float             # 5th percentile
    p25: float            # 25th percentile
    p50: float            # median (50th percentile)
    p75: float            # 75th percentile
    p99: float            # 99th percentile
    n_patches: int        # number of patches used to compute stats
    n_pixels: int         # total number of pixels (for weighting)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChannelStats":
        return cls(**d)


@dataclass
class PatientStats:
    """Container for all channel statistics of a single patient."""

    patient_id: str
    label: int            # 0 = HC, 1 = PD
    region: str           # 'putamen' or 'substantiaNigra'
    channels: Dict[int, ChannelStats] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "label": self.label,
            "region": self.region,
            "channels": {
                str(k): v.to_dict() for k, v in self.channels.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PatientStats":
        channels = {
            int(k): ChannelStats.from_dict(v)
            for k, v in d["channels"].items()
        }
        return cls(
            patient_id=d["patient_id"],
            label=d["label"],
            region=d["region"],
            channels=channels,
        )


@dataclass
class TargetStats:
    """Cohort-wide target statistics for cross-patient alignment.

    These are the statistics that each patient will be aligned TO.
    Computed as the median across all patients (robust to outliers).

    For each channel, we store:
      - target_mean: median of patient means
      - target_std: median of patient stds
      - target_p1: median of patient 1st percentiles
      - target_p99: median of patient 99th percentiles
    """

    channel_idx: int
    target_mean: float    # median of patient means
    target_std: float     # median of patient stds
    target_p1: float      # median of patient p1 values
    target_p99: float     # median of patient p99 values
    n_patients: int       # number of patients used

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TargetStats":
        return cls(**d)


@dataclass
class SpearmanResult:
    """Result of the Spearman correlation test for a single channel."""

    channel_idx: int
    channel_name: str     # 'pSyn' or 'IBA1'
    rho: float            # Spearman rank correlation coefficient
    p_value: float        # two-sided p-value
    significant: bool     # True if p_value < alpha
    alpha: float          # significance threshold used
    verdict: str          # human-readable interpretation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpearmanResult":
        return cls(**d)


# =====================================================================
#  2. Compute per-patient intensity statistics
# =====================================================================

def compute_patient_intensity_stats(
    dataset,              # ZarrPatchDataset (must be preloaded)
    max_patches_per_patient: int = 500,
    seed: int = 42,
) -> Dict[str, PatientStats]:
    """
    Compute per-patient, per-channel intensity statistics from the
    raw (unnormalized) data in the dataset's RAM cache.

    This function iterates over all patches in the dataset, extracts
    the raw pixel values from the in-memory cache, and computes
    intensity statistics for each (patient, channel) pair.

    Parameters
    ----------
    dataset : ZarrPatchDataset
        A pre-loaded dataset (``preload_data()`` must have been called
        so that ``_raw_cache`` is populated with raw numpy arrays).
    max_patches_per_patient : int
        Maximum number of patches to sample per patient for computing
        statistics.  For patients with many patches (>500), a random
        subset is used to keep computation time reasonable.  Default 500.
    seed : int
        Random seed for reproducible patch subsampling.  Default 42.

    Returns
    -------
    Dict[str, PatientStats]
        Mapping from patient_id to PatientStats containing per-channel
        intensity statistics.

    Notes
    -----
    - Statistics are computed on RAW (unnormalized) data.  This is
      critical because we want to detect biases in the raw signal,
      not artifacts introduced by normalization.
    - For Model A (IBA1 only), only channel index 1 is computed.
    - For Model B (pSyn only), only channel index 0 is computed.
    - For Model C (both), both channels are computed.
    - The function uses the dataset's ``_raw_cache`` dictionary, which
      maps (zarr_path, fov_id) -> numpy array of shape
      (C_needed, Z, Y, X) with raw float32 values.
    """
    rng = np.random.RandomState(seed)

    # Group patches by patient_id
    patient_patches: Dict[str, List[Any]] = {}
    for p in dataset.patches:
        if p.patient_id not in patient_patches:
            patient_patches[p.patient_id] = []
        patient_patches[p.patient_id].append(p)

    # Build pid -> label, region maps
    pid_to_label: Dict[str, int] = {}
    pid_to_region: Dict[str, str] = {}
    for p in dataset.patches:
        if p.patient_id not in pid_to_label:
            pid_to_label[p.patient_id] = p.label
            pid_to_region[p.patient_id] = p.region

    all_patient_stats: Dict[str, PatientStats] = {}

    for pid, patches in patient_patches.items():
        # Subsample if too many patches
        if len(patches) > max_patches_per_patient:
            indices = rng.choice(len(patches), max_patches_per_patient,
                                 replace=False)
            patches_subset = [patches[i] for i in indices]
        else:
            patches_subset = patches

        # Collect raw pixel values per channel
        # For each channel, we accumulate all pixel values across all
        # sampled patches for this patient
        channel_pixels: Dict[int, List[np.ndarray]] = {}

        for p in patches_subset:
            # Load raw patch from cache
            try:
                raw_patch = dataset._load_raw_patch_from_cache(
                    zarr_path=p.zarr_path,
                    fov_id=p.fov_id,
                    y_start=p.y_start,
                    x_start=p.x_start,
                    z_native=p.z_native,
                    is_store_fov=p.is_store_fov,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load patch for stats (patient=%s, "
                    "fov=%s): %s — skipping.",
                    pid, p.fov_id, exc,
                )
                continue

            # Skip patches that are all zeros (returned by fallback
            # when zarr structure is incompatible)
            if raw_patch.max() == 0.0:
                continue

            # Z-padding (same as in __getitem__)
            if p.z_padded:
                raw_patch = dataset._z_pad(raw_patch, dataset.target_z)

            # raw_patch shape: (C_needed, Z, H, W)
            for c_idx in range(raw_patch.shape[0]):
                if c_idx not in channel_pixels:
                    channel_pixels[c_idx] = []
                # Flatten spatial+Z dims to 1D for statistics
                channel_pixels[c_idx].append(raw_patch[c_idx].ravel())

        # Compute statistics per channel
        pstats = PatientStats(
            patient_id=pid,
            label=pid_to_label[pid],
            region=pid_to_region[pid],
        )

        for c_idx, pixel_list in channel_pixels.items():
            # Concatenate all pixel values for this (patient, channel)
            all_pixels = np.concatenate(pixel_list)

            # Map internal channel index to OME-Zarr channel index
            # dataset._channels_to_load maps model channels to raw channels
            # e.g., Model A: [1] (IBA1), Model B: [0] (pSyn), Model C: [1, 0]
            raw_ch_idx = dataset._channels_to_load[c_idx]

            cs = ChannelStats(
                patient_id=pid,
                channel_idx=raw_ch_idx,
                mean=float(np.mean(all_pixels)),
                std=float(np.std(all_pixels)),
                p1=float(np.percentile(all_pixels, 1)),
                p5=float(np.percentile(all_pixels, 5)),
                p25=float(np.percentile(all_pixels, 25)),
                p50=float(np.percentile(all_pixels, 50)),
                p75=float(np.percentile(all_pixels, 75)),
                p99=float(np.percentile(all_pixels, 99)),
                n_patches=len(patches_subset),
                n_pixels=len(all_pixels),
            )
            pstats.channels[raw_ch_idx] = cs

        all_patient_stats[pid] = pstats
        logger.info(
            "Patient %s (label=%d, region=%s): %d channels, "
            "stats computed from %d patches",
            pid, pstats.label, pstats.region,
            len(pstats.channels), len(patches_subset),
        )

    return all_patient_stats


# =====================================================================
#  3. Compute target statistics (median across patients)
# =====================================================================

def compute_target_stats(
    patient_stats: Dict[str, PatientStats],
) -> Dict[int, TargetStats]:
    """
    Compute cohort-wide target statistics as the **median** across
    all patients for each channel.

    Using the median (instead of the mean) makes the target robust
    to outlier patients with unusually bright or dim images.

    The target statistics are used for cross-patient alignment:
    each patient's channel is transformed so that its statistics
    match the target, removing systematic inter-patient variability.

    Parameters
    ----------
    patient_stats : Dict[str, PatientStats]
        Per-patient statistics from ``compute_patient_intensity_stats()``.

    Returns
    -------
    Dict[int, TargetStats]
        Mapping from channel_idx (0=pSyn, 1=IBA1) to TargetStats
        containing the cohort-wide target statistics.

    Notes
    -----
    The alignment formula for a patient's channel is:

        aligned = (raw - mu_patient) / sigma_patient * sigma_target + mu_target

    This is a standard z-score + rescale transformation.  It preserves
    the within-patient distribution shape while aligning the central
    tendency and spread to the cohort median.
    """
    # Collect per-channel lists of statistics across patients
    channel_data: Dict[int, Dict[str, List[float]]] = {}

    for pid, ps in patient_stats.items():
        for ch_idx, cs in ps.channels.items():
            if ch_idx not in channel_data:
                channel_data[ch_idx] = {
                    "means": [],
                    "stds": [],
                    "p1s": [],
                    "p99s": [],
                }
            channel_data[ch_idx]["means"].append(cs.mean)
            channel_data[ch_idx]["stds"].append(cs.std)
            channel_data[ch_idx]["p1s"].append(cs.p1)
            channel_data[ch_idx]["p99s"].append(cs.p99)

    target_stats: Dict[int, TargetStats] = {}

    for ch_idx, data in channel_data.items():
        n_patients = len(data["means"])
        ts = TargetStats(
            channel_idx=ch_idx,
            target_mean=float(np.median(data["means"])),
            target_std=float(np.median(data["stds"])),
            target_p1=float(np.median(data["p1s"])),
            target_p99=float(np.median(data["p99s"])),
            n_patients=n_patients,
        )
        target_stats[ch_idx] = ts

        logger.info(
            "Target stats for channel %d (n_patients=%d): "
            "mean=%.4f, std=%.4f, p1=%.4f, p99=%.4f",
            ch_idx, n_patients,
            ts.target_mean, ts.target_std,
            ts.target_p1, ts.target_p99,
        )

    return target_stats


# =====================================================================
#  4. Spearman correlation bias test
# =====================================================================

# Channel name mapping for human-readable output
_CHANNEL_NAMES = {0: "pSyn", 1: "IBA1"}


def run_spearman_bias_test(
    patient_stats: Dict[str, PatientStats],
    alpha: float = 0.05,
) -> List[SpearmanResult]:
    """
    Test whether mean raw channel intensity correlates with diagnosis.

    For each channel, the mean raw intensity per patient is computed
    and a Spearman rank correlation test is performed against the
    binary diagnosis label (0=HC, 1=PD).  If the correlation is
    statistically significant (p < alpha), it means that the channel's
    raw intensity is a confound — the model could learn to distinguish
    PD from HC based on brightness alone, rather than morphological
    features.

    When a significant correlation is detected, cross-patient
    normalization (Stage 2) is MANDATORY for that channel.

    Parameters
    ----------
    patient_stats : Dict[str, PatientStats]
        Per-patient statistics from ``compute_patient_intensity_stats()``.
    alpha : float
        Significance level for the Spearman test.  Default 0.05.

    Returns
    -------
    List[SpearmanResult]
        One result per channel with the correlation coefficient,
        p-value, significance flag, and a human-readable verdict.

    Notes
    -----
    - Spearman's rho is used instead of Pearson's r because the
      diagnosis label is binary (0/1), making the relationship
      potentially non-linear.  Spearman's rank-based approach is
      more appropriate for ordinal/binary variables.
    - A positive rho means PD patients tend to have HIGHER intensity
      for this channel; negative means PD patients have LOWER intensity.
    - The test has limited power with small sample sizes (e.g., 14 PD
      + 13 HC = 27 patients), so borderline p-values should be
      interpreted with caution.
    """
    # Collect per-channel data: (mean_intensity, label) for each patient
    channel_data: Dict[int, Tuple[List[float], List[int]]] = {}

    for pid, ps in patient_stats.items():
        for ch_idx, cs in ps.channels.items():
            if ch_idx not in channel_data:
                channel_data[ch_idx] = ([], [])
            channel_data[ch_idx][0].append(cs.mean)
            channel_data[ch_idx][1].append(ps.label)

    results: List[SpearmanResult] = []

    for ch_idx, (means, labels) in channel_data.items():
        means_arr = np.array(means)
        labels_arr = np.array(labels)

        # Check that both classes are present
        n_classes = len(np.unique(labels_arr))
        if n_classes < 2:
            result = SpearmanResult(
                channel_idx=int(ch_idx),
                channel_name=_CHANNEL_NAMES.get(ch_idx, f"ch{ch_idx}"),
                rho=0.0,
                p_value=1.0,
                significant=False,
                alpha=float(alpha),
                verdict=(
                    "SKIP: Only one class present — cannot compute "
                    "correlation."
                ),
            )
            results.append(result)
            continue

        # Spearman rank correlation
        rho, p_value = scipy_stats.spearmanr(means_arr, labels_arr)

        # Handle NaN (can happen if all values are identical)
        if np.isnan(rho):
            rho = 0.0
            p_value = 1.0

        # Ensure Python native types (not numpy) for JSON serialization
        is_significant = bool(p_value < alpha)

        # Build verdict string
        direction = "HIGHER" if rho > 0 else "LOWER"
        ch_name = _CHANNEL_NAMES.get(ch_idx, f"ch{ch_idx}")

        if is_significant:
            verdict = (
                f"BIAS DETECTED: {ch_name} intensity is significantly "
                f"correlated with diagnosis (rho={rho:.3f}, p={p_value:.4f}). "
                f"PD patients tend to have {direction} intensity. "
                f"CROSS-PATIENT NORMALIZATION IS MANDATORY for this channel."
            )
        else:
            verdict = (
                f"No significant bias: {ch_name} intensity does not "
                f"significantly correlate with diagnosis "
                f"(rho={rho:.3f}, p={p_value:.4f}). "
                f"Normalization still recommended for robustness."
            )

        result = SpearmanResult(
            channel_idx=int(ch_idx),
            channel_name=ch_name,
            rho=float(rho),
            p_value=float(p_value),
            significant=is_significant,
            alpha=float(alpha),
            verdict=verdict,
        )
        results.append(result)

        logger.info(
            "Spearman test [%s]: rho=%.4f, p=%.4f, significant=%s",
            ch_name, rho, p_value, is_significant,
        )
        if is_significant:
            logger.warning(
                "BIAS DETECTED in channel %s! "
                "Cross-patient normalization is MANDATORY.",
                ch_name,
            )

    return results


# =====================================================================
#  5. IntensityNormalizer — the runtime pipeline
# =====================================================================

class IntensityNormalizer:
    """
    Full intensity normalization pipeline for 2.75D patches.

    Applies stages in order:
      1. Cross-patient alignment (z-score + rescale) on RAW data
      2. Percentile clipping + Min-Max scaling to [0, 1]
      3. (Spearman test is pre-run, not applied per-patch)

    IMPORTANT: Alignment MUST be applied on raw data (before
    percentile clipping) because the patient stats (mu, sigma)
    are computed from raw intensities.  Applying them to [0,1]-
    scaled data would produce nonsensical values.

    This class is designed to replace the dataset's internal
    ``_normalize_patch`` method.  After creating an IntensityNormalizer
    with pre-computed statistics, set it on the dataset:

        normalizer = IntensityNormalizer(patient_stats, target_stats,
                                          model_type="C")
        dataset._normalizer = normalizer
        dataset.normalization = "intensity_pipeline"

    Parameters
    ----------
    patient_stats : Dict[str, PatientStats]
        Per-patient statistics from ``compute_patient_intensity_stats()``.
    target_stats : Dict[int, TargetStats]
        Cohort-wide target statistics from ``compute_target_stats()``.
        If None, cross-patient alignment is skipped (Stage 1 only).
    model_type : str
        "A" (IBA1 only), "B" (pSyn only), or "C" (both).
        Determines which channels are present and their mapping.
    use_patient_percentiles : bool
        If True, use the target-level p1/p99 (median across patients)
        for clipping instead of per-patch percentiles.  After alignment,
        all patients share the target statistics, so target percentiles
        are appropriate and stable.  Default True.
    apply_cross_patient_alignment : bool
        If True, apply Stage 1 (cross-patient alignment) on raw data
        before Stage 2 (percentile clipping + Min-Max).  Default True.
    force_alignment : bool
        If True, force cross-patient alignment even if the Spearman
        test did not detect a significant bias.  This is recommended
        for robustness — even without significant bias, alignment
        removes nuisance variability.  Default True.
    """

    # Channel mapping: model_type -> {internal_idx: raw_channel_idx}
    _CHANNEL_MAP = {
        "A": {0: 1},       # IBA1 only (raw channel 1)
        "B": {0: 0},       # pSyn only (raw channel 0)
        "C": {0: 1, 1: 0}, # IBA1 (raw ch 1) + pSyn (raw ch 0)
    }

    def __init__(
        self,
        patient_stats: Dict[str, PatientStats],
        target_stats: Optional[Dict[int, TargetStats]] = None,
        model_type: str = "C",
        use_patient_percentiles: bool = True,
        apply_cross_patient_alignment: bool = True,
        force_alignment: bool = True,
    ) -> None:
        self.model_type = model_type
        self.use_patient_percentiles = use_patient_percentiles
        self.apply_cross_patient_alignment = apply_cross_patient_alignment
        self.force_alignment = force_alignment

        # Channel mapping for this model type
        self._ch_map = self._CHANNEL_MAP[model_type]

        # Store per-patient statistics for lookup by patient_id
        self._patient_stats = patient_stats

        # Store target stats for cross-patient alignment
        self._target_stats = target_stats or {}

        # Pre-compute lookup: patient_id -> {raw_ch_idx: (mean, std, p1, p99)}
        self._patient_lookup: Dict[str, Dict[int, Tuple[float, float, float, float]]] = {}
        for pid, ps in patient_stats.items():
            self._patient_lookup[pid] = {}
            for ch_idx, cs in ps.channels.items():
                self._patient_lookup[pid][ch_idx] = (
                    cs.mean, cs.std, cs.p1, cs.p99,
                )

        logger.info(
            "IntensityNormalizer created: model=%s, "
            "use_patient_percentiles=%s, cross_patient_alignment=%s, "
            "%d patients, %d target channels",
            model_type, use_patient_percentiles,
            apply_cross_patient_alignment,
            len(patient_stats), len(self._target_stats),
        )

    def normalize_patch(
        self,
        patch: np.ndarray,
        patient_id: str,
    ) -> np.ndarray:
        """
        Apply the full normalization pipeline to a single patch.

        Parameters
        ----------
        patch : np.ndarray
            Raw patch of shape ``(C_out, Z, H, W)`` with float32 values.
        patient_id : str
            Patient identifier used to look up per-patient statistics
            for cross-patient alignment.

        Returns
        -------
        np.ndarray
            Normalized patch of the same shape, with values
            in [0, 1] after the full pipeline:
            Stage 1: cross-patient alignment on raw data,
            Stage 2: percentile clipping + Min-Max scaling.
        """
        out = patch.copy()
        n_channels = out.shape[0]

        for c in range(n_channels):
            raw_ch_idx = self._ch_map[c]  # map to OME-Zarr channel index
            ch_data = out[c]  # (Z, H, W) — RAW data

            # ── Stage 1: Cross-patient alignment (on RAW data) ──
            # IMPORTANT: Alignment MUST be applied on raw data because
            # the patient stats (mu, sigma) are computed from raw
            # intensities.  Applying them to [0,1]-scaled data would
            # produce nonsensical values that get clipped to 0 or 1,
            # destroying all information in the patch.
            if self.apply_cross_patient_alignment:
                out[c] = self._stage1_cross_patient_align(
                    ch_data, patient_id, raw_ch_idx,
                )
                ch_data = out[c]  # use aligned data for next stage

            # ── Stage 2: Percentile clipping + Min-Max [0,1] ────
            # After alignment, clip at percentiles and scale to [0,1].
            out[c] = self._stage2_percentile_clip(
                ch_data, patient_id, raw_ch_idx,
            )

        return out

    # -----------------------------------------------------------------
    # Stage 1: Cross-Patient Alignment (applied on RAW data)
    # -----------------------------------------------------------------

    def _stage1_cross_patient_align(
        self,
        ch_data: np.ndarray,
        patient_id: str,
        raw_ch_idx: int,
    ) -> np.ndarray:
        """
        Align patient's RAW channel statistics to the cohort target.

        This MUST be applied to RAW (unnormalized) data because the
        patient stats (mu_patient, sigma_patient) and target stats
        (mu_target, sigma_target) are computed from raw intensities.

        Formula:
            aligned = (data - mu_patient) / sigma_patient
                      * sigma_target + mu_target

        This preserves the within-patient distribution shape while
        aligning the central tendency and spread to the cohort median.
        No clipping is applied here — that's done in Stage 2.
        """
        # Look up patient stats
        patient_ch_stats = self._patient_lookup.get(patient_id, {}).get(raw_ch_idx)
        if patient_ch_stats is None:
            logger.debug(
                "No patient stats for %s ch %d — skipping alignment.",
                patient_id, raw_ch_idx,
            )
            return ch_data

        mu_patient, sigma_patient, _, _ = patient_ch_stats

        # Look up target stats
        target = self._target_stats.get(raw_ch_idx)
        if target is None:
            logger.debug(
                "No target stats for ch %d — skipping alignment.",
                raw_ch_idx,
            )
            return ch_data

        mu_target = target.target_mean
        sigma_target = target.target_std

        # Guard against zero std
        if sigma_patient < 1e-8 or sigma_target < 1e-8:
            logger.warning(
                "Near-zero std for patient %s ch %d "
                "(sigma_p=%.4f, sigma_t=%.4f) — skipping alignment.",
                patient_id, raw_ch_idx, sigma_patient, sigma_target,
            )
            return ch_data

        # Z-score normalize with patient stats, then rescale to target
        # No clipping here — data is still in raw intensity space.
        # Outliers will be handled by Stage 2 (percentile clipping).
        aligned = (ch_data - mu_patient) / sigma_patient * sigma_target + mu_target

        return aligned.astype(np.float32)

    # -----------------------------------------------------------------
    # Stage 2: Percentile Clipping + Min-Max Scaling (after alignment)
    # -----------------------------------------------------------------

    def _stage2_percentile_clip(
        self,
        ch_data: np.ndarray,
        patient_id: str,
        raw_ch_idx: int,
    ) -> np.ndarray:
        """
        Clip at 1st/99th percentile and Min-Max scale to [0, 1].

        This is applied AFTER cross-patient alignment (Stage 1).
        After alignment, all patients have approximately the same
        mean and std, so using per-patch percentiles is appropriate
        and stable.  Patient-level percentiles from raw stats are
        NOT used here because they were computed on raw (pre-alignment)
        data and don't correspond to the aligned intensity range.

        Two modes:
          - ``use_patient_percentiles=True``: use the target-level
            p1/p99 from the pre-computed target statistics (these are
            the median percentiles across all patients and remain
            valid after alignment since all patients are aligned to
            the target).
          - ``use_patient_percentiles=False``: compute p1/p99 from the
            patch itself (most robust after alignment).
        """
        if self.use_patient_percentiles:
            # After alignment, use TARGET percentiles (not patient's raw
            # percentiles, which are in raw intensity space).
            target = self._target_stats.get(raw_ch_idx)
            if target is not None:
                p1 = target.target_p1
                p99 = target.target_p99
            else:
                # Fallback: compute per-patch percentiles
                logger.debug(
                    "No target stats for %s ch %d — "
                    "falling back to per-patch percentiles.",
                    patient_id, raw_ch_idx,
                )
                p1 = np.percentile(ch_data, 1)
                p99 = np.percentile(ch_data, 99)
        else:
            # Per-patch percentiles (most robust after alignment)
            p1 = np.percentile(ch_data, 1)
            p99 = np.percentile(ch_data, 99)

        # Clip and scale
        if p99 - p1 < 1e-8:
            # Constant or near-constant channel -> map to 0
            return np.zeros_like(ch_data)

        clipped = np.clip(ch_data, p1, p99)
        normalized = (clipped - p1) / (p99 - p1)

        return normalized.astype(np.float32)


# =====================================================================
#  6. Convenience: run the full pre-scan pipeline
# =====================================================================

def run_full_pre_scan(
    dataset,
    alpha: float = 0.05,
    max_patches_per_patient: int = 500,
    seed: int = 42,
) -> Tuple[
    Dict[str, PatientStats],
    Dict[int, TargetStats],
    List[SpearmanResult],
]:
    """
    Run the complete pre-scan pipeline: compute patient stats,
    compute target stats, and run the Spearman bias test.

    This should be called once before training, after the dataset
    has been preloaded with ``preload_data()``.

    Parameters
    ----------
    dataset : ZarrPatchDataset
        A pre-loaded dataset.
    alpha : float
        Significance level for the Spearman test.  Default 0.05.
    max_patches_per_patient : int
        Maximum patches to sample per patient for stats.  Default 500.
    seed : int
        Random seed for subsampling.  Default 42.

    Returns
    -------
    patient_stats : Dict[str, PatientStats]
    target_stats : Dict[int, TargetStats]
    spearman_results : List[SpearmanResult]
    """
    logger.info("=" * 60)
    logger.info("INTENSITY PRE-SCAN: computing patient statistics ...")
    logger.info("=" * 60)

    # Stage 1: Patient stats
    patient_stats = compute_patient_intensity_stats(
        dataset,
        max_patches_per_patient=max_patches_per_patient,
        seed=seed,
    )

    # Stage 2: Target stats (median across patients)
    target_stats = compute_target_stats(patient_stats)

    # Stage 3: Spearman bias test
    spearman_results = run_spearman_bias_test(patient_stats, alpha=alpha)

    # Summary
    logger.info("=" * 60)
    logger.info("INTENSITY PRE-SCAN COMPLETE")
    logger.info("  Patients scanned: %d", len(patient_stats))
    logger.info("  Target channels: %d", len(target_stats))
    n_biased = sum(1 for r in spearman_results if r.significant)
    logger.info("  Biased channels (Spearman p < %.3f): %d / %d",
                alpha, n_biased, len(spearman_results))
    for r in spearman_results:
        logger.info("  %s: rho=%.4f, p=%.4f, %s",
                     r.channel_name, r.rho, r.p_value,
                     "BIAS!" if r.significant else "OK")
    logger.info("=" * 60)

    return patient_stats, target_stats, spearman_results


# =====================================================================
#  7. Serialization helpers (save/load stats to/from JSON)
# =====================================================================

# Pipeline version — increment when the pipeline order/semantics
# change so that cached stats from old versions are invalidated.
_PIPELINE_VERSION = 3  # v3: intensity_pipeline as default + JSON serialization fix


def save_stats_to_json(
    patient_stats: Dict[str, PatientStats],
    target_stats: Dict[int, TargetStats],
    spearman_results: List[SpearmanResult],
    path: str,
) -> None:
    """Save all pre-scan results to a JSON file.

    Handles numpy bool/int/float types that are not JSON-serializable
    by converting them to native Python types.
    """
    def _make_serializable(obj: Any) -> Any:
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    data = {
        "pipeline_version": _PIPELINE_VERSION,
        "patient_stats": {
            pid: _make_serializable(ps.to_dict()) for pid, ps in patient_stats.items()
        },
        "target_stats": {
            str(ch): _make_serializable(ts.to_dict()) for ch, ts in target_stats.items()
        },
        "spearman_results": [
            _make_serializable(r.to_dict()) for r in spearman_results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Stats saved to %s (pipeline_version=%d)", path, _PIPELINE_VERSION)


def load_stats_from_json(
    path: str,
) -> Tuple[
    Dict[str, PatientStats],
    Dict[int, TargetStats],
    List[SpearmanResult],
]:
    """Load pre-scan results from a JSON file.

    If the cached file was produced by a different pipeline version,
    a warning is logged and None is returned so the caller will
    recompute the stats.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check pipeline version
    cached_version = data.get("pipeline_version", 1)  # v1 = old buggy order
    if cached_version != _PIPELINE_VERSION:
        logger.warning(
            "Cached stats file %s has pipeline_version=%d but current "
            "version is %d.  The normalization pipeline order has "
            "changed — recomputing stats.  Old file: %s",
            path, cached_version, _PIPELINE_VERSION,
            "v1=clip-then-align (BUGGY)" if cached_version == 1 else "unknown",
        )
        return None, None, None

    patient_stats = {
        pid: PatientStats.from_dict(ps)
        for pid, ps in data["patient_stats"].items()
    }
    target_stats = {
        int(ch): TargetStats.from_dict(ts)
        for ch, ts in data["target_stats"].items()
    }
    spearman_results = [
        SpearmanResult.from_dict(r) for r in data["spearman_results"]
    ]

    logger.info("Stats loaded from %s (%d patients, %d channels, v%d)",
                path, len(patient_stats), len(target_stats), _PIPELINE_VERSION)
    return patient_stats, target_stats, spearman_results


# =====================================================================
#  8. Visualization helpers
# =====================================================================

def plot_patient_intensity_boxplot(
    patient_stats: Dict[str, PatientStats],
    channel_idx: int = 0,
    channel_name: str = "pSyn",
) -> bytes:
    """
    Box plot of mean raw intensity per patient, grouped by diagnosis.

    This visualizes whether there is a systematic intensity difference
    between PD and HC patients — the confound that the Spearman test
    detects statistically.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hc_means = []
    pd_means = []
    hc_pids = []
    pd_pids = []

    for pid, ps in patient_stats.items():
        if channel_idx in ps.channels:
            mean_val = ps.channels[channel_idx].mean
            if ps.label == 0:
                hc_means.append(mean_val)
                hc_pids.append(pid)
            else:
                pd_means.append(mean_val)
                pd_pids.append(pid)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    positions = [1, 2]
    bp = ax.boxplot(
        [hc_means, pd_means],
        positions=positions,
        widths=0.5,
        patch_artist=True,
        labels=["HC", "PD"],
    )

    # Color boxes
    colors = ["#4393C3", "#D6604D"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, (means, pids) in enumerate([(hc_means, hc_pids), (pd_means, pd_pids)]):
        x = np.full(len(means), positions[i]) + np.random.normal(0, 0.04, len(means))
        ax.scatter(x, means, alpha=0.7, s=40, zorder=5, edgecolors="black",
                   linewidth=0.5, color=colors[i])
        for xi, yi, pi in zip(x, means, pids):
            ax.annotate(pi, (xi, yi), fontsize=6, alpha=0.6,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_ylabel(f"Mean Raw Intensity ({channel_name})", fontsize=12)
    ax.set_title(f"Per-Patient Intensity Distribution — {channel_name}",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_spearman_scatter(
    patient_stats: Dict[str, PatientStats],
    channel_idx: int = 0,
    channel_name: str = "pSyn",
    alpha: float = 0.05,
) -> bytes:
    """
    Scatter plot of mean intensity vs diagnosis label with Spearman
    correlation annotation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = []
    labels = []
    pids = []

    for pid, ps in patient_stats.items():
        if channel_idx in ps.channels:
            means.append(ps.channels[channel_idx].mean)
            labels.append(ps.label)
            pids.append(pid)

    means_arr = np.array(means)
    labels_arr = np.array(labels)

    # Compute Spearman
    if len(np.unique(labels_arr)) >= 2:
        rho, p_val = scipy_stats.spearmanr(means_arr, labels_arr)
    else:
        rho, p_val = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    hc_mask = labels_arr == 0
    pd_mask = labels_arr == 1

    ax.scatter(np.where(hc_mask)[0], means_arr[hc_mask],
               color="#4393C3", s=80, label="HC", edgecolors="black",
               linewidth=0.5, zorder=5)
    ax.scatter(np.where(pd_mask)[0], means_arr[pd_mask],
               color="#D6604D", s=80, label="PD", edgecolors="black",
               linewidth=0.5, zorder=5)

    # Annotate
    sig_str = "*" if p_val < alpha else "n.s."
    ax.set_xlabel("Patient index (sorted)", fontsize=12)
    ax.set_ylabel(f"Mean Raw Intensity ({channel_name})", fontsize=12)
    ax.set_title(
        f"Spearman Correlation: {channel_name} vs Diagnosis\n"
        f"rho = {rho:.3f}, p = {p_val:.4f} {sig_str}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
