"""
OME-Zarr 2.75D Patch Dataset for Microglia Analysis
====================================================

PyTorch Dataset for extracting 2.75D patches from OME-Zarr stores
hosted in MinIO (S3-compatible) object storage.

Architecture variants
---------------------
    Model A (IBA1 only):  output shape  (1, 25, 256, 256)
    Model B (pSyn only):  output shape  (1, 25, 256, 256)
    Model C (both):       output shape  (2, 25, 256, 256)

Key features
------------
* Z-padding with zeros when the native Z depth < target_z slices.
* Empty-patch detection via **binary masks** from ``root['labels']``
  (never from raw fluorescence -- thermal noise is always > 0).
* Patch metadata persisted to JSON -- scan once, reuse forever.
* **SSD pre-download + in-memory caching** for fast training:
  Call ``preload_data()`` before training to download all zarr stores
  to local SSD and load raw arrays into RAM.  After pre-loading,
  ``__getitem__`` reads from in-memory numpy arrays with zero I/O,
  making training ~100x faster than reading from S3 on every call.
* **Per-channel normalization** (percentile / zscore / none):
  Raw fluorescence values are normalized per-channel in ``__getitem__``
  so that neural networks receive well-scaled inputs.
* **Basic augmentation** (random flips + 90-degree rotations):
  Controlled by the ``augment`` flag; applied after normalization,
  before tensor conversion.

OME-Zarr internal layout (per FOV)
-----------------------------------
    root['0'][0]                               -> pSyn  raw fluorescence (C=0)
    root['0'][1]                               -> IBA1  raw fluorescence (C=1)
    root['labels']['protein_mask']['0']         -> pSyn  binary mask
    root['labels']['cell_mask']['0']            -> IBA1  binary mask

MinIO folder layout
-------------------
    {base_folder}/
        HC/
            putamen/         HC1.zarr/  HC2.zarr/  ...
            substantiaNigra/ HC1.zarr/  HC2.zarr/  ...
        PD/
            putamen/         PD1.zarr/  PD2.zarr/  ...
            substantiaNigra/ PD1.zarr/  PD2.zarr/  ...

"""

from __future__ import annotations
import os
import logging

# 1. Set MinIO env vars (needed for s3fs / zarr connections).
os.environ["AWS_ACCESS_KEY_ID"] = "YOUR_MINIO_ACCESS_KEY"
os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR_MINIO_SECRET_KEY"
os.environ["AWS_ENDPOINT_URL"] = "YOUR_MINIO_ENDPOINT"
os.environ["CLEARML_AGENT_BOTO3_ENDPOINT_URL"] = "YOUR_MINIO_ENDPOINT"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# 2. ClearML initialization is ONLY done when running this script
#    standalone (for patch indexing).  When imported as a module by
#    train_lopo.py, the training script handles all ClearML setup.

import json
import logging
import os
import random
import shutil
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# -- Retry decorator for S3 operations --
def _retry_s3(fn, *args, max_retries: int = 5, base_delay: float = 2.0, **kwargs):
    """
    Retry an S3 operation with exponential backoff.

    Handles transient 502 Bad Gateway and other network errors
    that commonly occur when downloading large zarr stores from MinIO.

    Parameters
    ----------
    fn : callable
        The function to call (e.g. ``fs.get``).
    max_retries : int
        Maximum number of retry attempts (default 5).
    base_delay : float
        Base delay in seconds for exponential backoff (default 2.0).

    Returns
    -------
    The return value of ``fn(*args, **kwargs)``.

    Raises
    ------
    The last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            is_transient = (
                "502" in err_str
                or "503" in err_str
                or "500" in err_str
                or "bad gateway" in err_str
                or "service unavailable" in err_str
                or "internal server error" in err_str
                or "connectionreset" in err_str.replace(" ", "")
                or "timeout" in err_str
                or "broken pipe" in err_str
            )
            if not is_transient or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt) + (0.5 * (attempt + 1))
            logger.warning(
                "S3 transient error (attempt %d/%d): %s — "
                "retrying in %.1fs ...",
                attempt + 1, max_retries, e, delay,
            )
            time.sleep(delay)
    raise last_exc

import numpy as np
import s3fs
import torch
import zarr
from torch.utils.data import Dataset

# -- zarr 2.x / 3.x compatibility shim --
_ZARR_MAJOR = int(getattr(zarr, "__version__", "2").split(".")[0])

if _ZARR_MAJOR >= 3:
    try:
        from zarr.storage import FsspecStore as _RemoteStore  # zarr >= 3
    except ImportError:
        _RemoteStore = None  # type: ignore[assignment]
    _ZARR_V3 = True
    import warnings as _w
    _w.filterwarnings("ignore", message=".*asynchronous.*", category=UserWarning)
else:
    try:
        from zarr.storage import FSStore as _RemoteStore  # zarr < 3
    except ImportError:
        _RemoteStore = None  # type: ignore[assignment]
    _ZARR_V3 = False

logger = logging.getLogger(__name__)


# Data classes


@dataclass
class PatchMeta:
    """Metadata for a single extracted patch."""

    zarr_path: str        # MinIO key to the OME-Zarr store
    group: str            # 'HC' or 'PD'
    region: str           # 'putamen' or 'substantiaNigra'
    patient_id: str       # e.g. 'HC1', 'PD3'
    fov_id: str           # FOV sub-folder / zarr group name (e.g. '0001')
    y_start: int          # top-left Y pixel inside the FOV
    x_start: int          # top-left X pixel inside the FOV
    z_native: int         # original number of Z slices before padding
    z_padded: bool        # True if zero-padding was applied (Z < target_z)
    is_empty: bool        # True if **both** masks are all-zero at this location
    label: int            # 0 = HC, 1 = PD
    is_store_fov: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PatchMeta":
        return cls(**d)



# MinIO helpers



def _make_s3fs(
    endpoint: str,
    access_key: str,
    secret_key: str,
    secure: bool = True,
) -> s3fs.S3FileSystem:
    """Build an ``s3fs.S3FileSystem`` configured for a MinIO endpoint."""
    scheme = "https" if secure else "http"
    return s3fs.S3FileSystem(
        key=access_key,
        secret=secret_key,
        client_kwargs={
            "endpoint_url": f"{scheme}://{endpoint}",
            "region_name": "us-east-1",
        },
        config_kwargs={
            "read_timeout": 900,
            "connect_timeout": 300,
            "retries": {
                "max_attempts": 10,
                "mode": "adaptive"
            }
        },
        asynchronous=False,
    )


def _open_zarr_root(
    fs: s3fs.S3FileSystem,
    bucket: str,
    zarr_key: str,
) -> zarr.Group:
    """Open a remote OME-Zarr store and return the root group."""
    s3_path = f"{bucket}/{zarr_key}"

    if _ZARR_V3 and _RemoteStore is not None:
        store = _RemoteStore(fs=fs, path=s3_path)
    elif _RemoteStore is not None:
        store = _RemoteStore(s3_path, fs=fs)
    else:
        raise ImportError(
            "Neither FSStore (zarr<3) nor FsspecStore (zarr>=3) could be "
            "imported from zarr.storage.  Check your zarr installation."
        )

    return zarr.open(store, mode="r")



# Patch coordinate math



def _patch_starts(fov_size: int, patch_size: int, stride: int) -> List[int]:
    """Compute all valid start positions along one spatial axis."""
    starts: List[int] = []
    pos = 0
    while pos < fov_size:
        if pos + patch_size <= fov_size:
            starts.append(pos)
        else:
            starts.append(max(fov_size - patch_size, 0))
            break
        pos += stride
    seen = set()
    unique: List[int] = []
    for s in starts:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique



# Main Dataset


ModelType = Literal["A", "B", "C"]
NormMode = Literal["percentile", "zscore", "none"]


class ZarrPatchDataset(Dataset):
    """
    PyTorch Dataset that yields 2.75D patches from OME-Zarr stores in MinIO.

    Supports **SSD pre-download + in-memory caching** for fast training.
    Call ``preload_data()`` before the training loop to eliminate all
    network I/O during training.

    Parameters
    ----------
    minio_endpoint : str
        MinIO server hostname.
    minio_access_key : str
        Access key for MinIO authentication.
    minio_secret_key : str
        Secret key for MinIO authentication.
    bucket_name : str
        Name of the S3 bucket containing the data.
    base_folder : str
        Prefix inside the bucket where patient folders live.
    model_type : ``"A"`` | ``"B"`` | ``"C"``
        * **A** -- IBA1 only   -> output channels = 1
        * **B** -- pSyn only   -> output channels = 1
        * **C** -- both        -> output channels = 2
    patch_size : int
        Spatial patch size (default 256).
    target_z : int
        Desired Z depth (default 25).  Shorter stacks are zero-padded.
    stride : int | None
        Stride between patch centres.  Defaults to ``patch_size``.
    regions : list[str] | None
        Restrict to these brain regions.
    groups : list[str] | None
        Restrict to these cohorts.
    filter_empty : bool
        If True, patches where **both** masks are all-zero are excluded.
    index_path : str | Path | None
        Path to a JSON file that caches the patch index.
    secure : bool
        Use HTTPS for MinIO connection (default True).
    fov_size : int
        Expected Y = X size of each FOV in pixels (default 1200).
    max_fovs_per_patient : int | None
        Safety cap on the number of FOVs accepted per patient.
    local_cache_dir : str | None
        Local directory to cache downloaded zarr stores (default "/tmp/zarr_cache").
        If None, no local caching is performed (reads from S3 each time).
    preload_to_ram : bool
        If True, ``preload_data()`` will load raw arrays into RAM for
        zero-I/O patch access during training (default True).
    normalization : ``"percentile"`` | ``"zscore"`` | ``"none"``
        Per-channel normalization applied in ``__getitem__``:
        * **percentile** (default) -- clip at 1st/99th percentile per
          channel, then scale to [0, 1].
        * **zscore** -- subtract mean, divide by std per channel.
        * **none** -- return raw values unchanged.
    augment : bool
        If True, apply random augmentation (horizontal flip, vertical
        flip, 90-degree rotation) to each patch after normalization.
        Default False.  Set True for training datasets.
    """

    # Channel ordering in OME-Zarr
    _CH_PSYN = 0
    _CH_IBA1 = 1

    # Model-type -> channels to load from raw data
    _MODEL_CHANNELS: Dict[ModelType, List[int]] = {
        "A": [_CH_IBA1],
        "B": [_CH_PSYN],
        "C": [_CH_IBA1, _CH_PSYN],
    }

    # Number of output channels per model type
    _MODEL_N_CHANNELS: Dict[ModelType, int] = {
        "A": 1,
        "B": 1,
        "C": 2,
    }

    # Valid normalization modes
    _VALID_NORMALIZATION = ("percentile", "zscore", "none", "intensity_pipeline")

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    def __init__(
        self,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        bucket_name: str = "YOUR_STORAGE_NAME",
        base_folder: str = "YOUR_BASE_PREFIX",
        model_type: ModelType = "C",
        patch_size: int = 256,
        target_z: int = 25,
        stride: Optional[int] = None,
        regions: Optional[List[str]] = None,
        groups: Optional[List[str]] = None,
        filter_empty: bool = True,
        index_path: Optional[str | Path] = None,
        secure: bool = True,
        fov_size: int = 1200,
        max_fovs_per_patient: Optional[int] = None,
        local_cache_dir: Optional[str] = "/tmp/zarr_cache",
        preload_to_ram: bool = True,
        normalization: str = "percentile",
        augment: bool = False,
    ) -> None:
        super().__init__()

        # Store configuration ------------------------------------------------
        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.bucket_name = bucket_name
        self.base_folder = base_folder.rstrip("/")
        self.model_type: ModelType = model_type
        self.patch_size = patch_size
        self.target_z = target_z
        self.stride = stride if stride is not None else patch_size
        self.regions = regions or ["putamen", "substantiaNigra"]
        self.groups = groups or ["HC", "PD"]
        self.filter_empty = filter_empty
        self.secure = secure
        self.fov_size = fov_size
        self.max_fovs_per_patient = max_fovs_per_patient

        self.n_output_channels = self._MODEL_N_CHANNELS[self.model_type]
        self._channels_to_load = self._MODEL_CHANNELS[self.model_type]

        # -- Normalization & augmentation ------------------------------------
        if normalization not in self._VALID_NORMALIZATION:
            raise ValueError(
                f"normalization must be one of {self._VALID_NORMALIZATION}, "
                f"got '{normalization}'"
            )
        self.normalization: str = normalization
        self.augment: bool = augment

        # -- Intensity pipeline normalizer (set externally) ------------
        # When normalization="intensity_pipeline", this attribute must
        # be set to an IntensityNormalizer instance before training.
        # See intensity_normalization.py for details.
        self._normalizer = None

        # -- SSD caching + RAM pre-loading ----------------------------------
        self._local_cache_dir = Path(local_cache_dir) if local_cache_dir else None
        self._preload_to_ram = preload_to_ram
        self._raw_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._is_preloaded = False

        # Pre-compute Y/X start positions for a single FOV
        self._y_starts = _patch_starts(self.fov_size, self.patch_size, self.stride)
        self._x_starts = _patch_starts(self.fov_size, self.patch_size, self.stride)

        # Build or load patch index ------------------------------------------
        self.patches: List[PatchMeta] = []
        self._zarr_cache: OrderedDict[str, zarr.Group] = OrderedDict()
        self._cache_max = 64  # keep at most this many zarr roots open

        if index_path is not None:
            index_path = Path(index_path)
            if index_path.exists():
                logger.info("Loading cached patch index from %s", index_path)
                self._load_index(index_path)
            else:
                logger.info("Index file not found -- scanning MinIO ...")
                self._build_index()
                self._save_index(index_path)
        else:
            logger.info("No index_path provided -- scanning MinIO ...")
            self._build_index()

        logger.info(
            "ZarrPatchDataset ready: %d patches  (model=%s, filter_empty=%s, "
            "normalization=%s, augment=%s)",
            len(self.patches),
            self.model_type,
            self.filter_empty,
            self.normalization,
            self.augment,
        )

    # -----------------------------------------------------------------
    # Normalization
    # -----------------------------------------------------------------

    def _normalize_patch(self, patch: np.ndarray, patient_id: str = "") -> np.ndarray:
        """
        Normalize a patch per-channel.

        Parameters
        ----------
        patch : np.ndarray
            Raw patch of shape ``(C_out, Z, H, W)`` with float32 values.
        patient_id : str
            Patient identifier (required for ``"intensity_pipeline"``
            mode to look up per-patient statistics).

        Returns
        -------
        np.ndarray
            Normalized patch of the same shape.  The normalization mode
            is determined by ``self.normalization``:

            * ``"percentile"`` -- clip at 1st/99th percentile per channel,
              then linearly scale to [0, 1].
            * ``"zscore"`` -- subtract mean and divide by std per channel.
            * ``"none"`` -- return unchanged.
            * ``"intensity_pipeline"`` -- full pipeline: percentile clipping
              + Min-Max + cross-patient alignment via IntensityNormalizer.
              Requires ``self._normalizer`` to be set externally.

        Notes
        -----
        - Normalization is **per-channel**: for each channel ``c``, all
          spatial + Z values (i.e. ``patch[c].ravel()``) are used to
          compute the statistic.
        - For ``"percentile"`` mode, if the 1st and 99th percentiles
          are equal (constant channel), the channel is set to 0 to
          avoid division by zero.
        - For ``"zscore"`` mode, if the std is 0 (constant channel),
          the channel is set to 0.
        - For ``"intensity_pipeline"`` mode, the IntensityNormalizer
          applies patient-level percentiles for clipping and then
          aligns to cohort-wide target statistics.
        """
        if self.normalization == "none":
            return patch

        # Intensity pipeline: delegate to IntensityNormalizer
        if self.normalization == "intensity_pipeline":
            if self._normalizer is not None:
                return self._normalizer.normalize_patch(patch, patient_id)
            else:
                logger.warning(
                    "normalization='intensity_pipeline' but _normalizer "
                    "is not set. Falling back to per-patch percentile."
                )
                # Fallback to basic percentile normalization
                out = patch.copy()
                for c in range(out.shape[0]):
                    ch_data = out[c]
                    p1 = np.percentile(ch_data, 1)
                    p99 = np.percentile(ch_data, 99)
                    if p99 - p1 < 1e-8:
                        out[c] = 0.0
                    else:
                        out[c] = np.clip(ch_data, p1, p99)
                        out[c] = (out[c] - p1) / (p99 - p1)
                return out

        out = patch.copy()
        n_channels = out.shape[0]

        for c in range(n_channels):
            ch_data = out[c]  # view into (Z, H, W)

            if self.normalization == "percentile":
                p1 = np.percentile(ch_data, 1)
                p99 = np.percentile(ch_data, 99)
                if p99 - p1 < 1e-8:
                    # Constant or near-constant channel -> map to 0
                    out[c] = 0.0
                else:
                    out[c] = np.clip(ch_data, p1, p99)
                    out[c] = (out[c] - p1) / (p99 - p1)

            elif self.normalization == "zscore":
                mean = np.mean(ch_data)
                std = np.std(ch_data)
                if std < 1e-8:
                    out[c] = 0.0
                else:
                    out[c] = (ch_data - mean) / std

        return out

    # -----------------------------------------------------------------
    # Augmentation
    # -----------------------------------------------------------------

    def _augment_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Apply random spatial + intensity augmentation to a patch.

        Spatial augmentations (applied identically across all channels
        and Z-slices):
        - Random horizontal flip (50% chance)
        - Random vertical flip (50% chance)
        - Random 90-degree rotation (0°, 90°, 180°, or 270°)
        - Random elastic deformation (20% chance, mild)

        Intensity augmentations (applied per-channel for diversity,
        important for fluorescence microscopy):
        - Random Gaussian noise (sigma=0.01-0.03, per-channel)
        - Random brightness shift (±5% of [0,1] range, per-channel)
        - Random contrast change (0.9-1.1x, per-channel)
        - Random gamma correction (0.8-1.2, per-channel, 30% chance)
        - Random channel dropout (entire channel → 0, 10% chance per ch)
        - Random erasing (5-15% of spatial area, 20% chance)

        Parameters
        ----------
        patch : np.ndarray
            Patch of shape ``(C_out, Z, H, W)`` with values in [0, 1]
            (after normalization).

        Returns
        -------
        np.ndarray
            Augmented patch of the same shape, clipped to [0, 1].

        Notes
        -----
        - Spatial transforms are applied identically across all
          channels and Z-slices (i.e. the same flip/rotate is applied
          to every ``(c, z)`` pair).
        - Intensity transforms are applied per-channel to increase
          diversity and make the model robust to staining and
          illumination variability.
        - All transforms are differentiable-friendly (no aliasing).
        """
        # ── Spatial augmentations ────────────────────────────────────
        # Random horizontal flip
        if random.random() < 0.5:
            patch = np.flip(patch, axis=3).copy()

        # Random vertical flip
        if random.random() < 0.5:
            patch = np.flip(patch, axis=2).copy()

        # Random 90-degree rotation (0, 90, 180, 270)
        k = random.randint(0, 3)  # number of 90-degree rotations
        if k > 0:
            patch = np.rot90(patch, k=k, axes=(2, 3)).copy()

        # Random elastic deformation (mild, 20% chance)
        # Applies same deformation to all channels/Z-slices
        if random.random() < 0.2:
            patch = self._elastic_deform(patch, alpha=8, sigma=3)

        # ── Intensity augmentations (per-channel) ────────────────────
        # These are critical for fluorescence microscopy to make the
        # model robust to staining variability, photobleaching, and
        # illumination differences.
        out = patch.copy()
        n_channels = out.shape[0]

        for c in range(n_channels):
            ch_data = out[c]  # (Z, H, W)

            # Random Gaussian noise (per-channel)
            # Sigma drawn from [0.01, 0.03] — subtle but effective
            if random.random() < 0.5:
                sigma = random.uniform(0.01, 0.03)
                noise = np.random.normal(0, sigma, ch_data.shape).astype(np.float32)
                ch_data = ch_data + noise

            # Random brightness shift (±5% of range)
            if random.random() < 0.3:
                shift = random.uniform(-0.05, 0.05)
                ch_data = ch_data + shift

            # Random contrast change (0.9x-1.1x)
            if random.random() < 0.3:
                factor = random.uniform(0.9, 1.1)
                mean_val = np.mean(ch_data)
                ch_data = (ch_data - mean_val) * factor + mean_val

            # Random gamma correction (0.8-1.2, 30% chance)
            # Models non-linear intensity variations in fluorescence
            if random.random() < 0.3:
                gamma = random.uniform(0.8, 1.2)
                # Apply gamma: clip to [0,1] first for safety
                ch_data = np.clip(ch_data, 0.0, 1.0)
                ch_data = np.power(ch_data, gamma).astype(np.float32)

            # Random channel dropout (10% chance per channel)
            # Forces model to not rely on a single channel
            if random.random() < 0.1:
                ch_data = np.zeros_like(ch_data)

            out[c] = ch_data

        # Random erasing (5-15% of spatial area, 20% chance)
        # Applies same erasing mask across all channels/Z-slices
        if random.random() < 0.2:
            out = self._random_erase(out, area_ratio_range=(0.05, 0.15))

        # Clip back to [0, 1]
        out = np.clip(out, 0.0, 1.0)
        return out.astype(np.float32)

    def _elastic_deform(
        self,
        patch: np.ndarray,
        alpha: float = 8.0,
        sigma: float = 3.0,
    ) -> np.ndarray:
        """
        Apply random elastic deformation to a patch.

        Applies the SAME deformation to all channels and Z-slices
        (since they share spatial coordinates).

        Parameters
        ----------
        patch : np.ndarray
            Shape ``(C, Z, H, W)``.
        alpha : float
            Deformation amplitude (pixels).  Lower = milder.
        sigma : float
            Gaussian smoothing for displacement field.

        Returns
        -------
        np.ndarray
            Deformed patch of same shape.
        """
        from scipy.ndimage import gaussian_filter, map_coordinates

        C, Z, H, W = patch.shape

        # Generate displacement fields for H and W dimensions
        dx = gaussian_filter(
            (np.random.rand(H, W) * 2 - 1), sigma=sigma
        ) * alpha
        dy = gaussian_filter(
            (np.random.rand(H, W) * 2 - 1), sigma=sigma
        ) * alpha

        # Create coordinate grids
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        indices_x = np.clip(x + dx, 0, W - 1).astype(np.float32)
        indices_y = np.clip(y + dy, 0, H - 1).astype(np.float32)

        # Apply the same deformation to all (C, Z) pairs
        out = np.empty_like(patch)
        for c in range(C):
            for z in range(Z):
                out[c, z] = map_coordinates(
                    patch[c, z],
                    [indices_y.ravel(), indices_x.ravel()],
                    order=1, mode='reflect',
                ).reshape(H, W)

        return out

    def _random_erase(
        self,
        patch: np.ndarray,
        area_ratio_range: Tuple[float, float] = (0.05, 0.15),
    ) -> np.ndarray:
        """
        Apply random erasing to a patch.

        Selects a random rectangular region and fills it with random
        noise.  The same region is erased across all channels/Z-slices.

        Parameters
        ----------
        patch : np.ndarray
            Shape ``(C, Z, H, W)``.
        area_ratio_range : tuple
            (min, max) fraction of total spatial area to erase.

        Returns
        -------
        np.ndarray
            Patch with erased region.
        """
        C, Z, H, W = patch.shape
        total_area = H * W
        erase_area = total_area * random.uniform(*area_ratio_range)

        # Random aspect ratio
        aspect = random.uniform(0.3, 3.0)
        erase_h = int(np.sqrt(erase_area * aspect))
        erase_w = int(np.sqrt(erase_area / aspect))
        erase_h = min(erase_h, H)
        erase_w = min(erase_w, W)

        # Random position
        y_start = random.randint(0, H - erase_h) if erase_h < H else 0
        x_start = random.randint(0, W - erase_w) if erase_w < W else 0

        # Fill with random noise (same across all C, Z for consistency)
        noise_val = random.uniform(0.0, 1.0)
        patch[:, :, y_start:y_start + erase_h,
                     x_start:x_start + erase_w] = noise_val

        return patch

    # -----------------------------------------------------------------
    # SSD pre-download + RAM pre-loading
    # -----------------------------------------------------------------

    def preload_data(
        self,
        max_workers: int = 8,
        max_ram_gb: float = 64.0,
    ) -> None:
        """
        Download all zarr data to local SSD and load into RAM.

        This eliminates all network I/O during training, making it ~100x
        faster than reading from S3 on every ``__getitem__`` call.

        Two phases:
          1. **Download**: Multi-threaded download of all zarr stores
             from S3 to ``local_cache_dir`` (default: ``/tmp/zarr_cache``).
             Skips stores that are already on disk.
          2. **RAM load**: Open each local zarr store and read the
             raw data array (only needed channels) into a numpy array
             stored in ``self._raw_cache``.  After this, ``__getitem__``
             reads patches by slicing in-memory arrays -- zero disk and
             network I/O.

        **RAM safety**: Before Phase 2, the estimated total RAM usage
        is computed.  If it exceeds ``max_ram_gb``, Phase 2 is skipped
        and the dataset falls back to SSD-only reads (still fast since
        all data is on local SSD after Phase 1).

        Parameters
        ----------
        max_workers : int
            Number of parallel download / load threads.
        max_ram_gb : float
            Maximum RAM (in GB) allowed for Phase 2.  If the estimated
            total exceeds this, Phase 2 is skipped.  Default 64 GB.

        Expected timing on an A100 server with 10 Gbps network:
          - Download: ~30-60 seconds for ~25 GB of zarr data
          - RAM load: ~10-20 seconds for ~20 GB of raw arrays
          - Total: ~1-2 minutes

        After pre-loading, training epoch time drops from hours to minutes.
        """
        if self._is_preloaded:
            logger.info("Data already pre-loaded, skipping.")
            print("[PRELOAD] Data already pre-loaded, skipping.", flush=True)
            return

        # Collect unique zarr_keys from patch index
        zarr_keys = sorted(set(p.zarr_path for p in self.patches))
        logger.info(
            "Pre-loading %d zarr stores (max_workers=%d) ...",
            len(zarr_keys), max_workers,
        )
        print(
            f"[PRELOAD] Pre-loading {len(zarr_keys)} zarr stores "
            f"(max_workers={max_workers}) ...",
            flush=True,
        )

        fs = self._get_fs()
        t0 = time.time()

        # ==================================================================
        # Phase 1: Download zarr stores from S3 to local SSD
        # ==================================================================
        if self._local_cache_dir is not None:
            self._local_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Phase 1: Downloading zarr stores to %s ...",
                self._local_cache_dir,
            )
            print(
                f"[PRELOAD PHASE 1] Downloading zarr stores to "
                f"{self._local_cache_dir} ...",
                flush=True,
            )

            def _download_store(zarr_key: str) -> str:
                s3_path = f"{self.bucket_name}/{zarr_key}"
                local_path = self._local_cache_dir / zarr_key

                # Validate existing cache: a valid zarr store must have
                # at least .zgroup or .zarray.  If the directory exists
                # but is incomplete (no zarr metadata), remove it and
                # re-download.
                if local_path.exists():
                    has_zarr_meta = (
                        (local_path / ".zgroup").exists()
                        or (local_path / ".zarray").exists()
                        or (local_path / "0" / ".zarray").exists()
                    )
                    if has_zarr_meta:
                        return zarr_key  # already cached and valid
                    else:
                        logger.warning(
                            "Corrupt/incomplete cache for %s — "
                            "removing and re-downloading.",
                            zarr_key,
                        )
                        shutil.rmtree(local_path, ignore_errors=True)

                local_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    _retry_s3(
                        fs.get, s3_path, str(local_path),
                        recursive=True,
                        max_retries=5, base_delay=2.0,
                    )
                except Exception:
                    # Clean up partially downloaded directory on failure
                    if local_path.exists():
                        shutil.rmtree(local_path, ignore_errors=True)
                    raise
                return zarr_key

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_download_store, zk): zk for zk in zarr_keys}
                done_count = 0
                for f in as_completed(futures):
                    done_count += 1
                    try:
                        f.result()
                    except Exception as e:
                        logger.error("Download failed for %s: %s", futures[f], e)
                    if done_count % 5 == 0 or done_count == len(zarr_keys):
                        logger.info(
                            "  Downloaded %d/%d stores ...",
                            done_count, len(zarr_keys),
                        )
                        print(
                            f"[PRELOAD PHASE 1]   {done_count}/{len(zarr_keys)} "
                            f"stores downloaded ...",
                            flush=True,
                        )

            dl_time = time.time() - t0
            logger.info(
                "Phase 1 complete: %d stores downloaded in %.1f seconds.",
                len(zarr_keys), dl_time,
            )
            print(
                f"[PRELOAD PHASE 1] Complete! {len(zarr_keys)} stores "
                f"in {dl_time:.0f}s",
                flush=True,
            )

        # ==================================================================
        # Phase 2: Load raw data arrays into RAM (only needed channels)
        # ==================================================================
        # NOTE: We only load the channels required by the current model
        # type (self._channels_to_load), NOT the full (C, Z, Y, X) array.
        # This halves RAM usage for Model A/B and prevents OOM kills.
        # IMPORTANT: The cache stores RAW (unnormalized) data.
        # Normalization is applied per-patch in __getitem__.
        # ==================================================================
        if self._preload_to_ram:
            t1 = time.time()

            # Collect unique (zarr_key, fov_id, is_store_fov) tuples
            fov_keys = sorted(set(
                (p.zarr_path, p.fov_id, p.is_store_fov) for p in self.patches
            ))
            logger.info("Phase 2: Loading %d FOVs into RAM ...", len(fov_keys))
            print(
                f"[PRELOAD PHASE 2] Loading {len(fov_keys)} FOVs into RAM ...",
                flush=True,
            )

            # ---- RAM budget check before loading ----
            # Estimate per-FOV RAM: C_needed × target_z × fov_size² × 4 bytes
            _est_bytes_per_fov = (
                len(self._channels_to_load)
                * self.target_z
                * self.fov_size
                * self.fov_size
                * 4  # float32
            )
            _est_total_gb = len(fov_keys) * _est_bytes_per_fov / 1e9
            logger.info(
                "Phase 2 RAM estimate: %.1f GB  (%d FOVs × %d ch × %dZ × %d² "
                "× 4B)  [max_ram_gb=%.1f]",
                _est_total_gb, len(fov_keys), len(self._channels_to_load),
                self.target_z, self.fov_size, max_ram_gb,
            )
            if _est_total_gb > max_ram_gb:
                logger.warning(
                    "Phase 2 SKIPPED: estimated %.1f GB exceeds max_ram_gb=%.1f. "
                    "Falling back to SSD-only reads (data is already on "
                    "local SSD from Phase 1).  Training will still be fast!",
                    _est_total_gb, max_ram_gb,
                )
                print(
                    f"[PRELOAD PHASE 2] SKIPPED: estimated {_est_total_gb:.1f} GB "
                    f"exceeds max_ram_gb={max_ram_gb:.1f}. "
                    f"Using SSD-only reads.",
                    flush=True,
                )
                self._is_preloaded = True
                total_time = time.time() - t0
                logger.info(
                    "Pre-loading finished in %.1f seconds (SSD-only mode).",
                    total_time,
                )
                print(
                    f"[PRELOAD] Finished in {total_time:.0f}s. "
                    f"SSD-only mode — training reads from local disk.",
                    flush=True,
                )
                return

            def _load_fov(
                zarr_key: str, fov_id: str, is_store_fov: bool
            ) -> Tuple[Tuple[str, str], Optional[np.ndarray]]:
                # Open zarr store -- prefer local copy, fall back to S3
                if (self._local_cache_dir is not None
                        and (self._local_cache_dir / zarr_key).exists()):
                    root = zarr.open(
                        str(self._local_cache_dir / zarr_key), mode="r"
                    )
                else:
                    root = self._get_zarr_root(zarr_key)

                # Navigate to the FOV group
                if is_store_fov or fov_id == "root":
                    fov_root = root
                else:
                    try:
                        fov_root = root[fov_id]
                    except KeyError:
                        logger.warning(
                            "FOV '%s' not found in %s -- skipping.",
                            fov_id, zarr_key,
                        )
                        return (zarr_key, fov_id), None

                # Read ONLY the channels needed for this model type.
                # The full OME-Zarr raw array is (C_total, Z, Y, X) where
                # C_total=2 (pSyn + IBA1).  We slice out only the channels
                # in self._channels_to_load to halve RAM for Model A/B.
                try:
                    raw_arr = self._find_data_array(fov_root)
                    # Read only needed channels — avoids loading
                    # unnecessary data into RAM
                    data = np.asarray(
                        raw_arr[self._channels_to_load]
                    ).astype(np.float32)  # (C_needed, Z, Y, X)
                    return (zarr_key, fov_id), data
                except (KeyError, IndexError) as e:
                    logger.warning(
                        "Failed to load %s/%s: %s", zarr_key, fov_id, e
                    )
                    return (zarr_key, fov_id), None

            n_loaded = 0
            with ThreadPoolExecutor(max_workers=min(max_workers, 4)) as pool:
                futures = {
                    pool.submit(_load_fov, zk, fid, isf): (zk, fid)
                    for zk, fid, isf in fov_keys
                }
                for f in as_completed(futures):
                    try:
                        key, data = f.result()
                        if data is not None:
                            self._raw_cache[key] = data
                            n_loaded += 1
                    except Exception as e:
                        logger.error("RAM load failed for %s: %s", futures[f], e)

            load_time = time.time() - t1
            total_gb = sum(a.nbytes for a in self._raw_cache.values()) / 1e9
            logger.info(
                "Phase 2 complete: Loaded %.1f GB into RAM in %.1f seconds "
                "(%d/%d FOVs).",
                total_gb, load_time, n_loaded, len(fov_keys),
            )
            print(
                f"[PRELOAD PHASE 2] Complete! {total_gb:.1f} GB loaded "
                f"into RAM in {load_time:.0f}s ({n_loaded}/{len(fov_keys)} FOVs)",
                flush=True,
            )

        self._is_preloaded = True
        total_time = time.time() - t0
        logger.info("Pre-loading finished in %.1f seconds.", total_time)
        print(
            f"[PRELOAD] Finished in {total_time:.0f}s. "
            f"Training will now be fast!",
            flush=True,
        )

    # -----------------------------------------------------------------
    # MinIO / s3fs connection
    # -----------------------------------------------------------------

    def _get_fs(self) -> s3fs.S3FileSystem:
        """Return a (new) s3fs filesystem."""
        return _make_s3fs(
            self.minio_endpoint,
            self.minio_access_key,
            self.minio_secret_key,
            secure=self.secure,
        )

    # -----------------------------------------------------------------
    # Zarr root cache
    # -----------------------------------------------------------------

    def _get_zarr_root(self, zarr_key: str) -> zarr.Group:
        """Open a zarr root, preferring the local SSD cache if available."""
        # Check local cache first
        if (self._local_cache_dir is not None
                and (self._local_cache_dir / zarr_key).exists()):
            local_path = str(self._local_cache_dir / zarr_key)
            return zarr.open(local_path, mode="r")

        # Fall back to remote S3 store
        if zarr_key in self._zarr_cache:
            self._zarr_cache.move_to_end(zarr_key)
            return self._zarr_cache[zarr_key]

        fs = self._get_fs()
        root = _open_zarr_root(fs, self.bucket_name, zarr_key)

        self._zarr_cache[zarr_key] = root
        if len(self._zarr_cache) > self._cache_max:
            self._zarr_cache.popitem(last=False)
        return root

    # -----------------------------------------------------------------
    # MinIO discovery
    # -----------------------------------------------------------------

    def _is_zarr_store(self, fs: s3fs.S3FileSystem, s3_path: str) -> bool:
        """Check whether *s3_path* points to a valid zarr store."""
        return (
            fs.exists(f"{s3_path}/.zgroup")
            or fs.exists(f"{s3_path}/.zarray")
            or fs.exists(f"{s3_path}/0/.zarray")
        )

    def _discover_zarr_stores(self) -> List[Dict[str, str]]:
        """Walk the MinIO bucket and return a list of dicts describing stores."""
        fs = self._get_fs()
        prefix = self.base_folder + "/"
        stores: List[Dict[str, str]] = []

        for grp in self.groups:
            for reg in self.regions:
                folder_prefix = f"{prefix}{grp}/{reg}/"
                bucket_prefix = f"{self.bucket_name}/{folder_prefix}"

                try:
                    patient_entries = fs.ls(bucket_prefix, detail=False)
                except FileNotFoundError:
                    logger.warning("Folder not found: %s", folder_prefix)
                    continue

                for pe in patient_entries:
                    rel = pe.removeprefix(self.bucket_name + "/").rstrip("/")
                    entry_name = rel.replace(folder_prefix.rstrip("/"), "").strip("/")
                    if not entry_name or entry_name.startswith("."):
                        continue

                    patient_id = entry_name
                    if patient_id.endswith(".zarr"):
                        patient_id = patient_id[: -len(".zarr")]

                    s3_path = f"{self.bucket_name}/{rel}"

                    if self._is_zarr_store(fs, s3_path):
                        stores.append({
                            "zarr_key": rel,
                            "group": grp,
                            "region": reg,
                            "patient_id": patient_id,
                        })
                        continue

                    try:
                        fov_entries = fs.ls(s3_path, detail=False)
                    except FileNotFoundError:
                        continue

                    for fe in fov_entries:
                        fov_rel = fe.removeprefix(self.bucket_name + "/").rstrip("/")
                        fov_name = fov_rel.replace(rel + "/", "").strip("/")
                        if not fov_name or fov_name.startswith("."):
                            continue

                        if self._is_zarr_store(fs, fe.rstrip("/")):
                            fov_id = fov_name
                            if fov_id.endswith(".zarr"):
                                fov_id = fov_id[: -len(".zarr")]
                            stores.append({
                                "zarr_key": fov_rel,
                                "group": grp,
                                "region": reg,
                                "patient_id": patient_id,
                                "fov_id_override": fov_id,
                            })

        logger.info(
            "Discovered %d zarr stores across %d groups x %d regions.",
            len(stores), len(self.groups), len(self.regions),
        )
        return stores

    def _discover_fovs_in_store(self, zarr_key: str) -> List[str]:
        """List FOV sub-groups inside a patient zarr store."""
        fs = self._get_fs()
        fovs: List[str] = []

        try:
            all_keys = fs.ls(f"{self.bucket_name}/{zarr_key}", detail=False)
        except FileNotFoundError:
            return fovs

        rel_keys = [k.removeprefix(self.bucket_name + "/") for k in all_keys]
        prefix = zarr_key.rstrip("/") + "/"

        subdirs: set[str] = set()
        for rk in rel_keys:
            remainder = rk.replace(prefix, "")
            if not remainder:
                continue
            top = remainder.split("/")[0]
            if top.startswith("."):
                continue
            subdirs.add(top)

        fov_candidates = [
            s for s in subdirs
            if s.isdigit() or (len(s) <= 6 and s.isalnum())
        ]
        if 1 <= len(fov_candidates) <= 100:
            fovs = sorted(fov_candidates)
        else:
            fovs = ["root"]

        return fovs

    # -----------------------------------------------------------------
    # Empty-patch detection
    # -----------------------------------------------------------------

    def _is_patch_empty(
        self,
        root: zarr.Group,
        y_start: int,
        x_start: int,
    ) -> bool:
        """Return True if both masks are all-zero in the given window."""
        ps = self.patch_size

        try:
            protein_mask_arr = root["labels"]["protein_mask"]["0"]
            cell_mask_arr = root["labels"]["cell_mask"]["0"]
        except KeyError:
            logger.warning("Labels branch missing -- treating patch as non-empty.")
            return False

        y_end = min(y_start + ps, protein_mask_arr.shape[1])
        x_end = min(x_start + ps, protein_mask_arr.shape[2])

        protein_patch = protein_mask_arr[:, y_start:y_end, x_start:x_end]
        cell_patch = cell_mask_arr[:, y_start:y_end, x_start:x_end]

        if protein_patch.shape[1] < ps or protein_patch.shape[2] < ps:
            pad_y = ps - protein_patch.shape[1]
            pad_x = ps - protein_patch.shape[2]
            protein_patch = np.pad(
                protein_patch,
                ((0, 0), (0, pad_y), (0, pad_x)),
                mode="constant", constant_values=0,
            )
            cell_patch = np.pad(
                cell_patch,
                ((0, 0), (0, pad_y), (0, pad_x)),
                mode="constant", constant_values=0,
            )

        protein_has_signal = np.any(protein_patch > 0)
        cell_has_signal = np.any(cell_patch > 0)

        return (not protein_has_signal) and (not cell_has_signal)

    # -----------------------------------------------------------------
    # Index building
    # -----------------------------------------------------------------

    def _build_index(self) -> None:
        """Walk all OME-Zarr stores, compute patch coordinates, check emptiness."""
        stores = self._discover_zarr_stores()
        total_patches = 0
        empty_count = 0

        t0 = time.time()

        for store_info in stores:
            zarr_key = store_info["zarr_key"]
            grp = store_info["group"]
            reg = store_info["region"]
            pid = store_info["patient_id"]
            label = 0 if grp == "HC" else 1

            if "fov_id_override" in store_info:
                fovs = [store_info["fov_id_override"]]
                is_store_fov = True
            else:
                fovs = self._discover_fovs_in_store(zarr_key)
                is_store_fov = False

            if self.max_fovs_per_patient is not None:
                fovs = fovs[: self.max_fovs_per_patient]

            logger.info("  %s/%s/%s -- %d FOVs", grp, reg, pid, len(fovs))

            for fov_id in fovs:
                try:
                    root = self._get_zarr_root(zarr_key)
                except Exception as exc:
                    logger.warning("Failed to open %s: %s -- skipping.", zarr_key, exc)
                    continue

                if is_store_fov:
                    fov_root = root
                elif fov_id != "root":
                    try:
                        fov_root = root[fov_id]
                    except KeyError:
                        logger.warning(
                            "FOV group '%s' not found in %s -- skipping.",
                            fov_id, zarr_key,
                        )
                        continue
                else:
                    fov_root = root

                try:
                    raw_arr = fov_root["0"]
                    z_native = raw_arr.shape[1]
                except (KeyError, IndexError) as exc:
                    logger.warning(
                        "Cannot read Z depth from %s [FOV=%s]: %s -- skipping.",
                        zarr_key, fov_id, exc,
                    )
                    continue

                z_padded = z_native < self.target_z

                for y0 in self._y_starts:
                    for x0 in self._x_starts:
                        is_empty = self._is_patch_empty(fov_root, y0, x0)

                        meta = PatchMeta(
                            zarr_path=zarr_key,
                            group=grp,
                            region=reg,
                            patient_id=pid,
                            fov_id=fov_id,
                            y_start=y0,
                            x_start=x0,
                            z_native=z_native,
                            z_padded=z_padded,
                            is_empty=is_empty,
                            label=label,
                            is_store_fov=is_store_fov,
                        )

                        total_patches += 1
                        if is_empty:
                            empty_count += 1

                        if self.filter_empty and is_empty:
                            continue

                        self.patches.append(meta)

        elapsed = time.time() - t0
        logger.info(
            "Index built in %.1f s -- total patches: %d, empty: %d, retained: %d",
            elapsed, total_patches, empty_count, len(self.patches),
        )

    # -----------------------------------------------------------------
    # Index persistence
    # -----------------------------------------------------------------

    def _save_index(self, path: Path) -> None:
        """Serialize ``self.patches`` to a JSON file."""
        data = {
            "config": {
                "base_folder": self.base_folder,
                "model_type": self.model_type,
                "patch_size": self.patch_size,
                "target_z": self.target_z,
                "stride": self.stride,
                "fov_size": self.fov_size,
                "filter_empty": self.filter_empty,
                "regions": self.regions,
                "groups": self.groups,
            },
            "patches": [p.to_dict() for p in self.patches],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Patch index saved to %s  (%d patches)", path, len(self.patches))

    def _load_index(self, path: Path) -> None:
        """Load ``self.patches`` from a previously saved JSON file.

        Patches are filtered by ``self.regions`` and ``self.groups``
        so that even if the JSON index contains patches from multiple
        regions or groups, only those matching the current configuration
        are retained.  This prevents region mixing when a shared index
        file is reused across different training runs.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = data.get("config", {})
        for key in ("patch_size", "target_z", "model_type", "stride"):
            saved = cfg.get(key)
            current = getattr(self, key, None)
            if saved is not None and current is not None and saved != current:
                logger.warning(
                    "Index config mismatch for '%s': saved=%s, current=%s.  "
                    "Consider re-building the index.",
                    key, saved, current,
                )

        all_patches = [PatchMeta.from_dict(d) for d in data["patches"]]

        # ── Filter by regions and groups ────────────────────────
        # The cached index may contain patches from ALL regions and
        # groups.  We must keep only those matching the current
        # configuration to prevent region mixing.
        region_set = set(self.regions) if self.regions else None
        group_set  = set(self.groups)  if self.groups  else None

        before = len(all_patches)
        self.patches = [
            p for p in all_patches
            if (region_set is None or p.region in region_set)
            and (group_set  is None or p.group  in group_set)
        ]
        after = len(self.patches)
        n_filtered = before - after

        if n_filtered > 0:
            logger.info(
                "Filtered index: %d -> %d patches (removed %d not matching "
                "regions=%s groups=%s)",
                before, after, n_filtered,
                self.regions, self.groups,
            )

        logger.info("Loaded %d patches from %s", len(self.patches), path)

    # -----------------------------------------------------------------
    # Data loading (per-patch)
    # -----------------------------------------------------------------

    def _load_raw_patch_from_cache(
        self,
        zarr_path: str,
        fov_id: str,
        y_start: int,
        x_start: int,
        z_native: int,
        is_store_fov: bool = False,
    ) -> np.ndarray:
        """
        Load a single spatial patch from the in-memory cache.

        This is the fast path used after ``preload_data()`` has been called.
        Reads from numpy arrays in RAM -- zero disk/network I/O.

        Note: After the Phase 2 RAM-loading fix, the cached arrays only
        contain the channels needed for the current model type (shape
        ``(C_needed, Z, Y, X)``), not the full OME-Zarr array.

        Returns RAW (unnormalized) float32 data.
        """
        ps = self.patch_size
        cache_key = (zarr_path, fov_id)

        if cache_key not in self._raw_cache:
            # Fallback: load from zarr (SSD or S3)
            try:
                root = self._get_zarr_root(zarr_path)
                return self._load_raw_patch(
                    root, fov_id, y_start, x_start, z_native, is_store_fov,
                )
            except KeyError as e:
                logger.warning(
                    "Cannot load patch from %s (fov=%s): %s — "
                    "returning zeros. This FOV may have an incompatible "
                    "zarr structure.",
                    zarr_path, fov_id, e,
                )
                # Return a zero patch so training doesn't crash.
                # The patch will be normalized to zeros, which is
                # uninformative but won't break the batch.
                n_ch = len(self._channels_to_load)
                return np.zeros(
                    (n_ch, z_native, self.patch_size, self.patch_size),
                    dtype=np.float32,
                )

        raw_arr = self._raw_cache[cache_key]  # (C_needed, Z, Y, X)
        fov_y = raw_arr.shape[2]
        fov_x = raw_arr.shape[3]

        y_end = min(y_start + ps, fov_y)
        x_end = min(x_start + ps, fov_x)

        # Slice spatial window from all channels at once
        patch = raw_arr[:, :, y_start:y_end, x_start:x_end]
        h_read = patch.shape[2]
        w_read = patch.shape[3]

        if h_read < ps or w_read < ps:
            pad_y = ps - h_read
            pad_x = ps - w_read
            patch = np.pad(
                patch,
                ((0, 0), (0, 0), (0, pad_y), (0, pad_x)),
                mode="constant", constant_values=0,
            )

        return patch.astype(np.float32)

    def _load_raw_patch(
        self,
        root: zarr.Group,
        fov_id: str,
        y_start: int,
        x_start: int,
        z_native: int,
        is_store_fov: bool = False,
    ) -> np.ndarray:
        """
        Load a single spatial patch from all requested channels and return
        a float32 array of shape ``(C_out, Z, patch_size, patch_size)``
        where ``C_out`` depends on ``self.model_type``.

        Returns RAW (unnormalized) float32 data.
        """
        ps = self.patch_size

        fov_root = root if (fov_id == "root" or is_store_fov) else root[fov_id]

        # ── Robust data array access ────────────────────────────────
        # OME-Zarr stores typically put the highest-resolution data
        # under key "0", but some stores use a different layout.
        # We try "0" first, then fall back to scanning for the first
        # array that looks like multi-channel image data.
        raw_arr = self._find_data_array(fov_root)

        fov_y = raw_arr.shape[2]
        fov_x = raw_arr.shape[3]

        y_end = min(y_start + ps, fov_y)
        x_end = min(x_start + ps, fov_x)

        channels: List[np.ndarray] = []
        for ch_idx in self._channels_to_load:
            patch_3d = raw_arr[ch_idx, :, y_start:y_end, x_start:x_end]
            h_read = patch_3d.shape[1]
            w_read = patch_3d.shape[2]

            if h_read < ps or w_read < ps:
                pad_y = ps - h_read
                pad_x = ps - w_read
                patch_3d = np.pad(
                    patch_3d,
                    ((0, 0), (0, pad_y), (0, pad_x)),
                    mode="constant", constant_values=0,
                )

            channels.append(patch_3d)

        raw_patch = np.stack(channels, axis=0).astype(np.float32)
        return raw_patch

    def _find_data_array(self, fov_root) -> zarr.Array:
        """
        Find the raw image data array inside an OME-Zarr FOV group.

        Standard OME-Zarr layout uses ``fov_root["0"]`` for the
        highest-resolution level.  Some stores may lack the "0" key
        (e.g. if they store data directly or use a different naming
        convention).  This method tries the standard key first, then
        falls back to scanning for the first zarr Array that matches
        the expected shape pattern ``(C, Z, Y, X)``.

        Parameters
        ----------
        fov_root : zarr.Group
            The FOV group to search.

        Returns
        -------
        zarr.Array
            The data array of shape ``(C, Z, Y, X)``.

        Raises
        ------
        KeyError
            If no suitable data array can be found.
        """
        # Try standard OME-Zarr key "0" first
        try:
            arr = fov_root["0"]
            if isinstance(arr, zarr.Array):
                return arr
        except KeyError:
            pass

        # Fallback: scan group members for a suitable array
        # Some stores might use different resolution keys or store
        # the data array directly under the FOV group.
        if isinstance(fov_root, zarr.Group):
            # Check if the group itself looks like it has array members
            for key in fov_root.keys():
                try:
                    member = fov_root[key]
                    if isinstance(member, zarr.Array) and member.ndim == 4:
                        logger.info(
                            "Found data array under key '%s' instead of "
                            "'0' in FOV group. Shape: %s",
                            key, member.shape,
                        )
                        return member
                except (KeyError, AttributeError):
                    continue

            # Check if the fov_root itself is an array (edge case)
            if hasattr(fov_root, 'shape') and hasattr(fov_root, 'ndim'):
                logger.warning(
                    "FOV root appears to be an array directly "
                    "(shape=%s). Using it as data array.",
                    getattr(fov_root, 'shape', '?'),
                )
                return fov_root

        # Last resort: list available keys for debugging
        available = list(fov_root.keys()) if hasattr(fov_root, 'keys') else []
        raise KeyError(
            f"Cannot find data array in FOV group. "
            f"Available keys: {available}. "
            f"Expected key '0' or a 4D zarr.Array. "
            f"FOV group type: {type(fov_root).__name__}"
        )

    @staticmethod
    def _z_pad(arr: np.ndarray, target_z: int) -> np.ndarray:
        """Zero-pad ``arr`` along the Z axis (axis=1) to reach ``target_z``."""
        current_z = arr.shape[1]
        if current_z >= target_z:
            return arr

        diff = target_z - current_z
        pad_before = diff // 2
        pad_after = diff - pad_before

        return np.pad(
            arr,
            ((0, 0), (pad_before, pad_after), (0, 0), (0, 0)),
            mode="constant", constant_values=0,
        )

    # -----------------------------------------------------------------
    # Dataset interface
    # -----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Return ``(tensor, meta_dict)`` for patch ``idx``.

        Processing pipeline:
          1. Load raw patch from cache or zarr (unnormalized)
          2. Z-pad if necessary
          3. Normalize per-channel (``self.normalization``)
          4. Augment if ``self.augment`` is True
          5. Convert to torch tensor

        * ``tensor`` shape: ``(C_out, target_z, patch_size, patch_size)``
        * ``meta_dict``: serialisable metadata for this patch.
        """
        meta = self.patches[idx]

        # Fast path: read from in-memory cache (after preload_data())
        if self._is_preloaded and self._raw_cache:
            raw_patch = self._load_raw_patch_from_cache(
                zarr_path=meta.zarr_path,
                fov_id=meta.fov_id,
                y_start=meta.y_start,
                x_start=meta.x_start,
                z_native=meta.z_native,
                is_store_fov=meta.is_store_fov,
            )
        else:
            # Slow path: read from zarr (S3 or local SSD)
            root = self._get_zarr_root(meta.zarr_path)
            raw_patch = self._load_raw_patch(
                root=root,
                fov_id=meta.fov_id,
                y_start=meta.y_start,
                x_start=meta.x_start,
                z_native=meta.z_native,
                is_store_fov=meta.is_store_fov,
            )

        # Z-padding (on raw data before normalization)
        if meta.z_padded:
            raw_patch = self._z_pad(raw_patch, self.target_z)

        # Normalize per-channel (pass patient_id for intensity_pipeline mode)
        patch = self._normalize_patch(raw_patch, patient_id=meta.patient_id)

        # Augment (after normalization, before tensor conversion)
        if self.augment:
            patch = self._augment_patch(patch)

        # Convert to torch tensor
        tensor = torch.from_numpy(patch)  # (C_out, Z, ps, ps)

        # Build lightweight metadata dict
        meta_dict = {
            "zarr_path": meta.zarr_path,
            "group": meta.group,
            "region": meta.region,
            "patient_id": meta.patient_id,
            "fov_id": meta.fov_id,
            "y_start": meta.y_start,
            "x_start": meta.x_start,
            "z_native": meta.z_native,
            "z_padded": meta.z_padded,
            "label": meta.label,
            "is_store_fov": meta.is_store_fov,
        }

        return tensor, meta_dict

    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------

    def get_labels(self) -> torch.Tensor:
        """Return a 1-D int tensor of labels (0=HC, 1=PD) for all patches."""
        return torch.tensor([p.label for p in self.patches], dtype=torch.long)

    def get_groups(self) -> List[str]:
        """Return the cohort group (HC/PD) for each patch."""
        return [p.group for p in self.patches]

    def get_patient_ids(self) -> List[str]:
        """Return the patient ID for each patch."""
        return [p.patient_id for p in self.patches]

    def get_regions(self) -> List[str]:
        """Return the brain region for each patch."""
        return [p.region for p in self.patches]

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        n = len(self.patches)
        if n == 0:
            return "ZarrPatchDataset is empty."

        labels = self.get_labels()
        n_hc = int((labels == 0).sum())
        n_pd = int((labels == 1).sum())
        n_padded = sum(1 for p in self.patches if p.z_padded)
        regions_set = set(p.region for p in self.patches)
        patients_set = set(p.patient_id for p in self.patches)

        lines = [
            f"ZarrPatchDataset summary",
            f"  Model type    : {self.model_type}  -> {self.n_output_channels} channel(s)",
            f"  Patch size    : {self.patch_size}x{self.patch_size}",
            f"  Target Z      : {self.target_z}",
            f"  Stride        : {self.stride}",
            f"  Total patches : {n}",
            f"  HC patches    : {n_hc}",
            f"  PD patches    : {n_pd}",
            f"  Z-padded      : {n_padded}",
            f"  Regions       : {sorted(regions_set)}",
            f"  Patients      : {sorted(patients_set)}  ({len(patients_set)} total)",
            f"  Pre-loaded    : {self._is_preloaded}  ({len(self._raw_cache)} FOVs in RAM)",
            f"  Normalization : {self.normalization}",
            f"  Augment       : {self.augment}",
        ]
        return "\n".join(lines)



# Collate helper -- for use with DataLoader


def ensure_zarr_data_local(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str = "YOUR_STORAGE_NAME",
    base_folder: str = "YOUR_BASE_PREFIX",
    local_cache_dir: str = "/tmp/zarr_cache",
    max_workers: int = 8,
    clearml_dataset_project: Optional[str] = None,
    clearml_dataset_name: Optional[str] = None,
    regions: Optional[List[str]] = None,
    groups: Optional[List[str]] = None,
) -> str:
    """
    Ensure zarr data is available on local SSD.

    Strategy (pick first available):
      1. Check if data already exists in ``local_cache_dir`` on disk.
      2. Multi-threaded download from S3 (MinIO).

    NOTE: ClearML Dataset upload is intentionally DISABLED.  Uploading
    ~25 GB of zarr data to ClearML storage via HTTPS takes many hours
    and is redundant -- the data already lives on MinIO S3, which is
    much faster to download from.  The local SSD cache at
    ``local_cache_dir`` persists across tasks on the same agent, so
    re-downloads are only needed on the first run.

    Returns the local path where zarr data is available.
    """
    cache_path = Path(local_cache_dir)

    # ------------------------------------------------------------------
    # Strategy 1: Data already on disk from a previous run
    # ------------------------------------------------------------------
    if cache_path.exists() and any(cache_path.iterdir()):
        n_items = sum(1 for _ in cache_path.rglob(".zgroup"))
        if n_items > 0:
            logger.info(
                "Zarr data already on disk at %s (%d stores found). "
                "Skipping download.",
                cache_path, n_items,
            )
            print(
                f"[CACHE HIT] Zarr data already on disk at {cache_path} "
                f"({n_items} stores). No download needed!",
                flush=True,
            )
            return str(cache_path)

    # ------------------------------------------------------------------
    # Strategy 2: Multi-threaded download from S3 (MinIO)
    # ------------------------------------------------------------------
    logger.info(
        "No local cache found. Downloading zarr stores from S3 "
        "with %d workers ...",
        max_workers,
    )
    print(
        f"[S3 DOWNLOAD] No local cache found. "
        f"Downloading zarr stores from S3 with {max_workers} workers ...",
        flush=True,
    )
    fs = _make_s3fs(minio_endpoint, minio_access_key, minio_secret_key)
    regions = regions or ["putamen", "substantiaNigra"]
    groups = groups or ["HC", "PD"]
    base_folder = base_folder.rstrip("/")

    # Discover all zarr stores on S3
    all_zarr_keys: List[str] = []
    for grp in groups:
        for reg in regions:
            folder_prefix = f"{base_folder}/{grp}/{reg}/"
            bucket_prefix = f"{bucket_name}/{folder_prefix}"
            try:
                entries = fs.ls(bucket_prefix, detail=False)
            except FileNotFoundError:
                logger.warning("Folder not found on S3: %s", folder_prefix)
                continue

            for entry in entries:
                rel = entry.removeprefix(bucket_name + "/").rstrip("/")
                entry_name = rel.replace(folder_prefix.rstrip("/"), "").strip("/")
                if not entry_name or entry_name.startswith("."):
                    continue

                # Check if this is a zarr store directly
                s3_path = f"{bucket_name}/{rel}"
                if _is_zarr_store_static(fs, s3_path):
                    all_zarr_keys.append(rel)
                    continue

                # Or contains sub-FOV zarr stores
                try:
                    sub_entries = fs.ls(s3_path, detail=False)
                except FileNotFoundError:
                    continue
                for sub in sub_entries:
                    sub_rel = sub.removeprefix(bucket_name + "/").rstrip("/")
                    sub_name = sub_rel.replace(rel + "/", "").strip("/")
                    if not sub_name or sub_name.startswith("."):
                        continue
                    if _is_zarr_store_static(fs, sub.rstrip("/")):
                        all_zarr_keys.append(sub_rel)

    logger.info("Discovered %d zarr stores on S3.", len(all_zarr_keys))
    print(
        f"[S3 DOWNLOAD] Discovered {len(all_zarr_keys)} zarr stores on S3.",
        flush=True,
    )

    # Multi-threaded download
    cache_path.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def _download_one(zarr_key: str) -> str:
        s3_path = f"{bucket_name}/{zarr_key}"
        local_path = cache_path / zarr_key

        # Validate existing cache: must contain zarr metadata files.
        if local_path.exists():
            has_zarr_meta = (
                (local_path / ".zgroup").exists()
                or (local_path / ".zarray").exists()
                or (local_path / "0" / ".zarray").exists()
            )
            if has_zarr_meta:
                return zarr_key  # valid cache
            else:
                logger.warning(
                    "Corrupt/incomplete cache for %s — "
                    "removing and re-downloading.",
                    zarr_key,
                )
                shutil.rmtree(local_path, ignore_errors=True)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _retry_s3(
                fs.get, s3_path, str(local_path),
                recursive=True,
                max_retries=5, base_delay=2.0,
            )
        except Exception:
            # Clean up partially downloaded directory on failure
            if local_path.exists():
                shutil.rmtree(local_path, ignore_errors=True)
            raise
        return zarr_key

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, k): k for k in all_zarr_keys}
        done = 0
        for f in as_completed(futures):
            done += 1
            try:
                f.result()
            except Exception as e:
                logger.error("Download failed for %s: %s", futures[f], e)
            if done % 5 == 0 or done == len(all_zarr_keys):
                logger.info("  Downloaded %d/%d stores ...", done, len(all_zarr_keys))
                print(
                    f"[S3 DOWNLOAD]   {done}/{len(all_zarr_keys)} stores downloaded ...",
                    flush=True,
                )

    dl_time = time.time() - t0
    logger.info(
        "S3 download complete: %d stores in %.1f seconds.",
        len(all_zarr_keys), dl_time,
    )
    print(
        f"[S3 DOWNLOAD] Complete! {len(all_zarr_keys)} stores in {dl_time:.0f}s",
        flush=True,
    )

    return str(cache_path)


def _is_zarr_store_static(
    fs: s3fs.S3FileSystem, s3_path: str
) -> bool:
    """Check whether *s3_path* points to a valid zarr store (module-level)."""
    return (
        fs.exists(f"{s3_path}/.zgroup")
        or fs.exists(f"{s3_path}/.zarray")
        or fs.exists(f"{s3_path}/0/.zarray")
    )


def patch_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]],
) -> Tuple[torch.Tensor, Dict[str, List[Any]]]:
    """
    Custom collate function that stacks tensors and collects metadata
    into lists (no padding -- all tensors have identical shape).
    """
    tensors, metas = zip(*batch)
    stacked = torch.stack(tensors, dim=0)

    meta_combined: Dict[str, List[Any]] = {}
    for m in metas:
        for k, v in m.items():
            meta_combined.setdefault(k, []).append(v)

    return stacked, meta_combined



# Quick sanity test (run as script)

if __name__ == "__main__":
    from clearml import Task

    Task.add_requirements("boto3")
    Task.add_requirements("s3fs")

    task = Task.init(
        project_name="YOUR_STORAGE_NAME/Data_Download",
        task_name="Step_2.1_Patch_Indexing",
    )

    task.execute_remotely(queue_name="default")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "YOUR_MINIO_ENDPOINT")
    MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "YOUR_MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "YOUR_MINIO_SECRET_KEY")

    LOCAL_INDEX_PATH = "patch_index.json"
    MINIO_INDEX_BACKUP = "YOUR_STORAGE_NAME/YOUR_BASE_PREFIX/_system/patch_index_C.json"

    if not os.path.exists(LOCAL_INDEX_PATH):
        logger.info("Local index not found. Trying to download from MinIO backup...")
        try:
            fs = _make_s3fs(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)
            if fs.exists(MINIO_INDEX_BACKUP):
                fs.get(MINIO_INDEX_BACKUP, LOCAL_INDEX_PATH)
                logger.info("Successfully restored index from MinIO!")
            else:
                logger.info("MinIO backup does not exist. Will scan from scratch.")
        except Exception as e:
            logger.warning(f"Failed to check/download MinIO backup: {e}")

    ds = ZarrPatchDataset(
        minio_endpoint=MINIO_ENDPOINT,
        minio_access_key=MINIO_ACCESS_KEY,
        minio_secret_key=MINIO_SECRET_KEY,
        bucket_name="YOUR_STORAGE_NAME",
        base_folder="YOUR_BASE_PREFIX",
        model_type="C",
        patch_size=256,
        target_z=25,
        stride=236,
        filter_empty=True,
        index_path=LOCAL_INDEX_PATH,
        fov_size=1200,
    )

    if os.path.exists(LOCAL_INDEX_PATH):
        try:
            fs = _make_s3fs(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)
            if not fs.exists(MINIO_INDEX_BACKUP):
                logger.info("Uploading patch index backup to MinIO...")
                fs.put(LOCAL_INDEX_PATH, MINIO_INDEX_BACKUP)
                logger.info("Backup saved to MinIO!")
        except Exception as e:
            logger.warning(f"Failed to upload backup to MinIO: {e}")

    task.upload_artifact(
        name="patch_metadata_C",
        artifact_object=[p.to_dict() for p in ds.patches],
    )

    logger.info(ds.summary())

    # Quick sanity: read one patch
    tensor, meta = ds[0]
    logger.info("Patch 0: tensor shape=%s, label=%s", tensor.shape, meta["label"])
