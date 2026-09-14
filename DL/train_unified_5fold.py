"""
Unified 5-Fold GroupKFold Training Pipeline — Models A, B, C
=============================================================

Trains ALL THREE models (A=IBA1, B=pSyn, C=both) simultaneously in a
single ClearML task using GroupKFold(5) cross-validation.

Key design decisions
--------------------
1. **Unified Dataloader**: Dataset outputs (2, 25, H, W) with model_type="C".
   After reshape to (50, H, W):
     - Channels  0-24 = IBA1 z-slices  (Model A input)
     - Channels 25-49 = pSyn  z-slices  (Model B input)
     - All 50 channels                = Model C input
   NOTE: Internal channel order is [IBA1, pSyn] because the dataset's
   _MODEL_CHANNELS["C"] = [_CH_IBA1=1, _CH_PSYN=0].  The mapping is
   documented and consistent throughout the pipeline.

2. **Summed loss, single backward**: loss = loss_A + loss_B + loss_C.
   Gradients are independent (no shared parameters), so each model
   learns as if trained separately — but data loading happens once.

3. **Smart conv1 init**: Average pretrained ImageNet conv1 across the
   channel dim, tile to new channel count, divide by (N/3).

4. **Frozen layers**: layer1 + layer2 permanently frozen.
   Only layer3, layer4, fc are trained.

5. **GroupKFold(5)**: Groups by Patient_ID to prevent data leakage.
   Inner GroupKFold(4) splits train into train+val.

6. **Patient-level AUC**: Patch probabilities aggregated by patient
   (mean) for AUC computation during validation/test.

Usage
-----
    python train_unified_5fold.py

Designed for ClearML remote execution but works locally too.
"""

from __future__ import annotations

# ==================================================================
#  0.  MinIO + ClearML setup BEFORE any other imports
# ==================================================================

import os

os.environ["AWS_ACCESS_KEY_ID"]     = "YOUR_MINIO_ACCESS_KEY"
os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR_MINIO_SECRET_KEY"
os.environ["AWS_ENDPOINT_URL"]      = "YOUR_MINIO_ENDPOINT"
os.environ["CLEARML_AGENT_BOTO3_ENDPOINT_URL"] = (
    "YOUR_MINIO_ENDPOINT"
)
os.environ["AWS_DEFAULT_REGION"]    = "us-east-1"

from clearml import Task

Task.add_requirements("boto3")
Task.add_requirements("s3fs")
Task.add_requirements("scikit-learn")
Task.add_requirements("scipy")
Task.add_requirements("matplotlib")
Task.add_requirements("pandas")
Task.add_requirements("seaborn")

# ── Default hyper-parameters (overridable from ClearML UI) ────
_params = dict(
    # Dataset
    architecture    = "cnn",
    region          = "putamen",       # set per-loop at runtime
    regions         = "substantiaNigra,putamen",   # comma-separated
    seeds           = "42,43,44",                  # comma-separated
    patch_size      = 256,
    target_z        = 25,
    stride          = 236,
    fov_size        = 1200,
    filter_empty    = True,
    normalization   = "intensity_pipeline",
    # Training
    batch_size      = 64,
    max_epochs      = 15,
    lr              = 1e-4,
    weight_decay    = 1e-2,
    patience        = 5,
    warmup_epochs   = 3,              # linear warmup duration
    dropout         = 0.5,
    label_smoothing = 0.0,            # plain CE by default
    num_workers     = 8,
    seed            = 42,
    # Pre-loading
    local_cache_dir  = "/tmp/zarr_cache",
    preload_to_ram   = True,
    max_ram_gb       = 64.0,
    max_workers      = 8,
    # ClearML Dataset
    clearml_dataset_project = "YOUR_STORAGE_NAME/DL_Training",
    clearml_dataset_name    = "zarr_microglia_data",
    # Queue
    queue_name      = "YOUR_QUEUE_NAME",
)

task = Task.init(
    project_name="YOUR_STORAGE_NAME/DL_Training",
    task_name=f"Unified_ABC_{_params['architecture']}_ALL_v3",
)

task.connect(_params)
task.execute_remotely(queue_name=_params["queue_name"])

# ==================================================================
#  1.  Imports
# ==================================================================

import copy
import io
import json
import logging
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
_font_candidates = [
    '/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]
_sans_serif = []
for _fpath in _font_candidates:
    if os.path.isfile(_fpath):
        try:
            fm.fontManager.addfont(_fpath)
            _font_name = fm.FontProperties(fname=_fpath).get_name()
            _sans_serif.append(_font_name)
        except Exception:
            pass
if not _sans_serif:
    _sans_serif = ['DejaVu Sans']
matplotlib.rcParams['font.sans-serif'] = _sans_serif + ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import s3fs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from torchvision.models import resnet18, ResNet18_Weights

# Project-local imports
from zarr_patch_dataset import (
    ZarrPatchDataset,
    patch_collate_fn,
    _make_s3fs,
    ensure_zarr_data_local,
)
from intensity_normalization import (
    run_full_pre_scan,
    compute_target_stats,
    run_spearman_bias_test,
    IntensityNormalizer,
    save_stats_to_json,
    load_stats_from_json,
)

import gc

logger = logging.getLogger(__name__)

# Per-donor intensity statistics, filled by main() before run_training().
_GLOBAL_PATIENT_STATS = {}


# ==================================================================
#  2.  Configuration
# ==================================================================

# ── MinIO (fixed) ─────────────────────────────────────────────
MINIO_ENDPOINT  = "YOUR_MINIO_ENDPOINT"
MINIO_ACCESS    = "YOUR_MINIO_ACCESS_KEY"
MINIO_SECRET    = "YOUR_MINIO_SECRET_KEY"
MINIO_BUCKET    = "YOUR_STORAGE_NAME"
MINIO_BASE      = "YOUR_BASE_PREFIX"

# ── New results root on MinIO ─────────────────────────────────
MINIO_RESULTS_ROOT = "YOUR_OUTPUT_PREFIX"

PATCH_INDEX_MINIO = "YOUR_STORAGE_NAME/YOUR_BASE_PREFIX/_system/patch_index_FROM_CLEARML.json"
LOCAL_INDEX_PATH  = "/tmp/patch_index_FROM_CLEARML.json"

# ── Outlier patients excluded from ML dataset ────────────────
EXCLUDED_PATIENTS = {"PD8", "PD10", "HC9"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tag() -> str:
    """Suffix that keeps ClearML scalar series separate across seeds."""
    return f"s{int(_params['seed'])}"


def minio_path(model_type: str, region: str, *parts: str) -> str:
    """Build MinIO key: YOUR_OUTPUT_PREFIX/{MODEL_TYPE}_{ARCH}/{REGION}/{parts}"""
    arch = _params["architecture"]
    # v2 runs are written under their own prefix, one directory per seed,
    # so repeated cross-validation runs never overwrite each other.
    base = (f"{MINIO_RESULTS_ROOT}_v2/seed{int(_params['seed'])}"
            f"/{model_type}_{arch}/{region}")
    if parts:
        base += "/" + "/".join(parts)
    return base


# ==================================================================
#  3.  Smart Conv1 Initialization + Model Creation
# ==================================================================

def _smart_conv1_init(in_channels: int, pretrained_conv1_weight: torch.Tensor) -> nn.Conv2d:
    """
    Create conv1 with smart initialization from ImageNet weights.

    Strategy:
      1. Take pretrained conv1 weights (64, 3, 7, 7)
      2. Average across channel dim -> (64, 1, 7, 7)
      3. Tile/repeat to (64, in_channels, 7, 7)
      4. Divide by (in_channels / 3) to preserve activation magnitudes

    This avoids the discontinuity between "pretrained" and "Kaiming"
    channels that caused AUC crashes in the previous approach.
    """
    orig_weight = pretrained_conv1_weight.clone()  # (64, 3, 7, 7)

    # Average across the 3 input channels -> (64, 1, 7, 7)
    avg_weight = orig_weight.mean(dim=1, keepdim=True)

    # Tile to new channel count -> (64, in_channels, 7, 7)
    new_weight = avg_weight.repeat(1, in_channels, 1, 1)

    # Scale to preserve expected activation magnitude
    scale = in_channels / 3.0
    new_weight = new_weight / scale

    conv1 = nn.Conv2d(
        in_channels, 64,
        kernel_size=7, stride=2, padding=3, bias=False,
    )
    conv1.weight.data = new_weight

    logger.info(
        "Smart conv1 init: %d channels, scale factor=%.2f",
        in_channels, scale,
    )
    return conv1


def create_model(
        in_channels: int,
        dropout: float = 0.5,
        calibration_loader: Optional[DataLoader] = None,
        channel_slice: Optional[slice] = None,
) -> nn.Module:
    """
    Create a ResNet-18 with smart conv1 init, partial freezing,
    and calibrated BatchNorm.

    Loads pretrained ImageNet weights FIRST, then replaces conv1 with
    smart initialization.  Layers 1-2 are frozen (low-level ImageNet
    features), layers 3-4 + fc are trainable (can adapt to microscopy).

    Trainable: conv1, bn1, layer3, layer4, fc
    Frozen:    layer1, layer2

    **BatchNorm calibration**: If a ``calibration_loader`` is provided,
    the model runs one forward pass in train mode to re-compute BN
    running statistics on actual microscopy data BEFORE freezing BN.
    This is CRITICAL because ImageNet running stats are meaningless
    after conv1 is replaced with smart init for 25/50-channel input.
    Using ImageNet stats with microscopy data causes systematic feature
    inversion → AUC < 0.5.

    After calibration, all BN layers are pinned to eval mode with the
    recalibrated running stats, preventing fold-specific batch stats
    on small datasets.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = _smart_conv1_init(in_channels, model.conv1.weight.data)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(512, 2),
    )

    # Freeze layer1 and layer2 (low-level ImageNet features)
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False

    # ── Calibrate BN running stats on microscopy data ──────────
    # ImageNet running stats are INVALID after smart_conv1_init
    # replaces 3-channel conv1 with 25/50-channel conv1.
    # We must re-compute running_mean/running_var on actual data
    # before pinning BN to eval mode.
    if calibration_loader is not None:
        logger.info("Calibrating BN running stats on microscopy data ...")

        # ИСПРАВЛЕНИЕ: перенести модель на GPU ДО калибровки!
        # Без этого вход на GPU, а веса на CPU → RuntimeError
        model = model.to(DEVICE)

        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()
                module.momentum = 0.1
                module.training = True

        # Determine which channel slice to use based on in_channels
        # Model A: channels 0-24 (IBA1), Model B: channels 25-49 (pSyn),
        # Model C: all 50 channels
        if channel_slice is not None:
            ch_slice = channel_slice
        elif in_channels == 25:
            ch_slice = slice(0, 25)
        else:
            ch_slice = slice(None)

        # Run calibration batches (use momentum-based update)
        with torch.no_grad():
            n_cal_batches = 0
            for tensors, metas in calibration_loader:
                tensors = tensors.to(DEVICE)
                B, C, Z, H, W = tensors.shape
                x = tensors.reshape(B, C * Z, H, W)
                x_input = x[:, ch_slice, :, :]
                _ = model(x_input)  # теперь и данные, и модель на GPU ✅
                n_cal_batches += 1
                if n_cal_batches >= 50:
                    break
        logger.info(
            "BN calibration done: %d batches, in_channels=%d. "
            "Running stats now reflect microscopy data distribution.",
            n_cal_batches, in_channels,
        )

    # Pin all BN layers to eval mode permanently.
    # After calibration, running_stats reflect the actual microscopy
    # data distribution (if calibration_loader was provided).
    # If no calibration_loader, we still freeze BN but warn that
    # running stats may be from ImageNet and could cause AUC < 0.5.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False
            module.track_running_stats = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(
        "Model created: in_channels=%d, trainable=%.2fM, frozen=%.2fM, "
        "BN_calibrated=%s",
        in_channels, n_trainable / 1e6, n_frozen / 1e6,
                     calibration_loader is not None,
    )
    return model


# ==================================================================
#  4.  Utility helpers
# ==================================================================

def seed_everything(seed: int) -> None:
    """Reproducibility helper."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def upload_to_minio(local_path: str, minio_key: str, fs: s3fs.S3FileSystem) -> None:
    s3_uri = f"{MINIO_BUCKET}/{minio_key}"
    try:
        fs.put(local_path, s3_uri)
        logger.info("Uploaded -> s3://%s", s3_uri)
    except Exception as exc:
        logger.error("Failed to upload %s -> %s: %s", local_path, s3_uri, exc)


def upload_bytes_to_minio(data: bytes, minio_key: str, fs: s3fs.S3FileSystem) -> None:
    s3_uri = f"{MINIO_BUCKET}/{minio_key}"
    try:
        with fs.open(s3_uri, "wb") as f:
            f.write(data)
        logger.info("Uploaded (bytes) -> s3://%s", s3_uri)
    except Exception as exc:
        logger.error("Failed to upload bytes -> %s: %s", s3_uri, exc)


def upload_df_to_minio(df: pd.DataFrame, minio_key: str, fs: s3fs.S3FileSystem) -> None:
    """Upload a DataFrame as CSV to MinIO."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    upload_bytes_to_minio(csv_bytes, minio_key, fs)


def _report_image_to_clearml(
    clearml_task: Task, title: str, series: str,
    png_bytes: bytes, iteration: int = 0,
) -> None:
    try:
        # ClearML uploads asynchronously. Deleting the file straight after the
        # call raced the uploader and every figure was dropped with
        # "Skipping upload, could not find object file". Keep the files for the
        # lifetime of the task instead -- they are a few hundred KB each.
        _img_dir = "/tmp/clearml_report_images"
        os.makedirs(_img_dir, exist_ok=True)
        _safe = "".join(c if c.isalnum() or c in "-_." else "_"
                        for c in f"{title}_{series}_{iteration}")
        tmp_path = os.path.join(_img_dir, _safe + ".png")
        with open(tmp_path, "wb") as tmp:
            tmp.write(png_bytes)
        clearml_task.get_logger().report_image(
            title, series, iteration=iteration, local_path=tmp_path,
        )
    except Exception as exc:
        logger.warning("Failed to report image to ClearML (%s/%s): %s",
                       title, series, exc)


# ==================================================================
#  5.  Unified Training / Evaluation
# ==================================================================

def train_one_epoch(
    models: Dict[str, nn.Module],
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dataset: ZarrPatchDataset,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Train all 3 models for one epoch. Returns dict of
    {model_name: (avg_loss, accuracy, auc_roc)}.
    """
    dataset.augment = True

    for m in models.values():
        m.train()
        # Pin all BN layers to eval mode (use calibrated running stats)
        for module in m.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    # Track per-model metrics
    trackers = {name: {"loss": 0.0, "labels": [], "probs": [], "count": 0}
                for name in models}

    for tensors, metas in loader:
        tensors = tensors.to(device)          # (B, 2, 25, H, W)
        labels  = torch.tensor(metas["label"], dtype=torch.long, device=device)
        batch_size = tensors.size(0)

        # ── Reshape to (B, 50, H, W) ──────────────────────────
        B, C, Z, H, W = tensors.shape
        x = tensors.reshape(B, C * Z, H, W)  # (B, 50, H, W)

        # Channel mapping (from dataset model_type="C"):
        #   ch 0-24 = IBA1 z-slices,  ch 25-49 = pSyn z-slices
        x_A = x[:,  0:25, :, :]   # IBA1  -> Model A
        x_B = x[:, 25:50, :, :]   # pSyn  -> Model B
        x_C = x                    # both  -> Model C

        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=device)
        inputs = {"A": x_A, "B": x_B, "C": x_C}

        for name, model in models.items():
            logits = model(inputs[name])       # (B, 2)
            loss = criterion(logits, labels)
            total_loss = total_loss + loss

            trackers[name]["loss"] += loss.item() * batch_size
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            trackers[name]["probs"].extend(probs)
            trackers[name]["labels"].extend(labels.cpu().tolist())
            trackers[name]["count"] += batch_size

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for m in models.values() for p in m.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()

    dataset.augment = False

    results = {}
    for name, t in trackers.items():
        n = t["count"]
        if n == 0:
            results[name] = (0.0, 0.0, 0.0)
            continue
        avg_loss = t["loss"] / n
        arr_probs = np.array(t["probs"])
        arr_labels = np.array(t["labels"])
        acc = accuracy_score(arr_labels, (arr_probs >= 0.5).astype(int))
        try:
            auc = roc_auc_score(arr_labels, arr_probs)
        except ValueError:
            auc = 0.0
        if np.isnan(auc):
            auc = 0.0
        results[name] = (avg_loss, acc, auc)

    return results


@torch.no_grad()
def evaluate(
    models: Dict[str, nn.Module],
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dataset: ZarrPatchDataset,
) -> Dict[str, Tuple[float, float, float, List[float], List[int]]]:
    """
    Evaluate all 3 models. Returns dict of
    {model_name: (avg_loss, accuracy, auc_roc, all_probs, all_labels)}.
    """
    dataset.augment = False

    for m in models.values():
        m.eval()

    trackers = {name: {"loss": 0.0, "labels": [], "probs": [], "count": 0}
                for name in models}

    for tensors, metas in loader:
        tensors = tensors.to(device)
        labels  = torch.tensor(metas["label"], dtype=torch.long, device=device)
        batch_size = tensors.size(0)

        B, C, Z, H, W = tensors.shape
        x = tensors.reshape(B, C * Z, H, W)

        x_A = x[:,  0:25, :, :]
        x_B = x[:, 25:50, :, :]
        x_C = x

        inputs = {"A": x_A, "B": x_B, "C": x_C}

        for name, model in models.items():
            logits = model(inputs[name])
            loss = criterion(logits, labels)

            trackers[name]["loss"] += loss.item() * batch_size
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            trackers[name]["probs"].extend(probs)
            trackers[name]["labels"].extend(labels.cpu().tolist())
            trackers[name]["count"] += batch_size

    results = {}
    for name, t in trackers.items():
        n = t["count"]
        if n == 0:
            results[name] = (0.0, 0.0, 0.0, [], [])
            continue
        avg_loss = t["loss"] / n
        arr_probs = np.array(t["probs"])
        arr_labels = np.array(t["labels"])
        acc = accuracy_score(arr_labels, (arr_probs >= 0.5).astype(int))
        try:
            auc = roc_auc_score(arr_labels, arr_probs)
        except ValueError:
            auc = 0.0
        if np.isnan(auc):
            auc = 0.0
        results[name] = (avg_loss, acc, auc, t["probs"], t["labels"])

    return results


# ==================================================================
#  6.  Patient-level AUC
# ==================================================================

def compute_patient_auc(
    probs: List[float],
    labels: List[int],
    patient_ids: List[str],
) -> Tuple[float, Dict[str, Dict[str, Any]]]:
    """
    Aggregate patch probabilities to patient level (mean) and compute AUC.

    Returns (patient_auc, patient_details) where patient_details maps
    patient_id -> {"prob": float, "label": int, "pred": int}.
    """
    pid_to_probs: Dict[str, List[float]] = {}
    pid_to_label: Dict[str, int] = {}

    for prob, label, pid in zip(probs, labels, patient_ids):
        pid_to_probs.setdefault(pid, []).append(prob)
        pid_to_label[pid] = label

    patient_probs = []
    patient_labels = []
    patient_details: Dict[str, Dict[str, Any]] = {}

    for pid in sorted(pid_to_probs.keys()):
        mean_prob = float(np.mean(pid_to_probs[pid]))
        lbl = pid_to_label[pid]
        patient_probs.append(mean_prob)
        patient_labels.append(lbl)
        patient_details[pid] = {
            "prob": mean_prob,
            "label": lbl,
            "pred": int(mean_prob >= 0.5),
        }

    try:
        pauc = roc_auc_score(patient_labels, patient_probs)
    except ValueError:
        pauc = 0.0
    if np.isnan(pauc):
        pauc = 0.0

    return pauc, patient_details


# ==================================================================
#  7.  Visualisation helpers
# ==================================================================

def plot_roc_curve(labels, probs, fold_name):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_val = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=150)
    ax.plot(fpr, tpr, color="#2166AC", lw=2.5, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#B2182B", lw=1.2, alpha=0.7, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(f"ROC - {fold_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_confusion_matrix(labels, preds, fold_name):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=150)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["HC", "PD"], yticklabels=["HC", "PD"],
                ax=ax, annot_kws={"size": 16})
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title(f"Confusion Matrix - {fold_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_roc_overlay(all_fold_data):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    tprs_interp = []
    base_fpr = np.linspace(0, 1, 101)
    aucs = []
    cmap = plt.cm.viridis
    n_folds = len(all_fold_data)
    for i, fd in enumerate(all_fold_data):
        fpr, tpr, _ = roc_curve(fd["labels"], fd["probs"])
        auc_val = roc_auc_score(fd["labels"], fd["probs"])
        aucs.append(auc_val)
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs_interp.append(tpr_interp)
        color = cmap(i / max(n_folds - 1, 1))
        ax.plot(fpr, tpr, lw=0.9, alpha=0.5, color=color,
                label=f"{fd['fold_name']} (AUC={auc_val:.2f})")
    tprs_array = np.array(tprs_interp)
    mean_tpr = tprs_array.mean(axis=0)
    std_tpr  = tprs_array.std(axis=0)
    ax.plot(base_fpr, mean_tpr, color="#B2182B", lw=2.5,
            label=f"Mean (AUC={np.mean(aucs):.3f} +/- {np.std(aucs):.3f})")
    ax.fill_between(base_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    color="#B2182B", alpha=0.15, label="+/- 1 SD")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("GroupKFold-5 CV - ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, frameon=True, fancybox=True, ncol=2)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_training_curves(train_losses, val_losses, train_aucs, val_aucs, fold_name):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    ax1.plot(epochs, train_losses, "o-", color="#2166AC", lw=1.8, markersize=3, label="Train")
    ax1.plot(epochs, val_losses,   "s-", color="#B2182B", lw=1.8, markersize=3, label="Val")
    ax1.set_xlabel("Epoch", fontsize=12); ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"Loss - {fold_name}", fontsize=13, fontweight="bold")
    ax1.legend(loc="best", fontsize=10); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, train_aucs, "o-", color="#2166AC", lw=1.8, markersize=3, label="Train")
    ax2.plot(epochs, val_aucs,   "s-", color="#B2182B", lw=1.8, markersize=3, label="Val")
    ax2.axhline(0.5, ls="--", color="grey", alpha=0.5, label="Chance")
    ax2.set_xlabel("Epoch", fontsize=12); ax2.set_ylabel("AUC-ROC", fontsize=12)
    ax2.set_title(f"AUC-ROC - {fold_name}", fontsize=13, fontweight="bold")
    ax2.legend(loc="best", fontsize=10); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ==================================================================
#  8.  Main Training Loop
# ==================================================================

def run_training(
    dataset: ZarrPatchDataset,
    clearml_task: Task,
) -> None:
    """
    Run GroupKFold(5) cross-validation training all 3 models.
    """
    seed_everything(_params["seed"])
    logger.info("Device: %s", DEVICE)

    region = _params["region"]
    arch   = _params["architecture"]
    fs = _make_s3fs(MINIO_ENDPOINT, MINIO_ACCESS, MINIO_SECRET)

    # Note: outlier patients are already removed in main() BEFORE
    # the intensity pre-scan, so dataset.patches is already clean.

    # ── Build group arrays for GroupKFold ──────────────────────
    patient_ids_all = [p.patient_id for p in dataset.patches]
    labels_all      = [p.label for p in dataset.patches]
    groups = np.array(patient_ids_all)
    y      = np.array(labels_all)
    X      = np.arange(len(dataset.patches))

    unique_pids = sorted(set(patient_ids_all))
    logger.info("GroupKFold: %d patches, %d unique patients, region=%s",
                len(dataset.patches), len(unique_pids), region)

    # ── Region integrity check ──────────────────────────────────
    # Verify that ALL patches in the dataset belong to the expected
    # region. If patches from other regions leaked in (e.g. from a
    # corrupted index), log a CRITICAL warning.
    regions_in_data = set(p.region for p in dataset.patches)
    if regions_in_data != {region}:
        logger.critical(
            "REGION LEAK DETECTED! Expected only '%s' but found: %s. "
            "This WILL cause AUC corruption — the model will learn "
            "region differences instead of PD vs HC!",
            region, sorted(regions_in_data),
        )
        raise RuntimeError(
            f"Region leak: expected '{region}', found {sorted(regions_in_data)}. "
            f"ABORTING to prevent corrupted training!"
        )
    else:
        logger.info("Region integrity OK: all patches belong to '%s'", region)

    # Group-STRATIFIED K-fold with shuffling: plain GroupKFold is
    # deterministic, so repeating it under different seeds would reproduce
    # the identical partition. Stratifying also prevents degenerate folds
    # such as the 4-PD / 1-HC putamen fold of the previous run.
    gkf = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=int(_params["seed"]))

    # ── Per-model collectors ───────────────────────────────────
    all_results = {name: {
        "golden_rows": [],
        "fold_roc_data": [],
        "patient_rows": [],
        "fold_summaries": [],
    } for name in ["A", "B", "C"]}

    # ════════════════════════════════════════════════════════════
    #  OUTER LOOP — 5 folds
    # ════════════════════════════════════════════════════════════

    for fold_idx, (train_val_idx, test_idx) in enumerate(
        gkf.split(X, y, groups)
    ):
        fold_name = f"Fold_{fold_idx:02d}"
        logger.info("=" * 60)
        logger.info("FOLD %d/5  (train+val=%d, test=%d)",
                     fold_idx + 1, len(train_val_idx), len(test_idx))
        logger.info("=" * 60)

        # ── Inner split: train_val -> train + val ──────────────
        tv_groups = groups[train_val_idx]
        tv_y      = y[train_val_idx]
        tv_X      = train_val_idx  # actual indices into dataset

        inner_gkf = StratifiedGroupKFold(
            n_splits=4, shuffle=True, random_state=int(_params["seed"]))
        inner_splits = list(inner_gkf.split(tv_X, tv_y, tv_groups))
        train_idx_rel, val_idx_rel = inner_splits[0]
        train_idx = tv_X[train_idx_rel]
        val_idx   = tv_X[val_idx_rel]

        # Log patient composition
        train_pids = sorted(set(groups[train_idx]))
        val_pids   = sorted(set(groups[val_idx]))
        test_pids  = sorted(set(groups[test_idx]))
        logger.info("  Donors -- train=%d val=%d test=%d | "
                    "test PD=%d HC=%d",
                    len(train_pids), len(val_pids), len(test_pids),
                    sum(1 for p in test_pids if str(p).startswith("PD")),
                    sum(1 for p in test_pids if str(p).startswith("HC")))
        logger.info("  Train patients: %s", [str(p) for p in train_pids])
        logger.info("  Val   patients: %s", val_pids)
        logger.info("  Test  patients: %s", test_pids)

        # ── Fold-restricted intensity normalisation ────────────
        # The cohort target and the Spearman bias screen are re-estimated
        # from THIS fold's TRAINING donors only, then applied unchanged to
        # the validation and test donors. Per-donor statistics are derived
        # from each donor's own data and carry no label information.
        _train_only_stats = {
            pid: st for pid, st in _GLOBAL_PATIENT_STATS.items()
            if pid in set(train_pids)
        }
        if not _train_only_stats:
            raise RuntimeError(f"{fold_name}: no training-donor intensity stats")

        _fold_target_stats = compute_target_stats(_train_only_stats)
        _fold_spearman = run_spearman_bias_test(_train_only_stats, alpha=0.05)
        for _sr in _fold_spearman:
            logger.info("  [%s] Spearman (train donors only) [%s]: rho=%.4f p=%.4f %s",
                        fold_name, _sr.channel_name, _sr.rho, _sr.p_value,
                        "BIAS" if _sr.significant else "OK")

        dataset._normalizer = IntensityNormalizer(
            patient_stats=_GLOBAL_PATIENT_STATS,   # per-donor, own data only
            target_stats=_fold_target_stats,       # TRAIN donors of this fold only
            model_type="C",
            use_patient_percentiles=True,
            apply_cross_patient_alignment=True,
            force_alignment=True,
        )
        logger.info("  [%s] normalisation target refit on %d training donors",
                    fold_name, len(_train_only_stats))

        # ── Create DataLoaders ─────────────────────────────────
        train_ds = Subset(dataset, train_idx.tolist())
        val_ds   = Subset(dataset, val_idx.tolist())
        test_ds  = Subset(dataset, test_idx.tolist())

        train_loader = DataLoader(
            train_ds, batch_size=_params["batch_size"], shuffle=True,
            num_workers=_params["num_workers"], collate_fn=patch_collate_fn,
            pin_memory=True, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=_params["batch_size"], shuffle=False,
            num_workers=_params["num_workers"], collate_fn=patch_collate_fn,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=_params["batch_size"], shuffle=False,
            num_workers=_params["num_workers"], collate_fn=patch_collate_fn,
            pin_memory=True,
        )

        # ── Create 3 models with BN calibration ──────────────
        # Each model gets a calibration pass on the TRAIN split
        # to re-compute BN running stats on microscopy data.
        # This replaces the ImageNet running stats that are INVALID
        # after smart_conv1_init replaces 3ch conv1 with 25/50ch.
        models = {
            "A": create_model(
                in_channels=25, dropout=_params["dropout"],
                calibration_loader=train_loader,
                channel_slice=slice(0, 25),  # IBA1 channels
            ).to(DEVICE),
            "B": create_model(
                in_channels=25, dropout=_params["dropout"],
                calibration_loader=train_loader,
                channel_slice=slice(25, 50),  # pSyn channels
            ).to(DEVICE),
            "C": create_model(
                in_channels=50, dropout=_params["dropout"],
                calibration_loader=train_loader,
                channel_slice=slice(None),  # all 50 channels
            ).to(DEVICE),
        }

        # ── Loss function ─────────────────────────────────────
        train_labels_list = [dataset.patches[i].label for i in train_idx]
        counts = np.bincount(train_labels_list, minlength=2).astype(np.float64)
        total = counts.sum()
        if total > 0 and counts.min() > 0:
            weights = total / (2.0 * counts)
            class_weights = torch.tensor(weights.astype(np.float32), device=DEVICE)
        else:
            class_weights = torch.tensor([1.0, 1.0], device=DEVICE)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=_params["label_smoothing"],
        )

        # ── Optimizer (all trainable params from all models) ───
        trainable_params = []
        for model in models.values():
            trainable_params.extend(
                filter(lambda p: p.requires_grad, model.parameters())
            )
        optimizer = AdamW(
            trainable_params,
            lr=_params["lr"],
            weight_decay=_params["weight_decay"],
        )
        # ── Scheduler: Linear warmup + CosineAnnealingLR ───────
        warmup_epochs = _params["warmup_epochs"]
        total_epochs  = _params["max_epochs"]
        cosine_epochs = total_epochs - warmup_epochs

        if warmup_epochs > 0 and cosine_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-3,       # start at 0.1% of lr
                end_factor=1.0,          # reach full lr
                total_iters=warmup_epochs,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_epochs,     # anneal over remaining epochs
                eta_min=_params["lr"] * 1e-2,  # floor at 1% of peak lr
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
            logger.info("Scheduler: LinearLR warmup (%d ep) + CosineAnnealingLR (%d ep)",
                        warmup_epochs, cosine_epochs)
        elif warmup_epochs == 0:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=_params["lr"] * 1e-2,
            )
            logger.info("Scheduler: CosineAnnealingLR (no warmup, %d ep)", total_epochs)
        else:
            # edge case: warmup >= max_epochs
            scheduler = LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=total_epochs,
            )
            logger.info("Scheduler: LinearLR only (warmup=%d >= max_epochs=%d)",
                        warmup_epochs, total_epochs)

        # ── Training loop ─────────────────────────────────────
        best_val_auc = {name: 0.0 for name in models}
        best_models  = {name: None for name in models}
        best_epoch   = {name: 0 for name in models}
        patience_ctr = {name: 0 for name in models}
        all_train_losses = {name: [] for name in models}
        all_val_losses   = {name: [] for name in models}
        all_train_aucs   = {name: [] for name in models}
        all_val_aucs     = {name: [] for name in models}

        for epoch in range(1, _params["max_epochs"] + 1):
            # ── Train ──────────────────────────────────────────
            train_results = train_one_epoch(
                models, train_loader, criterion, optimizer, DEVICE, dataset,
            )

            # ── Validate ───────────────────────────────────────
            val_results = evaluate(
                models, val_loader, criterion, DEVICE, dataset,
            )

            # ── Compute patient-level val AUC per model ────────
            val_patient_aucs = {}
            for name in models:
                _, _, v_auc, v_probs, v_labels = val_results[name]
                all_train_losses[name].append(train_results[name][0])
                all_val_losses[name].append(0.0 if len(v_probs) == 0 else val_results[name][0])
                all_train_aucs[name].append(train_results[name][2])
                all_val_aucs[name].append(v_auc)

                # Patient-level AUC for validation
                if len(v_probs) > 0:
                    val_pids_list = [dataset.patches[val_idx[i]].patient_id
                                     for i in range(len(v_probs))]
                    vp_auc, _ = compute_patient_auc(v_probs, v_labels, val_pids_list)
                    val_patient_aucs[name] = vp_auc
                else:
                    val_patient_aucs[name] = 0.0

            # ── Log to ClearML ─────────────────────────────────
            for name in models:
                t_loss, t_acc, t_auc = train_results[name]
                v_loss, v_acc, v_auc = val_results[name][0], val_results[name][1], val_results[name][2]
                clearml_task.get_logger().report_scalar(
                    f"{name}/Loss", "train", value=t_loss, iteration=epoch)
                clearml_task.get_logger().report_scalar(
                    f"{name}/Loss", "val", value=v_loss, iteration=epoch)
                clearml_task.get_logger().report_scalar(
                    f"{name}/AUC", "train", value=t_auc, iteration=epoch)
                clearml_task.get_logger().report_scalar(
                    f"{name}/AUC", "val", value=v_auc, iteration=epoch)
                clearml_task.get_logger().report_scalar(
                    f"{name}/Patient_AUC", "val",
                    value=val_patient_aucs[name], iteration=epoch)

            # ── Scheduler step ────────────────────────────────
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            # ── Best model tracking per model ──────────────────
            # Selection uses PATCH-level validation AUC. The donor-level AUC
            # is computed from only ~4 validation donors, so it takes 5 discrete
            # values, saturates at 1.000 within the first epochs and freezes the
            # checkpoint on an untrained network. All *reported* metrics remain
            # donor-level; this choice affects only which epoch is kept.
            _warm = int(_params["warmup_epochs"])
            for name in models:
                v_auc = val_results[name][2]
                if epoch <= _warm:
                    # Do not select (or penalise) during LR warmup.
                    if best_models[name] is None:
                        best_models[name] = copy.deepcopy(models[name].state_dict())
                        best_epoch[name] = epoch
                    continue
                if v_auc > best_val_auc[name]:
                    best_val_auc[name] = v_auc
                    best_models[name] = copy.deepcopy(models[name].state_dict())
                    best_epoch[name] = epoch
                    patience_ctr[name] = 0
                else:
                    patience_ctr[name] += 1

            logger.info(
                "Epoch %d/%d | lr=%.2e | " % (epoch, _params["max_epochs"], current_lr) +
                " | ".join(
                    f"{n}: tr_loss={train_results[n][0]:.4f} "
                    f"va_auc={val_results[n][2]:.4f} "
                    f"vp_auc={val_patient_aucs[n]:.4f}"
                    for n in models
                )
            )

            # ── Early stopping (all models exhausted patience) ──
            if all(patience_ctr[n] >= _params["patience"] for n in models):
                for _n in models:
                    logger.info("  selected epoch for %s: %d (val patch AUC=%.4f)",
                                _n, best_epoch[_n], best_val_auc[_n])
                logger.info("Early stopping: all models exceeded patience=%d",
                            _params["patience"])
                break

        # ── Load best models ───────────────────────────────────
        for name in models:
            if best_models[name] is not None:
                models[name].load_state_dict(best_models[name])

        # ── Upload training curves per model ───────────────────
        for name in models:
            tc_png = plot_training_curves(
                all_train_losses[name], all_val_losses[name],
                all_train_aucs[name], all_val_aucs[name],
                f"{name}_{fold_name}",
            )
            tc_key = minio_path(name, region, "figures", f"training_curves_{name}_{fold_name}.png")
            upload_bytes_to_minio(tc_png, tc_key, fs)
            _report_image_to_clearml(
                clearml_task, f"{name}/Training", f"curves_{fold_name}", tc_png,
                iteration=fold_idx,
            )

        # ── Upload best model weights per model ────────────────
        for name in models:
            if best_models[name] is not None:
                buf = io.BytesIO()
                torch.save(best_models[name], buf)
                buf.seek(0)
                ckpt_key = minio_path(name, region, "checkpoints",
                                      f"best_model_{name}_fold_{fold_idx}.pth")
                upload_bytes_to_minio(buf.read(), ckpt_key, fs)

        # ── Test evaluation ────────────────────────────────────
        test_results = evaluate(models, test_loader, criterion, DEVICE, dataset)

        for name in models:
            t_loss, t_acc, t_auc, t_probs, t_labels = test_results[name]

            # Patient-level aggregation on test set
            test_pids_list = [dataset.patches[test_idx[i]].patient_id
                              for i in range(len(t_probs))]
            patient_auc_val, patient_details = compute_patient_auc(
                t_probs, t_labels, test_pids_list,
            )

            # ── ROC curve ──────────────────────────────────────
            if len(t_probs) > 0 and len(set(t_labels)) > 1:
                roc_png = plot_roc_curve(t_labels, t_probs, f"{name}_{fold_name}")
                roc_key = minio_path(name, region, "figures",
                                     f"roc_curve_{name}_{fold_name}.png")
                upload_bytes_to_minio(roc_png, roc_key, fs)
                _report_image_to_clearml(
                    clearml_task, f"{name}/ROC", f"fold_{fold_idx}", roc_png,
                )
                all_results[name]["fold_roc_data"].append({
                    "fold_name": fold_name,
                    "labels": t_labels,
                    "probs": t_probs,
                })

            # ── Confusion matrix ───────────────────────────────
            if len(t_probs) > 0:
                preds = (np.array(t_probs) >= 0.5).astype(int).tolist()
                cm_png = plot_confusion_matrix(t_labels, preds, f"{name}_{fold_name}")
                cm_key = minio_path(name, region, "figures",
                                    f"confusion_matrix_{name}_{fold_name}.png")
                upload_bytes_to_minio(cm_png, cm_key, fs)
                _report_image_to_clearml(
                    clearml_task, f"{name}/CM", f"fold_{fold_idx}", cm_png,
                )

            # ── Golden CSV rows (patch-level) ──────────────────
            for i in range(len(t_probs)):
                p = dataset.patches[test_idx[i]]
                all_results[name]["golden_rows"].append({
                    "Patient_ID": p.patient_id,
                    "Region": p.region,
                    "Zarr_Path": p.zarr_path,
                    "Y_start": p.y_start,
                    "X_start": p.x_start,
                    "True_Label": p.label,
                    "Pred_Prob_PD": t_probs[i],
                    "Fold": fold_idx,
                })

            # ── Patient-level rows ─────────────────────────────
            for pid, details in patient_details.items():
                # Use actual patch region (not _params), to detect
                # any region leakage bugs in the data
                actual_region = dataset.patches[test_idx[0]].region if len(test_idx) > 0 else region
                all_results[name]["patient_rows"].append({
                    "Patient_ID": pid,
                    "Region": actual_region,
                    "True_Label": details["label"],
                    "Pred_Prob_PD": details["prob"],
                    "Pred_Label": details["pred"],
                    "Patient_AUC": patient_auc_val,
                    "Fold": fold_idx,
                    "Model": name,
                })

            # ── Fold summary ───────────────────────────────────
            fold_summary = {
                "fold": fold_idx,
                "model": name,
                "region": region,
                "test_patch_auc": t_auc,
                "test_patient_auc": patient_auc_val,
                "test_loss": t_loss,
                "test_acc": t_acc,
                "n_test_patches": len(t_probs),
                "n_test_patients": len(patient_details),
            }
            all_results[name]["fold_summaries"].append(fold_summary)

            # ── Upload fold predictions immediately ────────────
            if len(all_results[name]["golden_rows"]) > 0:
                df_golden = pd.DataFrame(all_results[name]["golden_rows"])
                golden_key = minio_path(name, region, "predictions",
                                        f"golden_predictions_{name}.csv")
                upload_df_to_minio(df_golden, golden_key, fs)

            if len(all_results[name]["patient_rows"]) > 0:
                df_patient = pd.DataFrame(all_results[name]["patient_rows"])
                patient_key = minio_path(name, region, "predictions",
                                         f"patient_predictions_{name}.csv")
                upload_df_to_minio(df_patient, patient_key, fs)

            # ── Log to ClearML ─────────────────────────────────
            logger.info(
                "TEST %s %s -- loss=%.4f patch_auc=%.4f "
                "patient_auc=%.4f n_patients=%d",
                name, fold_name, t_loss, t_auc, patient_auc_val,
                len(patient_details),
            )
            clearml_task.get_logger().report_scalar(
                f"{name}/Test_Patient_AUC", "fold",
                value=patient_auc_val, iteration=fold_idx,
            )

        # ── Free GPU memory ────────────────────────────────────
        del models, optimizer, scheduler, criterion
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ════════════════════════════════════════════════════════════
    #  POST-PROCESSING: aggregated results per model
    # ════════════════════════════════════════════════════════════

    for name in ["A", "B", "C"]:
        r = all_results[name]

        # ── Overlay ROC curves ─────────────────────────────────
        if len(r["fold_roc_data"]) > 1:
            overlay_png = plot_roc_overlay(r["fold_roc_data"])
            overlay_key = minio_path(name, region, "figures",
                                     f"roc_overlay_{name}.png")
            upload_bytes_to_minio(overlay_png, overlay_key, fs)
            _report_image_to_clearml(
                clearml_task, f"{name}/ROC", "overlay", overlay_png,
            )

        # ── Fold summary CSV ───────────────────────────────────
        if len(r["fold_summaries"]) > 0:
            df_summary = pd.DataFrame(r["fold_summaries"])
            summary_key = minio_path(name, region, "predictions",
                                     f"fold_summary_{name}.csv")
            upload_df_to_minio(df_summary, summary_key, fs)

            # ── Compute and log mean AUC ───────────────────────
            mean_patch_auc   = df_summary["test_patch_auc"].mean()
            std_patch_auc    = df_summary["test_patch_auc"].std()
            mean_patient_auc = df_summary["test_patient_auc"].mean()
            std_patient_auc  = df_summary["test_patient_auc"].std()

            logger.info(
                "%s SUMMARY: patch_auc=%.3f +/- %.3f, "
                "patient_auc=%.3f +/- %.3f (%d folds)",
                name, mean_patch_auc, std_patch_auc,
                mean_patient_auc, std_patient_auc, len(r["fold_summaries"]),
            )
            clearml_task.get_logger().report_scalar(
                f"{name}/Summary", "mean_patch_auc", value=mean_patch_auc, iteration=0)
            clearml_task.get_logger().report_scalar(
                f"{name}/Summary", "mean_patient_auc", value=mean_patient_auc, iteration=0)

    logger.info("Training complete for region=%s", region)


# ==================================================================
#  9.  Entry point
# ==================================================================

def main() -> None:
    regions = [r.strip() for r in str(_params["regions"]).split(",") if r.strip()]
    seeds = [int(x) for x in str(_params["seeds"]).split(",") if str(x).strip()]
    logger.info("=" * 70)
    logger.info("V3 SINGLE TASK: regions=%s x seeds=%s  (%d training runs)",
                regions, seeds, len(regions) * len(seeds))
    logger.info("Data are downloaded, preloaded and pre-scanned ONCE per region.")
    logger.info("=" * 70)

    for _region_i, region in enumerate(regions, 1):
        logger.info("#" * 70)
        logger.info("# REGION %d/%d: %s", _region_i, len(regions), region)
        logger.info("#" * 70)
        _params["region"] = region
        _run_region(region, seeds)
        gc.collect()

    logger.info("ALL REGIONS AND SEEDS COMPLETE")


def _run_region(region: str, seeds: list) -> None:
    seed_everything(seeds[0])
    logger.info("Preparing data for region=%s (once for %d seeds)", region, len(seeds))

    # ── Download patch index from MinIO ────────────────────────
    if not os.path.exists(LOCAL_INDEX_PATH):
        logger.info("Downloading patch index from MinIO …")
        try:
            fs_tmp = _make_s3fs(MINIO_ENDPOINT, MINIO_ACCESS, MINIO_SECRET)
            if fs_tmp.exists(PATCH_INDEX_MINIO):
                fs_tmp.get(PATCH_INDEX_MINIO, LOCAL_INDEX_PATH)
                logger.info("Patch index downloaded to %s", LOCAL_INDEX_PATH)
            else:
                logger.warning(
                    "Patch index not found at %s — will scan MinIO.",
                    PATCH_INDEX_MINIO,
                )
        except Exception as exc:
            logger.warning("Failed to download index: %s", exc)

    # ── Create dataset (model_type="C" loads both channels) ────
    dataset = ZarrPatchDataset(
        minio_endpoint=MINIO_ENDPOINT,
        minio_access_key=MINIO_ACCESS,
        minio_secret_key=MINIO_SECRET,
        bucket_name=MINIO_BUCKET,
        base_folder=MINIO_BASE,
        model_type="C",                   # load BOTH channels
        patch_size=_params["patch_size"],
        target_z=_params["target_z"],
        stride=_params["stride"],
        fov_size=_params["fov_size"],
        regions=[region],                  # filter to one region
        groups=["HC", "PD"],              # ALL patients
        filter_empty=_params["filter_empty"],
        index_path=LOCAL_INDEX_PATH,
        local_cache_dir=_params["local_cache_dir"],
        preload_to_ram=_params["preload_to_ram"],
        normalization=_params["normalization"],
        augment=False,
    )

    logger.info("Dataset: %d patches from %d patients (region=%s)",
                len(dataset.patches),
                len(set(p.patient_id for p in dataset.patches)),
                region)

    # ── Preload data to RAM ────────────────────────────────────
    dataset.preload_data(
        max_workers=_params["max_workers"],
        max_ram_gb=_params["max_ram_gb"],
    )

    # ── Remove outlier patients BEFORE pre-scan ───────────────
    # CRITICAL: outlier patients must be removed before computing
    # intensity statistics, otherwise the target stats are biased
    # by the outlier patients and the normalizer is calibrated to
    # a corrupted target.
    n_before = len(dataset.patches)
    dataset.patches = [
        p for p in dataset.patches
        if p.patient_id not in EXCLUDED_PATIENTS
    ]
    n_after = len(dataset.patches)
    removed = n_before - n_after
    if removed > 0:
        logger.info(
            "Excluded %d outlier patients %s BEFORE pre-scan: "
            "%d -> %d patches (-%d)",
            len(EXCLUDED_PATIENTS), sorted(EXCLUDED_PATIENTS),
            n_before, n_after, removed,
        )
    else:
        logger.warning("No patches removed -- outlier patients not found!")

    # ── Intensity normalization pre-scan ───────────────────────
    # Now runs on CLEAN data (without outliers)
    logger.info("Running intensity pre-scan ...")
    patient_stats, target_stats, spearman_results = run_full_pre_scan(
        dataset,
        alpha=0.05,
        max_patches_per_patient=500,
        seed=_params["seed"],
    )

    # Per-donor statistics are computed from each donor's own patches and
    # contain no label information; they are reused across folds. The cohort
    # TARGET is NOT used globally here -- run_training() re-estimates it from
    # each outer fold's training donors (see the fold loop).
    global _GLOBAL_PATIENT_STATS
    _GLOBAL_PATIENT_STATS = patient_stats

    # ── Create and set IntensityNormalizer ─────────────────────
    normalizer = IntensityNormalizer(
        patient_stats=patient_stats,
        target_stats=target_stats,   # placeholder; overwritten per fold
        model_type="C",  # both channels
        use_patient_percentiles=True,
        apply_cross_patient_alignment=True,
        force_alignment=True,
    )
    dataset._normalizer = normalizer

    # ── Save pre-scan stats to MinIO ──────────────────────────
    fs = _make_s3fs(MINIO_ENDPOINT, MINIO_ACCESS, MINIO_SECRET)

    stats_json_path = "/tmp/intensity_stats.json"
    save_stats_to_json(patient_stats, target_stats, spearman_results, stats_json_path)
    stats_key = f"{MINIO_RESULTS_ROOT}_v2/_system/intensity_stats_{region}.json"
    upload_to_minio(stats_json_path, stats_key, fs)

    # ── Log Spearman results ───────────────────────────────────
    for sr in spearman_results:
        logger.info("Spearman [%s]: rho=%.4f p=%.4f %s",
                     sr.channel_name, sr.rho, sr.p_value,
                     "BIAS!" if sr.significant else "OK")

    # ── Run training: one pass per seed, same in-RAM dataset ───
    for _i, _seed in enumerate(seeds, 1):
        logger.info("=" * 70)
        logger.info("REGION %s | SEED %d  (run %d/%d)", region, _seed, _i, len(seeds))
        logger.info("Reusing the preloaded dataset and per-donor intensity stats.")
        logger.info("=" * 70)
        _params["seed"] = _seed
        seed_everything(_seed)
        try:
            run_training(dataset, task)
        except Exception:
            logger.exception("Run failed for region=%s seed=%d -- continuing",
                             region, _seed)
        gc.collect()

    # Free RAM before the next region is preloaded.
    try:
        dataset._raw_cache.clear()
        dataset._zarr_cache.clear()
        dataset._is_preloaded = False
        logger.info("Freed RAM cache for region %s", region)
    except Exception as _exc:
        logger.warning("Could not clear RAM cache: %s", _exc)
    del dataset
    gc.collect()
    logger.info("Region %s complete", region)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
