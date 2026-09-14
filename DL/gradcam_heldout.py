"""
Held-out Grad-CAM figures in the ORIGINAL visual style.
=======================================================

Visual style is ported verbatim from train_final_and_gradcam.py (lines 183-265):
same 2x3 peak-Z composite, same Z-slice strip, same colormaps (Greens / Reds /
Greys_r + jet alpha=0.5), same green AOR contour, same activation profile, same
bold suptitle.

Only the data path differs, and only where the reviewers required it:

  * inputs are UNMODIFIED (no channel is zeroed), so the network sees exactly
    what it saw at test time                                       [R1.3, R2.3]
  * patches come from HELD-OUT donors only: for every outer fold the fold's own
    checkpoint is applied to that fold's test donors                     [R1.3]
  * normalisation is re-estimated from that fold's training donors alone
                                                                   [R1.4, R2.1]

The relevance VOLUME that the original figure draws is reconstructed as a
separable product: the lateral distribution comes from Grad-CAM at layer4[-1],
the axial distribution from |input x gradient| summed over space per input
channel (each channel is one optical section). Both come from a single forward
and backward pass on the unmodified input, so no ablated tensor is ever built.
"""

from __future__ import annotations
import os

os.environ["AWS_ACCESS_KEY_ID"] = "YOUR_MINIO_ACCESS_KEY"
os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR_MINIO_SECRET_KEY"
os.environ["AWS_ENDPOINT_URL"] = "YOUR_MINIO_ENDPOINT"
os.environ["CLEARML_AGENT_BOTO3_ENDPOINT_URL"] = "YOUR_MINIO_ENDPOINT"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

from clearml import Task

_params = dict(
    region="substantiaNigra",
    seed=42,
    models="A,B,C",
    n_donors_per_group=3,
    patch_size=256,
    target_z=25,
    stride=236,
    fov_size=1200,
    filter_empty=True,
    normalization="intensity_pipeline",
    batch_size=8,
    num_workers=4,
    dropout=0.5,
    local_cache_dir="/tmp/zarr_cache",
    preload_to_ram=True,
    max_ram_gb=64.0,
    max_workers=8,
    clearml_dataset_project="YOUR_STORAGE_NAME/DL_Training",
    clearml_dataset_name="zarr_microglia_data",
    architecture="cnn",
    queue_name="YOUR_QUEUE_NAME",
)

task = Task.init(project_name="YOUR_STORAGE_NAME/DL_Training",
                 task_name="XAI_v3_heldout_original_style")
task.connect(_params)
task.execute_remotely(queue_name=_params["queue_name"])

# ==================================================================
#  1.  Imports
# ==================================================================
import io
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights
from scipy import ndimage

from zarr_patch_dataset import ZarrPatchDataset, patch_collate_fn, _make_s3fs
from intensity_normalization import (
    run_full_pre_scan, compute_target_stats, IntensityNormalizer,
)

logger = logging.getLogger(__name__)

MINIO_ENDPOINT = "YOUR_MINIO_ENDPOINT"
MINIO_ACCESS = "YOUR_MINIO_ACCESS_KEY"
MINIO_SECRET = "YOUR_MINIO_SECRET_KEY"
MINIO_BUCKET = "YOUR_STORAGE_NAME"
RESULTS_ROOT = "YOUR_OUTPUT_PREFIX"
XAI_ROOT = "YOUR_XAI_PREFIX"
PATCH_INDEX_MINIO = "YOUR_STORAGE_NAME/YOUR_BASE_PREFIX/_system/patch_index_FROM_CLEARML.json"
LOCAL_INDEX_PATH = "/tmp/patch_index_FROM_CLEARML.json"
EXCLUDED_PATIENTS = {"PD8", "PD10", "HC9"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CH_IBA1 = slice(0, 25)
CH_PSYN = slice(25, 50)


def put_bytes(data, key, fs):
    uri = f"{MINIO_BUCKET}/{key}"
    try:
        with fs.open(uri, "wb") as f:
            f.write(data)
        logger.info("Uploaded -> s3://%s", uri)
    except Exception as exc:
        logger.error("Upload failed %s: %s", uri, exc)


# ==================================================================
#  2.  Model reconstruction (mirrors training)
# ==================================================================
def _smart_conv1(in_channels, w):
    avg = w.mean(dim=1, keepdim=True)
    new_w = avg.repeat(1, in_channels, 1, 1) / (in_channels / 3.0)
    conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv.weight.data = new_w
    return conv


def build_model(in_channels, dropout):
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.conv1 = _smart_conv1(in_channels, m.conv1.weight.data)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 2))
    return m


def load_fold_model(seed, model_name, region, fold, fs):
    in_ch = 50 if model_name == "C" else 25
    m = build_model(in_ch, float(_params["dropout"]))
    key = (f"{MINIO_BUCKET}/{RESULTS_ROOT}/seed{seed}/{model_name}_cnn/{region}"
           f"/checkpoints/best_model_{model_name}_fold_{fold}.pth")
    with fs.open(key, "rb") as f:
        state = torch.load(io.BytesIO(f.read()), map_location="cpu")
    m.load_state_dict(state)
    m.to(DEVICE).eval()
    return m


def slice_for(model_name, x):
    if model_name == "A":
        return x[:, CH_IBA1]
    if model_name == "B":
        return x[:, CH_PSYN]
    return x


# ==================================================================
#  3.  Saliency on UNMODIFIED inputs -> relevance volume
# ==================================================================
class Saliency:
    def __init__(self, model):
        self.model = model
        self.acts = None
        self.grads = None
        tgt = model.layer4[-1]
        self._h1 = tgt.register_forward_hook(self._fwd)
        self._h2 = tgt.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.acts = out

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def close(self):
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x, model_name):
        """Returns (cam_3d [Z,H,W] in 0..1, prob_pd float)."""
        x = x.clone().detach().requires_grad_(True)
        logits = self.model(x)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        self.model.zero_grad(set_to_none=True)
        logits[0, 1].backward()

        # lateral: Grad-CAM at layer4[-1]
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts.detach()).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam2d = cam.detach().cpu().numpy()

        # axial: |input x gradient| per input channel, one channel per section
        z_rel = (x.grad * x).abs().sum(dim=(2, 3))[0].detach().cpu().numpy()
        if model_name == "C":
            z_rel = z_rel[:25] + z_rel[25:]
        z_rel = z_rel - z_rel.min()
        z_rel = z_rel / (z_rel.max() + 1e-8)

        cam_3d = cam2d[None, :, :] * z_rel[:, None, None]
        cam_3d = cam_3d / (cam_3d.max() + 1e-8)
        return cam_3d, prob


# ==================================================================
#  4.  Plotting -- PORTED VERBATIM from train_final_and_gradcam.py
# ==================================================================
def plot_gradcam_3d(raw_patch, cam_3d, patient_id, label, model_name, region):
    """Identical layout, colormaps and titles to the original implementation."""
    group_name = "PD" if label == 1 else "HC"
    iba1 = raw_patch[0]
    psyn = raw_patch[1]
    Z = iba1.shape[0]

    threshold_3d = np.percentile(cam_3d[cam_3d > 0], 80) if cam_3d.max() > 0 else 0.9
    aor_3d_mask = (cam_3d >= threshold_3d).astype(np.float32)
    aor_per_slice = aor_3d_mask.sum(axis=(1, 2))
    peak_z = int(np.argmax(aor_per_slice))

    # ── FIGURE 1: Peak Z-slice + 3D AOR contour ─────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)

    axes[0, 0].imshow(iba1[peak_z], cmap='Greens')
    axes[0, 0].set_title(f"IBA1 at Peak Z={peak_z}", fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(psyn[peak_z], cmap='Reds')
    axes[0, 1].set_title(f"pSyn at Peak Z={peak_z}", fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(psyn[peak_z], cmap='Greys_r')
    axes[0, 2].imshow(cam_3d[peak_z], cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[0, 2].set_title(f"Grad-CAM at Peak Z={peak_z}", fontsize=12, fontweight="bold")
    axes[0, 2].axis('off')

    aor_peak_2d = aor_3d_mask[peak_z]
    aor_dilated = ndimage.binary_dilation(aor_peak_2d > 0.5, iterations=2)
    contour = aor_dilated ^ (aor_peak_2d > 0.5)

    axes[1, 0].imshow(psyn[peak_z], cmap='Reds')
    contour_overlay = np.zeros((psyn.shape[1], psyn.shape[2], 4))
    contour_overlay[contour, 1] = 1.0
    contour_overlay[contour, 3] = 1.0
    axes[1, 0].imshow(contour_overlay)
    axes[1, 0].set_title(f"3D AOR contour on pSyn (Z={peak_z})", fontsize=12, fontweight="bold")
    axes[1, 0].axis('off')

    psyn_mip = psyn.max(axis=0)
    cam_mip = cam_3d.max(axis=0)
    axes[1, 1].imshow(psyn_mip, cmap='Greys_r')
    axes[1, 1].imshow(cam_mip, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[1, 1].set_title("Classic MIP Grad-CAM", fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].plot(range(Z), cam_3d.reshape(Z, -1).mean(axis=1), color='blue', label='Mean CAM')
    axes[1, 2].plot(range(Z), aor_per_slice / max(aor_per_slice.max(), 1), color='red', label='AOR Area (norm)')
    axes[1, 2].axvline(x=peak_z, color='green', linestyle='--', label=f'Peak Z={peak_z}')
    axes[1, 2].set_title("Activation profile across Z-depth", fontsize=12)
    axes[1, 2].set_xlabel("Z-slice")
    axes[1, 2].legend()

    fig.suptitle(f"Model {model_name} | {group_name} Patient: {patient_id} | Region: {region}",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()

    buf1 = io.BytesIO()
    fig.savefig(buf1, format="png", bbox_inches="tight")
    plt.close(fig)
    buf1.seek(0)

    # ── FIGURE 2: Per-slice Grad-CAM grid ───────────
    slice_indices = list(range(0, Z, max(1, Z // 8)))[:8]
    n_slices = len(slice_indices)

    fig, axes = plt.subplots(3, n_slices, figsize=(4 * n_slices, 12), dpi=150)

    for col, z_idx in enumerate(slice_indices):
        axes[0, col].imshow(psyn[z_idx], cmap='Reds', vmin=0, vmax=psyn.max() + 1e-8)
        axes[0, col].set_title(f"pSyn Z={z_idx}", fontsize=10)
        axes[0, col].axis('off')

        axes[1, col].imshow(psyn[z_idx], cmap='Greys_r')
        axes[1, col].imshow(cam_3d[z_idx], cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[1, col].set_title(f"Grad-CAM Z={z_idx}", fontsize=10)
        axes[1, col].axis('off')

        aor_z = aor_3d_mask[z_idx]
        aor_dil_z = ndimage.binary_dilation(aor_z > 0.5, iterations=1)
        contour_z = aor_dil_z ^ (aor_z > 0.5)

        axes[2, col].imshow(psyn[z_idx], cmap='Reds')
        contour_z_vis = np.zeros((psyn.shape[1], psyn.shape[2], 4))
        contour_z_vis[contour_z, 1] = 1.0
        contour_z_vis[contour_z, 3] = 1.0
        axes[2, col].imshow(contour_z_vis)
        axes[2, col].set_title(f"3D AOR Z={z_idx}", fontsize=10)
        axes[2, col].axis('off')

    axes[0, 0].set_ylabel("pSyn\n(Lewy bodies)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Per-slice\nGrad-CAM", fontsize=12, fontweight="bold")
    axes[2, 0].set_ylabel("3D AOR\nContour", fontsize=12, fontweight="bold")

    fig.suptitle(f"Z-Slices XAI | {group_name} Patient: {patient_id}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    buf2 = io.BytesIO()
    fig.savefig(buf2, format="png", bbox_inches="tight")
    plt.close(fig)
    buf2.seek(0)

    return buf1.read(), buf2.read()


# ==================================================================
#  5.  Main
# ==================================================================
def build_dataset(region):
    if not os.path.exists(LOCAL_INDEX_PATH):
        fs_tmp = _make_s3fs(MINIO_ENDPOINT, MINIO_ACCESS, MINIO_SECRET)
        if fs_tmp.exists(PATCH_INDEX_MINIO):
            fs_tmp.get(PATCH_INDEX_MINIO, LOCAL_INDEX_PATH)

    ds = ZarrPatchDataset(
        minio_endpoint=MINIO_ENDPOINT, minio_access_key=MINIO_ACCESS,
        minio_secret_key=MINIO_SECRET, bucket_name=MINIO_BUCKET,
        base_folder="YOUR_BASE_PREFIX", model_type="C",
        patch_size=_params["patch_size"], target_z=_params["target_z"],
        stride=_params["stride"], fov_size=_params["fov_size"],
        regions=[region], groups=["HC", "PD"],
        filter_empty=_params["filter_empty"], index_path=LOCAL_INDEX_PATH,
        local_cache_dir=_params["local_cache_dir"],
        preload_to_ram=_params["preload_to_ram"],
        normalization=_params["normalization"], augment=False,
    )
    ds.preload_data(max_workers=_params["max_workers"],
                    max_ram_gb=_params["max_ram_gb"])
    ds.patches = [p for p in ds.patches if p.patient_id not in EXCLUDED_PATIENTS]
    logger.info("Dataset: %d patches, %d donors", len(ds.patches),
                len(set(p.patient_id for p in ds.patches)))
    patient_stats, _, _ = run_full_pre_scan(ds, alpha=0.05,
                                            max_patches_per_patient=500, seed=42)
    return ds, patient_stats


def main():
    region = _params["region"]
    seed = int(_params["seed"])
    models = [m.strip() for m in str(_params["models"]).split(",") if m.strip()]
    n_per_group = int(_params["n_donors_per_group"])
    fs = _make_s3fs(MINIO_ENDPOINT, MINIO_ACCESS, MINIO_SECRET)

    ds, patient_stats = build_dataset(region)
    pid_arr = np.array([p.patient_id for p in ds.patches])

    for model_name in models:
        logger.info("=" * 70)
        logger.info("XAI v3 | model %s | region %s | seed %d", model_name, region, seed)
        logger.info("=" * 70)

        key = (f"{MINIO_BUCKET}/{RESULTS_ROOT}/seed{seed}/{model_name}_cnn/"
               f"{region}/predictions/patient_predictions_{model_name}.csv")
        with fs.open(key, "rb") as f:
            preds = pd.read_csv(io.BytesIO(f.read()))

        done = {"PD": 0, "HC": 0}

        for fold in sorted(preds.Fold.unique()):
            fold_rows = preds[preds.Fold == fold]
            test_pids = set(fold_rows.Patient_ID)
            train_pids = set(preds.Patient_ID) - test_pids

            train_stats = {k: v for k, v in patient_stats.items() if k in train_pids}
            ds._normalizer = IntensityNormalizer(
                patient_stats=patient_stats,
                target_stats=compute_target_stats(train_stats),
                model_type="C", use_patient_percentiles=True,
                apply_cross_patient_alignment=True, force_alignment=True)

            model = load_fold_model(seed, model_name, region, int(fold), fs)
            sal = Saliency(model)

            for _, row in fold_rows.iterrows():
                grp = "PD" if row.True_Label == 1 else "HC"
                if done[grp] >= n_per_group:
                    continue
                donor = row.Patient_ID
                idx = np.where(pid_arr == donor)[0]
                if len(idx) == 0:
                    continue

                # choose the patch with the strongest predicted PD evidence
                loader = DataLoader(Subset(ds, idx.tolist()),
                                    batch_size=_params["batch_size"], shuffle=False,
                                    num_workers=_params["num_workers"],
                                    collate_fn=patch_collate_fn)
                best = (-1.0, None)
                with torch.no_grad():
                    for tensors, _metas in loader:
                        t = tensors.to(DEVICE)
                        B, C, Zd, H, W = t.shape
                        xf = t.reshape(B, C * Zd, H, W)
                        p = torch.softmax(model(slice_for(model_name, xf)), 1)[:, 1]
                        j = int(torch.argmax(p).item())
                        if float(p[j]) > best[0]:
                            best = (float(p[j]), xf[j:j + 1].detach().clone())
                if best[1] is None:
                    continue

                x_full = best[1]
                cam_3d, prob = sal(slice_for(model_name, x_full), model_name)
                raw = x_full[0].detach().cpu().numpy().reshape(2, 25,
                                                              x_full.shape[-2],
                                                              x_full.shape[-1])

                try:
                    png1, png2 = plot_gradcam_3d(raw, cam_3d, donor,
                                                 int(row.True_Label), model_name, region)
                    base = f"{XAI_ROOT}/seed{seed}/{model_name}_cnn/{region}"
                    put_bytes(png1, f"{base}/GradCAM3D_{model_name}_{grp}_{donor}_AOR.png", fs)
                    put_bytes(png2, f"{base}/GradCAM3D_{model_name}_{grp}_{donor}_PerSlice.png", fs)
                    done[grp] += 1
                    logger.info("  %s %s (fold %s, held-out) p(PD)=%.3f -> figures written",
                                grp, donor, fold, prob)
                except Exception:
                    logger.exception("figure failed for %s", donor)

            sal.close()
            del model
            torch.cuda.empty_cache()
            if done["PD"] >= n_per_group and done["HC"] >= n_per_group:
                break

        logger.info("model %s done: PD=%d HC=%d", model_name, done["PD"], done["HC"])

    logger.info("XAI v3 complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
