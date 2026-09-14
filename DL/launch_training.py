"""Launch the revision (v2) training runs to the ClearML queue.

Same bundling mechanism as abc_launch_experiments.py, but:
  * entry point is train_unified_5fold_v2.py (leakage fix + donor-level
    model selection + group-stratified shuffled CV)
  * one task per (region, seed) so repeated cross-validation is covered
  * results land under YOUR_OUTPUT_PREFIX/seed{N}/... on MinIO

Usage:
    python v4_launch_experiments.py            # enqueue everything
    python v4_launch_experiments.py --dry-run  # build+check bundle only
"""

import base64
import os
import shutil
import sys

_UNICODE_REPLACEMENTS = {
    "═": "=",    "─": "-",    "—": "--",
    "–": "-",    "→": "->",   "←": "<-",
    "▶": ">",    "±": "+/-",  "×": "x",
    "≤": "<=",   "≥": ">=",   "≈": "~",
    "★": "*",    "✅": "[OK]", "…": "...",
    "²": "^2",
}


def _ascii_clean(content: str) -> str:
    for old, new in _UNICODE_REPLACEMENTS.items():
        content = content.replace(old, new)
    return content


HELPER_MODULES = [
    "zarr_patch_dataset.py",
    "intensity_normalization.py",
]

_IMPORTS_MARKER = "#  1.  Imports"

_BUNDLE_HEADER = '''\
# ==============================================================
#  AUTO-GENERATED: write helper modules to disk for imports
#  (embedded by v4_launch_experiments.py for ClearML remote execution)
# ==============================================================
import base64 as _b64
import os as _os
import sys as _sys

_HELPER_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_helpers")
_os.makedirs(_HELPER_DIR, exist_ok=True)

_EMBEDDED_MODULES = {{
{modules_dict}
}}

for _mod_name, _mod_b64 in _EMBEDDED_MODULES.items():
    _mod_path = _os.path.join(_HELPER_DIR, _mod_name)
    with open(_mod_path, "wb") as _f:
        _f.write(_b64.b64decode(_mod_b64))

_sys.path.insert(0, _HELPER_DIR)
# ===== END AUTO-GENERATED BLOCK =====

'''


def _create_bundle(train_script_path: str) -> str:
    with open(train_script_path, "r", encoding="utf-8") as f:
        main_content = _ascii_clean(f.read())

    script_dir = os.path.dirname(os.path.abspath(train_script_path))
    modules_lines = []
    for mod_name in HELPER_MODULES:
        mod_path = os.path.join(script_dir, mod_name)
        if not os.path.isfile(mod_path):
            print("  WARNING: %s not found at %s" % (mod_name, mod_path))
            continue
        with open(mod_path, "r", encoding="utf-8") as f:
            mod_content = _ascii_clean(f.read())
        b64 = base64.b64encode(mod_content.encode("ascii", errors="replace")).decode("ascii")
        modules_lines.append('    "%s": "%s",' % (mod_name, b64))

    header = _BUNDLE_HEADER.format(modules_dict="\n".join(modules_lines))

    idx = main_content.find(_IMPORTS_MARKER)
    if idx == -1:
        raise ValueError("Could not find marker '%s' in %s." % (_IMPORTS_MARKER, train_script_path))
    line_start = main_content.rfind("\n", 0, idx) + 1
    bundle_content = main_content[:line_start] + header + main_content[line_start:]

    bundle_path = os.path.join(script_dir, "train_unified_5fold_v4_bundle.py")
    with open(bundle_path, "w", encoding="ascii", errors="replace") as f:
        f.write(bundle_content)
    return bundle_path


def _cleanup_bundle(bundle_path: str) -> None:
    try:
        os.unlink(bundle_path)
        helpers_dir = os.path.join(os.path.dirname(bundle_path), "_helpers")
        if os.path.isdir(helpers_dir):
            shutil.rmtree(helpers_dir, ignore_errors=True)
    except OSError:
        pass


# ==============================================================
#  Configuration
# ==============================================================

PROJECT     = "YOUR_STORAGE_NAME/DL_Training"
BASE_SCRIPT = "train_unified_5fold_v4.py"
QUEUE       = "YOUR_QUEUE_NAME"

# Repeated group-stratified CV: 3 seeds x 2 regions = 6 tasks.
SEEDS   = [42, 43, 44]
REGIONS = ["substantiaNigra", "putamen"]

REQUIRED_PACKAGES = [
    "clearml",
    "boto3",
    "s3fs",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "pandas",
    "seaborn",
    "torch",
    "torchvision",
    "zarr<3",
]

COMMON_PARAMS = {
    "patch_size":   256,
    "target_z":     25,
    "stride":       236,
    "fov_size":     1200,
    "filter_empty": True,
    "normalization": "intensity_pipeline",
    "batch_size":   16,
    "max_epochs":   15,
    "lr":           1e-4,
    "weight_decay": 1e-2,
    "patience":     5,
    "warmup_epochs": 3,
    "dropout":      0.5,
    "label_smoothing": 0.0,
    "num_workers":  8,
    "local_cache_dir":  "/tmp/zarr_cache",
    "preload_to_ram":   True,
    "max_ram_gb":       64.0,
    "max_workers": 8,
    "clearml_dataset_project": "YOUR_STORAGE_NAME/DL_Training",
    "clearml_dataset_name":    "zarr_microglia_data",
    "architecture": "cnn",
}

EXPERIMENTS = [
    {
        "name": "V4_ABC_cnn_ALL",
        "queue_name": QUEUE,
        # One task. Data are downloaded, preloaded and pre-scanned once per
        # region; the seeds are looped inside the script.
        "overrides": {
            "regions": ",".join(REGIONS),
            "seeds":   ",".join(str(x) for x in SEEDS),
            "region":  REGIONS[0],   # placeholder, reset per loop at runtime
            "seed":    SEEDS[0],
        },
    },
]


def build_params(exp):
    merged = {**COMMON_PARAMS, **exp["overrides"], "queue_name": exp["queue_name"]}
    return {"General/%s" % k: str(v) for k, v in merged.items()}


def main():
    dry_run = "--dry-run" in sys.argv

    print("Creating self-contained bundle from %s ..." % BASE_SCRIPT)
    bundle_path = _create_bundle(BASE_SCRIPT)
    print("  Bundle: %s" % bundle_path)

    import py_compile
    py_compile.compile(bundle_path, doraise=True)
    print("  Syntax check: OK")

    with open(bundle_path, "rb") as f:
        raw = f.read()
    try:
        raw.decode("cp1251")
        print("  cp1251 decode: OK")
    except UnicodeDecodeError:
        print("  WARNING: bundle not cp1251-safe!")

    for marker in ("StratifiedGroupKFold", "_fold_target_stats",
                   "run_spearman_bias_test", "MINIO_RESULTS_ROOT}_v2",
                   "def _run_region", "for _i, _seed in enumerate"):
        txt = raw.decode("ascii", errors="replace")
        print("  contains %-24s : %s" % (marker, "yes" if marker in txt else "NO -- CHECK"))

    if dry_run:
        _cleanup_bundle(bundle_path)
        print("\nDry run complete, nothing enqueued.")
        return

    from clearml import Task

    enqueued = []
    try:
        for exp in EXPERIMENTS:
            print("\n" + "=" * 60)
            print("Creating task: %s" % exp["name"])
            print("=" * 60)

            task = Task.create(
                project_name=PROJECT,
                task_name=exp["name"],
                script=bundle_path,
                packages=REQUIRED_PACKAGES,
                add_task_init_call=False,
            )
            task.set_parameters(build_params(exp))
            Task.enqueue(task.id, queue_name=exp["queue_name"])

            enqueued.append((exp["queue_name"], exp["name"], task.id))
            print("  Task ID : %s" % task.id)
            print("  Queue   : %s" % exp["queue_name"])
            for k, v in exp["overrides"].items():
                print("    %-12s = %s" % (k, v))
    finally:
        _cleanup_bundle(bundle_path)
        print("\nCleaned up: %s" % bundle_path)

    print("\n" + "=" * 60)
    print("All %d experiments enqueued" % len(enqueued))
    print("=" * 60)
    for q, name, tid in enqueued:
        print("  [%s]  %-38s  (id=%s)" % (q, name, tid))
    print("\nResults will appear on MinIO under: YOUR_OUTPUT_PREFIX/seed{N}/{MODEL}_cnn/{region}/")


if __name__ == "__main__":
    main()
