from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {"name": "Cosmic_Ray_ML_Colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    }
    notebook["cells"] = [
        markdown(
            """
# Cosmic-Ray Shower Detection and Reconstruction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPOSITORY/blob/main/Cosmic_Ray_ML_Colab.ipynb)

**Portable exam notebook: Original Deep Sets plus a controlled Attention Deep Sets extension**

This notebook covers the required preprocessing, visualization, architecture, training, optimization, evaluation, and model comparison. It has two execution modes:

- `quick`: a short deterministic demonstration for Colab and presentation rehearsal;
- `full`: the complete training protocol used for final-quality results.

The frozen reference numbers shown near the end came from completed local GPU runs. A quick run is deliberately labeled as a demo and is not substituted for those final results.
"""
        ),
        markdown(
            """
## 1. Physics objective and model choice

An event is a set of 217 detector waveforms, each with 48 time bins. Signal events contain an extensive air shower; background events contain noise, isolated hits, or instrumental bursts. We predict:

1. the probability that a shower is present;
2. the supplied logarithmic shower energy for signal events;
3. the shower-core position `(x, y)` for signal events.

The default **Original Deep Sets** model uses one shared 1D CNN for every detector waveform, concatenates detector coordinates, and combines active detectors with masked mean, standard deviation, and maximum. This follows the time → detector → event hierarchy, respects arbitrary detector ordering, and remains easy to explain.

The optional **Attention Deep Sets** model changes only aggregation: it learns normalized weights for active detectors. This may improve reconstruction, but attention weights are model importance coefficients—not proof of physical causality.
"""
        ),
        markdown("## 2. Configuration\n\nEdit only this cell for a normal run. Keep `quick` and `original` for the first execution."),
        code(
            """
import os
import sys
import subprocess
from pathlib import Path

RUN_MODE = os.environ.get("COSMIC_RUN_MODE", "quick")       # "quick" or "full"
MODEL_VARIANT = os.environ.get("COSMIC_MODEL", "original") # "original" or "attention"
USE_GOOGLE_DRIVE = False
EVALUATE_TEST = False  # Set True only for the final selected full model.
REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git"
REPO_BRANCH = "main"

assert RUN_MODE in {"quick", "full"}
assert MODEL_VARIANT in {"original", "attention"}
if RUN_MODE == "quick":
    EVALUATE_TEST = False

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if "YOUR_USERNAME" in REPO_URL:
        raise ValueError("Replace YOUR_USERNAME/YOUR_REPOSITORY before uploading the notebook.")
    PROJECT_ROOT = Path("/content/cosmic-ray-ml")
    if not (PROJECT_ROOT / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(PROJECT_ROOT)], check=True)
else:
    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "cosmic_ml").exists():
        raise RuntimeError("Run this notebook from the repository root.")

if IN_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_ROOT = Path("/content/drive/MyDrive/cosmic_ray_ml")
    DEFAULT_OUTPUT_DIR = DRIVE_ROOT / "outputs" / f"{MODEL_VARIANT}_{RUN_MODE}"
    DEFAULT_ARCHIVE = DRIVE_ROOT / "cosmic_array_dataset.npz"
elif IN_COLAB:
    DRIVE_ROOT = None
    DEFAULT_OUTPUT_DIR = Path("/content/outputs") / f"{MODEL_VARIANT}_{RUN_MODE}"
    DEFAULT_ARCHIVE = Path("/content/data/cosmic_array_dataset.npz")
else:
    STORAGE_ROOT = Path(os.environ.get("COSMIC_STORAGE_ROOT", str(PROJECT_ROOT)))
    DRIVE_ROOT = None
    DEFAULT_OUTPUT_DIR = STORAGE_ROOT / "outputs" / f"{MODEL_VARIANT}_{RUN_MODE}"
    DEFAULT_ARCHIVE = STORAGE_ROOT / "data" / "cosmic_array_dataset.npz"

default_data = Path("/content/data") if IN_COLAB else STORAGE_ROOT / "data"
DATA_DIR = Path(os.environ.get("COSMIC_DATA_DIR", str(default_data)))
OUTPUT_DIR = Path(os.environ.get("COSMIC_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
default_archive = DEFAULT_ARCHIVE if DRIVE_ROOT is not None else DATA_DIR / "cosmic_array_dataset.npz"
ARCHIVE_PATH = Path(os.environ.get("COSMIC_ARCHIVE_PATH", str(default_archive)))
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Mode: {RUN_MODE}; model: {MODEL_VARIANT}; Colab: {IN_COLAB}")
print(f"Project: {PROJECT_ROOT}")
print(f"Data: {DATA_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Compressed archive/cache: {ARCHIVE_PATH}")
"""
        ),
        markdown("## 3. Environment and GPU check\n\nColab users must select **Runtime → Change runtime type → GPU** before continuing."),
        code(
            """
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)

import json
import platform
import numpy as np
import torch

environment = {
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
}
print(json.dumps(environment, indent=2))
if IN_COLAB and not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. Select Runtime > Change runtime type > GPU and reconnect.")
"""
        ),
        markdown(
            """
## 4. Download and prepare the dataset

The official compressed dataset is about 1.8 GB and expands to roughly 9.7 GB. The download script checks the exact expected byte count. Preparation extracts memory-mappable arrays, makes one deterministic seed-42 stratified 70/15/15 split, and computes every normalization statistic from training data only.

The waveform transform is `sign(x) × log(1 + |x|)`, which compresses the long positive tail without discarding negative electronic noise. Inactive detectors are explicitly masked.
"""
        ),
        code(
            """
archive = ARCHIVE_PATH
expected_bytes = 1_908_350_470
if not archive.exists() or archive.stat().st_size != expected_bytes:
    subprocess.run(["bash", "scripts/download_data.sh", str(archive)], check=True)
else:
    print(f"Using existing verified archive: {archive}")

local_archive = DATA_DIR / "cosmic_array_dataset.npz"
if archive.resolve() != local_archive.resolve():
    if not local_archive.exists():
        local_archive.symlink_to(archive)
        print(f"Linked runtime archive {local_archive} -> {archive}")
    elif local_archive.stat().st_size != expected_bytes:
        raise RuntimeError(f"Unexpected existing runtime archive: {local_archive}")

print("Extracted arrays stay on the fast runtime disk; Drive caches only the archive and outputs.")

subprocess.run([sys.executable, "-m", "cosmic_ml.prepare_data", "--data-dir", str(DATA_DIR)], check=True)
stats = json.loads((DATA_DIR / "processed" / "stats.json").read_text())
print(json.dumps(stats, indent=2))
"""
        ),
        markdown("## 5. Inspect geometry and one normalized waveform\n\nThese plots connect the tensor representation to the physical detector layout and time response."),
        code(
            """
import matplotlib.pyplot as plt
from cosmic_ml.data import CosmicDataset, load_splits

positions = np.load(DATA_DIR / "extracted" / "positions.npy")
splits = load_splits(DATA_DIR)
sample = CosmicDataset(splits["train"][:1], DATA_DIR, stats)[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(positions[:, 0], positions[:, 1], c=sample["mask"].numpy(), s=22, cmap="viridis")
axes[0].set(title="Detector geometry", xlabel="x", ylabel="y", aspect="equal")
active_index = int(torch.where(sample["mask"] > 0)[0][0])
axes[1].plot(sample["waveform"][active_index].numpy())
axes[1].set(title=f"Normalized waveform: detector {active_index}", xlabel="Time bin", ylabel="Normalized signal")
fig.tight_layout()
plt.show()
print(f"Tensor shape presented to the model: {tuple(sample['waveform'].shape)} = (detectors, time bins)")
"""
        ),
        markdown(
            """
## 6. Architecture check

Both variants share the temporal CNN, detector-coordinate encoder, event network, and three prediction heads. The only experimental change is equal-weight versus learned-weight aggregation.
"""
        ),
        code(
            """
from cosmic_ml.data import normalized_positions
from cosmic_ml.model import build_model, model_config

position_tensor = torch.from_numpy(normalized_positions(DATA_DIR, stats))
model = build_model(MODEL_VARIANT, position_tensor)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(model)
print("Configuration:", model_config(model))
print(f"Trainable parameters: {parameter_count:,}")
"""
        ),
        markdown(
            """
## 7. Training and validation-controlled selection

The loss is binary cross-entropy plus signal-only SmoothL1 energy and core losses. Background has no physical energy/core target, so those two terms are masked. AdamW, learning-rate reduction, gradient clipping, and early stopping are used. The best checkpoint is selected by total validation loss.
"""
        ),
        code(
            """
smoke = os.environ.get("COSMIC_SMOKE", "0") == "1"
if smoke:
    epochs, max_train, max_val = 1, 128, 64
elif RUN_MODE == "quick":
    epochs, max_train, max_val = 5, 5000, 1000
else:
    epochs, max_train, max_val = 40, None, None

command = [
    sys.executable, "-m", "cosmic_ml.train",
    "--model-type", MODEL_VARIANT,
    "--data-dir", str(DATA_DIR),
    "--output-dir", str(OUTPUT_DIR),
    "--epochs", str(epochs),
    "--batch-size", "64",
    "--num-workers", "2" if IN_COLAB else "0",
    "--device", "cuda" if torch.cuda.is_available() else "cpu",
]
if max_train is not None:
    command += ["--max-train-samples", str(max_train), "--max-val-samples", str(max_val)]

print("Running:", " ".join(command))
subprocess.run(command, check=True)
"""
        ),
        markdown(
            """
## 8. Evaluation

Quick mode evaluates a deterministic validation subset. Full mode still defaults to validation; set `EVALUATE_TEST = True` only after the final model is chosen. Lower energy/core errors and Brier score are better; higher ROC-AUC, PR-AUC, and accuracy are better.
"""
        ),
        code(
            """
evaluation_split = "test" if RUN_MODE == "full" and EVALUATE_TEST else "val"
evaluation_dir = OUTPUT_DIR / f"evaluation_{evaluation_split}"
command = [
    sys.executable, "-m", "cosmic_ml.evaluate",
    "--checkpoint", str(OUTPUT_DIR / "best.pt"),
    "--data-dir", str(DATA_DIR),
    "--split", evaluation_split,
    "--batch-size", "128",
    "--num-workers", "2" if IN_COLAB else "0",
    "--device", "cuda" if torch.cuda.is_available() else "cpu",
    "--output-dir", str(evaluation_dir),
]
if RUN_MODE == "quick" or smoke:
    command += ["--max-samples", "64" if smoke else "1000"]
subprocess.run(command, check=True)

metrics = json.loads((evaluation_dir / "metrics.json").read_text())
print(f"{MODEL_VARIANT.title()} {RUN_MODE} result on {evaluation_split}:")
print(json.dumps(metrics, indent=2))

from IPython.display import display, Image
display(Image(filename=str(evaluation_dir / "classification_curves.png")))
display(Image(filename=str(evaluation_dir / "reconstruction.png")))
"""
        ),
        markdown(
            """
## 9. Frozen full-run comparison

The following values are loaded from the checked-in provenance file. They are from the completed 7,500-event local test evaluations, not from the quick notebook run. This distinction prevents a small demonstration from being mistaken for the final experiment.
"""
        ),
        code(
            """
reference = json.loads((PROJECT_ROOT / "results" / "reference_metrics.json").read_text())
metrics_to_show = [
    "classification/roc_auc",
    "classification/accuracy_at_0.5",
    "classification/brier",
    "energy/mae",
    "energy/rmse",
    "core/median_distance",
    "core/p68_distance",
]
header = "| Metric | Original | Attention |\\n|---|---:|---:|\\n"
rows = "".join(
    f"| {name} | {reference['original'][name]:.6f} | {reference['attention'][name]:.6f} |\\n"
    for name in metrics_to_show
)
from IPython.display import Markdown
display(Markdown(header + rows))
print(reference["provenance"])
"""
        ),
        markdown(
            """
## 10. Interpretation and conclusion

- Original Deep Sets is the recommended main presentation model: it is compact, strong, permutation invariant, and easy to connect to the detector hierarchy.
- Attention adds only a small scoring network and improves accuracy, Brier score, energy MAE, and core-error summaries in the completed comparison.
- Attention slightly worsens ROC-AUC, PR-AUC, and energy RMSE, so the conclusion is a reconstruction trade-off rather than a universal victory.
- Local GPU execution is the canonical full experiment because storage and hardware persist. Colab is the most transferable demo and can attempt a full run, but its GPU type, availability, and runtime lifetime are not guaranteed.
- Reproducibility includes the split seed, training-only normalization, explicit signal masking, saved runtime versions, separate output directories, and provenance-labeled reference metrics.
"""
        ),
    ]
    output = ROOT / "Cosmic_Ray_ML_Colab.ipynb"
    nbf.write(notebook, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
