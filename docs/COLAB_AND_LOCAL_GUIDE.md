# Colab and Local GPU Reproduction Guide

## Decision

The best deliverable is a **hybrid repository**:

1. The included completed Colab full runs are the source of the final reported comparison.
2. The local GPU workflow is an equally supported route for complete independent reruns.
3. The main Colab notebook is the portable teaching/demo route and can also perform a full rerun.

This avoids choosing between reliability and accessibility. A hosted notebook is easier for the examiner to open, while a local run is safer for a 1.8 GB compressed dataset that expands to roughly 9.7 GB and must survive preprocessing and training.

## Why Original Deep Sets is the default

Each event contains 217 detector waveforms with 48 time bins. The model follows that hierarchy:

1. A shared 1D CNN extracts temporal features from each detector waveform.
2. Normalized detector coordinates add geometry.
3. A shared detector MLP creates one learned representation per detector.
4. Permutation-invariant masked statistics combine active detectors into one event vector.
5. Three heads predict shower probability, supplied log-energy, and core coordinates.

This is a good exam model because it is compact, physics-motivated, handles inactive detectors explicitly, and has a clean permutation-invariance argument. Its limitation is that equal-weight pooling cannot learn that some active detectors are more informative than others.

Attention Deep Sets keeps the same CNN, coordinate features, detector encoder, event encoder, heads, losses, and data split. It changes only mean/std aggregation by learning normalized detector weights. This makes it a controlled improvement. Its main drawbacks are extra conceptual complexity and the fact that attention weights are learned importance coefficients, not guaranteed causal explanations.

## Colab limitations to explain honestly

Google states that Colab GPU availability, hardware type, usage limits, idle timeout, and maximum VM lifetime can change and are not guaranteed. Hosted runtimes also lose their temporary filesystem when the session ends. Therefore:

- quick mode is the reliable classroom demonstration;
- full mode may finish, but should not be the only copy of an important experiment;
- the completed local metrics remain the presentation reference;
- optional Google Drive storage can preserve downloaded data and outputs across sessions.

Official references:

- https://research.google.com/colaboratory/faq.html
- https://research.google.com/colaboratory/local-runtimes.html

Colab can also use a local runtime, which gives the Colab interface while executing on your desktop hardware. Only connect notebooks you trust because they can access the local machine.

## Colab quick mode

Purpose: verify download, preprocessing, GPU detection, training, checkpointing, validation evaluation, and plots without claiming final scientific performance.

Configuration:

```python
RUN_MODE = "quick"
MODEL_VARIANT = "original"
USE_GOOGLE_DRIVE = False
EVALUATE_TEST = False
```

Expected training configuration:

- 5,000 training events;
- 1,000 validation events;
- 5 epochs;
- batch size 64;
- Original Deep Sets by default.

The quick subset is deterministic because it is taken from the same seed-42 split. Its result can differ from the full run and must be labeled “quick demo,” not “final result.”

## Colab full mode

Configuration:

```python
RUN_MODE = "full"
MODEL_VARIANT = "original"
USE_GOOGLE_DRIVE = True
EVALUATE_TEST = True
```

Expected training configuration:

- 35,000 training events;
- 7,500 validation events;
- up to 40 epochs with eight-epoch early stopping;
- batch size 64;
- test evaluation only after the model is selected by validation loss.

With Drive enabled, the compressed archive and training outputs persist, while the extracted memory-mapped arrays stay on Colab's faster temporary disk. After a disconnect, reconnect, remount Drive, and rerun setup/preparation; the archive is reused but the extracted arrays must be recreated. If training disconnects before completion, restart training. The current program saves the best model and history but does not resume optimizer state mid-run.

## Local NVIDIA GPU workflow

### 1. Check the GPU and driver

```bash
nvidia-smi
```

### 2. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the PyTorch build compatible with the driver using the command from https://pytorch.org/get-started/locally/, then:

```bash
pip install -r requirements-colab.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. Download and prepare data

```bash
bash scripts/download_data.sh
python -m cosmic_ml.prepare_data --data-dir data
```

Preparation performs a stratified seed-42 70/15/15 split, computes normalization only from training data, and writes memory-mappable arrays plus statistics.

### 4. Train without touching another run

```bash
python -m cosmic_ml.train \
  --model-type original \
  --data-dir data \
  --output-dir outputs/original_full \
  --epochs 40 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda
```

Use a distinct directory for Attention:

```bash
python -m cosmic_ml.train \
  --model-type attention \
  --data-dir data \
  --output-dir outputs/attention_full \
  --epochs 40 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda
```

### 5. Evaluate

First inspect validation performance. Evaluate the test split once for the chosen final checkpoint:

```bash
python -m cosmic_ml.evaluate \
  --checkpoint outputs/original_full/best.pt \
  --data-dir data \
  --split test \
  --device cuda
```

## GitHub setup and update checklist

### Initial setup

1. Create an empty GitHub repository.
2. Update the repository URL in the notebook and README.
3. Add the MIT `LICENSE` file.
4. Confirm that `git status` does not include unintended files from `data/`, `outputs/`, or `build/`.
5. Do not commit the downloaded dataset, extracted arrays, or large temporary training artifacts such as `.npy`, `.pt`, `.pth`, or dataset `.npz` files. Selected compact result files stored intentionally under `results/` may be kept for reproducibility.
6. Run the CPU tests and validate the notebook.
7. If a manifest is maintained, regenerate `MANIFEST.sha256` only after all intended file changes are complete and verify it from the repository root.
8. Review the final Git diff, then commit and push.
9. Open `Cosmic_Ray_ML_Colab.ipynb` from GitHub in Colab and confirm that the **Open in Colab** badge works.
10. Run `quick` mode once to verify the complete portable pipeline.
11. Keep executed notebook outputs only when they are reasonably small and clearly labeled.

### Subsequent updates

For later repository changes:

1. Confirm that `git status` contains only the files you intend to update and does not list anything under `data/`, `outputs/`, or `build/`.
2. Check that no dataset, extracted arrays, temporary outputs, or other large generated files have been added accidentally.
3. Do not commit `.pt`, `.pth`, `.npy`, or dataset/prediction `.npz` files.
4. Run the CPU tests and validate that the clean notebook can be generated.
5. Regenerate `MANIFEST.sha256` only after every intended file change is complete.
6. Verify the manifest from the repository root.
7. Review the final diff, then commit and push the changes.
8. Reopen the main notebook from GitHub in Colab and confirm that the badge, repository update, and quick-mode execution still work.

### Useful commands

For the initial repository setup:

```bash
git init
git add .
git status
git commit -m "Add reproducible Colab and local GPU workflow"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Maintenance commands:

```bash
python -m unittest discover -s tests -v
python scripts/build_notebook.py --force
python scripts/build_manifest.py

# macOS
shasum -a 256 -c MANIFEST.sha256

git status
git add .
git commit -m "Update reproducible Colab workflow"
git push
```

On Linux, use `sha256sum -c MANIFEST.sha256` for the verification step. For a
fork, generate a retargeted clean notebook with
`--repo-slug OWNER/REPOSITORY` before replacing any canonical notebook.


## Experimental setup and reproducibility

The main model comparison uses completed Colab GPU full runs, while the same pipeline is also supported on local NVIDIA GPUs for persistent hardware and storage. Colab defaults to a short deterministic demonstration, but full mode preserves the same validation and test protocol. Final metrics come from the completed full runs, not from quick mode.

## Reproducibility boundaries

- Deterministic splits and seeds reduce variability but GPU kernels and hardware can still cause small numerical differences.
- Colab may assign a GPU different from the desktop GTX 1080 Ti, so wall-clock time and exact learning trajectory can differ.
- Software versions evolve. The requirements constrain scientific packages, while the notebook records the actual Python, PyTorch, CUDA, and GPU versions used at runtime.
- Reference metrics are provenance-labeled and are not presented as newly rerun Colab results.
