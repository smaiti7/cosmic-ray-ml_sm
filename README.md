# Cosmic-Ray ML — Colab and Local GPU Edition

This is the portable GitHub release of the University of Genova cosmic-ray shower project. It keeps the validated local workflow and adds a single Google Colab notebook for easy demonstration and transfer.

## Recommended use

- **For reproducible final numbers:** use the full local GPU run. A local GPU has persistent storage, predictable hardware, and no hosted-runtime timeout.
- **GitHub sharing and live demonstration:** use the Colab notebook in `quick` mode. It trains on a subset and proves that the complete pipeline works.
- **Optional hosted rerun:** use the notebook in `full` mode only when a Colab GPU and enough runtime/storage are available.

The default presentation model is **Original Deep Sets**. **Attention Deep Sets** is included as a controlled comparison that changes only detector aggregation.

## Repository contents

| Path | Purpose |
|---|---|
| `Cosmic_Ray_ML_Colab.ipynb` | Documented end-to-end notebook with quick/full switches |
| `cosmic_ml/` | Preprocessing, models, training, evaluation, and metrics |
| `scripts/download_data.sh` | Downloads the official 1.8 GB dataset and checks its byte size |
| `results/reference_metrics.json` | Frozen results from the completed local GPU experiments |
| `docs/COLAB_AND_LOCAL_GUIDE.md` | Detailed instructions, design reasons, and limitations |
| `tests/` | CPU unit tests for Original and Attention Deep Sets |

Large files are intentionally excluded by `.gitignore`: the dataset, extracted arrays, model checkpoints, predictions, and run outputs must not be committed to GitHub.

## Repository

GitHub user: `smaiti7`  
Repository: `cosmic-ray-ml_sm`

The Colab badge and clone configuration point to this repository.

## Run in Google Colab

1. Upload this folder to a GitHub repository.
2. Open `Cosmic_Ray_ML_Colab.ipynb` on GitHub and click the **Open in Colab** badge.
3. In Colab select **Runtime > Change runtime type > GPU**.
4. Leave `RUN_MODE = "quick"` and `MODEL_VARIANT = "original"` for the first run.
5. Choose **Runtime > Run all**.
6. Change `MODEL_VARIANT` to `attention` only when you want the controlled extension.
7. Use `RUN_MODE = "full"` for the full dataset and up to 40 epochs. Keep the tab/runtime active and consider saving outputs to Google Drive.

Quick mode uses 5,000 training and 1,000 validation events for 5 epochs. Its metrics are a pipeline demonstration and must **not** replace the final reference metrics from the completed full runs in the presentation.

## Run locally with an NVIDIA GPU

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install the CUDA-enabled PyTorch command selected at https://pytorch.org/get-started/locally/
pip install -r requirements-colab.txt
bash scripts/download_data.sh
python -m cosmic_ml.prepare_data --data-dir data
python -m cosmic_ml.train --model-type original --data-dir data --output-dir outputs/original --epochs 40 --batch-size 64 --device cuda
python -m cosmic_ml.evaluate --checkpoint outputs/original/best.pt --data-dir data --split test --device cuda
```

The exact PyTorch CUDA install command depends on the installed NVIDIA driver. Use the current official PyTorch selector rather than hard-coding a CUDA wheel in the repository.

For Attention Deep Sets, change `--model-type original` to `--model-type attention` and use a different output directory.

## Scientific evaluation rule

Use training data to fit weights, validation data for model selection and early stopping, and the frozen test split only for the final selected model. Energy and core losses/metrics are evaluated only for true signal events because those targets are undefined for background.

## Reference results

The checked-in JSON contains results already obtained from complete local GPU runs. The main values are:

| Metric | Original | Attention |
|---|---:|---:|
| ROC-AUC | 0.99019 | 0.98956 |
| Accuracy | 0.98333 | 0.98533 |
| Brier score | 0.01467 | 0.01347 |
| Energy MAE | 0.08442 | 0.08283 |
| Median core error | 0.64290 | 0.61303 |
| Core p68 error | 0.94166 | 0.87575 |

Attention improves several calibration and reconstruction metrics, but Original Deep Sets remains the simplest strong model to explain. The comparison is a trade-off, not a universal win.

## License and data

No license is invented or added in this release. Add the license approved by the project owners before public redistribution. The dataset stays at its official external download location and is not redistributed here.
