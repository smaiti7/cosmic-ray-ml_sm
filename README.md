# Cosmic-Ray ML — Colab and Local GPU Edition

This is the portable GitHub release of the University of Genova cosmic-ray shower project. It provides an end-to-end PyTorch workflow for shower classification, energy reconstruction, and core-position reconstruction, with both Original Deep Sets and Attention Deep Sets models. The repository includes a portable Colab notebook together with saved results from completed full Colab runs.

## Recommended use

- **Main final comparison:** use the saved completed Colab full-run results for Original and Attention Deep Sets.
- **GitHub sharing and live demonstration:** use `Cosmic_Ray_ML_Colab.ipynb` in `quick` mode.
- **Full rerun:** use `full` mode when sufficient Colab GPU runtime and storage are available; Google Drive can be used for persistent outputs.
- **Reproducibility reference:** earlier desktop-GPU results are retained in `results/reference_metrics.json`.

The default demonstration model is the Baseline or**Original Deep Sets**. **Attention Deep Sets** is a controlled extension that changes only detector aggregation.

## Repository contents

| Path | Purpose |
|---|---|
| `Cosmic_Ray_ML_Colab.ipynb` | Main documented notebook with quick/full switches |
| `cosmic_ml/` | Preprocessing, models, training, evaluation, and metrics |
| `scripts/download_data.sh` | Downloads and verifies the official dataset |
| `results/original_full/` | Saved outputs from the completed Original Colab full run |
| `results/attention_full/` | Saved outputs from the completed Attention Colab full run |
| `results/reference_metrics.json` | Earlier desktop-GPU results used as a reproducibility reference |
| `full_runs/` | Executed archival notebooks for the Original and Attention full runs |
| `docs/COLAB_AND_LOCAL_GUIDE.md` | Additional instructions and implementation notes |
| `tests/` | CPU unit tests for both model variants |

The dataset and extracted training arrays remain excluded by `.gitignore`. Selected compact full-run outputs are included under `results/` for reproducibility, while large temporary training artifacts should not be committed unnecessarily.

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

Quick mode uses 5,000 training and 1,000 validation events for 5 epochs. Its metrics demonstrate that the pipeline works, but the final model comparison in Section 9 is loaded from the saved completed Colab full-run results.

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

## Full-run results

The main comparison uses the completed Colab full runs evaluated on the same 7,500-event test set.

| Metric | Original | Attention |
|---|---:|---:|
| ROC-AUC | 0.98989 | **0.99065** |
| PR-AUC | 0.99378 | **0.99418** |
| Accuracy | 0.98493 | **0.98613** |
| Brier score | 0.01372 | **0.01340** |
| Energy MAE | **0.08416** | 0.08459 |
| Energy RMSE | 0.14553 | **0.14321** |
| Mean core error | 1.35518 | **1.29757** |
| Median core error | 0.64200 | **0.63902** |
| Core P68 error | 0.96211 | **0.93567** |

Attention performs better on **8 of the 9 reported metrics**. The improvements are modest, with the Original model retaining a marginally lower energy MAE. Overall, the results suggest a small benefit from learned detector weighting while the simpler fixed-aggregation baseline remains highly competitive.

Earlier desktop-GPU results are retained in `results/reference_metrics.json` as a reproducibility reference and show broadly consistent performance.

## License and data

No license is invented or added in this release. Add the license approved by the project owners before public redistribution. The dataset stays at its official external download location and is not redistributed here.
