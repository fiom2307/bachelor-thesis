# Motor Imagery EEG Representation Analysis

Code for the Bachelor's thesis:

> A Comparative Analysis of Learned Representations in Motor Imagery EEG Decoders with XAI

The repository compares CSP+LDA, EEGNet, and an additional CSP+SVM baseline on BCI Competition IV Dataset 2a, then analyzes spatial, temporal, and frequency relevance patterns.

## Repository Structure

```text
src/
  data/              preprocessing, loading, labels, dataset access
  models/            model implementations and train/load wrappers
  pipelines/         subject-level evaluation pipelines
  analysis/          reusable relevance/statistical analysis code
  visualization/     plotting helpers
  utils/             shared config, paths, result helpers

scripts/             runnable entry points
data/                local BCI IV 2a files, ignored by Git
models/              generated trained model caches, ignored by Git
results/             generated metrics, statistics, and figures
```

Generated artifacts are organized under:

- `models/csp_lda`, `models/eegnet`, `models/csp_svm`
- `models/experiments/{spatially_smoothed_csp,early_weighted_csp}`
- `models/exploratory/...`
- `results/performance`, `results/representations`, `results/statistics`
- `results/experiments/{spatially_smoothed_csp,early_weighted_csp}`
- `results/exploratory/...`

## Setup

This project was developed with Python 3.12.4.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the BCI Competition IV Dataset 2a files in `data/`:

```text
data/A01T.gdf
data/A01E.gdf
data/A01E.mat
...
data/A09T.gdf
data/A09E.gdf
data/A09E.mat
```

The `data/`, generated `models/`, and generated `results/` folders are local artifacts and are ignored by Git, except for selected thesis figures kept under `results/`.

## Main Execution Order

Run the main model comparison:

```powershell
python -m scripts.compare_model_accuracies
python -m scripts.compare_model_accuracies_with_svm
```

Generate performance outputs:

```powershell
python -m scripts.plot_confusion_matrices
python -m scripts.print_classification_reports
python -m scripts.report_rejected_trials
```

Generate representation analyses:

```powershell
python -m scripts.plot_csp_pattern_analysis
python -m scripts.plot_shap_analysis
```

Run statistical analyses:

```powershell
python -m scripts.run_performance_statistical_analysis
python -m scripts.run_channel_statistical_analysis
python -m scripts.run_channel_sensorimotor_statistical_analysis
python -m scripts.run_channel_entropy_statistical_analysis
python -m scripts.run_temporal_statistical_analysis
python -m scripts.run_frequency_statistical_analysis
```

Run the two final representation-guided experiments:

```powershell
python -m scripts.csp_spatially_distributed
python -m scripts.compare_spatially_distributed_csp
python -m scripts.csp_early_weighted
python -m scripts.compare_csp_early_weighted_temporal_relevance
```

Exploratory scripts include the CSP/EEGNet time-window comparisons, CSP component-count comparison, channel/ROI ablations, and temporal ablation experiment.

## Notes

`src/models/EEGModels.py` identifies itself as `ARL_EEGModels`, a Keras/TensorFlow collection of EEG CNN models. It is used here for the EEGNet implementation and retains the license/provenance text included in that file.
