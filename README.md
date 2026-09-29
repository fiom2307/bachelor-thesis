# Motor Imagery EEG Representation Analysis

CSP+LDA and EEGNet comparison on BCI Competition IV Dataset 2a, including spatial, temporal, and frequency representation analysis. CSP+SVM is included as an additional baseline.

## Repository Structure

```text
src/
  data/              preprocessing and dataset loading
  models/            model implementations
  pipelines/         evaluation pipelines
  analysis/
    csp_pattern_analysis/  
    shap_analysis/         
    statistics/            
    exploratory/           
  visualization/     plotting functions
  utils/             configuration, paths, and shared utilities

scripts/
  main/              model evaluation and representation analysis
  statistics/        statistical tests
  experiments/       representation-guided experiments
  exploratory/       additional experiments

data/                BCI Competition IV Dataset 2a
models/              trained models
results/             generated results and figures
```

`data/`, `models/`, and most of `results/` are ignored by Git.

## Setup

The project was developed with Python 3.12.4.

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

## Running the Code

### Model evaluation

```powershell
python -m scripts.main.compare_model_accuracies
python -m scripts.main.compare_model_accuracies_with_svm
python -m scripts.main.plot_confusion_matrices
python -m scripts.main.print_classification_reports
python -m scripts.main.report_rejected_trials
```

### Representation analysis

```powershell
python -m scripts.main.plot_csp_pattern_analysis
python -m scripts.main.plot_shap_analysis
```

### Statistical analysis

```powershell
python -m scripts.statistics.run_performance_statistical_analysis
python -m scripts.statistics.run_channel_statistical_analysis
python -m scripts.statistics.run_channel_sensorimotor_statistical_analysis
python -m scripts.statistics.run_channel_entropy_statistical_analysis
python -m scripts.statistics.run_temporal_statistical_analysis
python -m scripts.statistics.run_frequency_statistical_analysis
```

### Representation-guided experiments

```powershell
python -m scripts.experiments.csp_spatially_distributed
python -m scripts.experiments.compare_spatially_distributed_csp
python -m scripts.experiments.csp_early_weighted
python -m scripts.experiments.compare_csp_early_weighted_temporal_relevance
```

Additional experiments are available in `scripts/exploratory/`, including the CSP and EEGNet time-window experiments, CSP component comparison, channel and ROI ablations, and temporal ablation.

## EEGNet

The EEGNet implementation uses `src/models/EEGModels.py` from the `ARL_EEGModels` collection. The original provenance and license information are retained in the file.