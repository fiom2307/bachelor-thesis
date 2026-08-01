## Setup

This project was developed using:

```text
Python 3.12.4
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Analysis

Run the model accuracy comparison:

```powershell
python -m scripts.compare_model_accuracies
```

Generate the confusion matrices:

```powershell
python -m scripts.plot_confusion_matrices
```

Print the classification reports:

```powershell
python -m scripts.print_classification_reports
```

Report the number of rejected and retained trials for each subject:

```powershell
python -m scripts.report_rejected_trials
```

Generate the ERD/ERS topographies, time-frequency representations, and power spectral density plots:

```powershell
python -m scripts.plot_spectral_analysis
```

Generate the CSP+LDA occlusion relevance plots:

```powershell
python -m scripts.plot_csp
```

Compute and generate the EEGNet SHAP relevance plots:

```powershell
python -m scripts.plot_shap
```

Compare the channel-wise and temporal relevance obtained for EEGNet and CSP+LDA:

```powershell
python -m scripts.compare_relevance
```

Deactivate the virtual environment after finishing:

```powershell
deactivate
```