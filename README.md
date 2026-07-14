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

Deactivate the virtual environment:

```powershell
deactivate
```