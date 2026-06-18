from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from data_loader import load_epochs, load_left_right_true_labels
from eegnet_train import (
    normalize_epochs,
    prepare_eegnet_input,
    train_or_load_eegnet,
    predict_eegnet,
)



DATA_DIR = Path("data/")

accuracies = []
subjects_to_run = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for subj in subjects_to_run:

    train_file  = DATA_DIR / f"A0{subj}T.gdf"
    eval_file  = DATA_DIR / f"A0{subj}E.gdf"
    mat_file = DATA_DIR / f"A0{subj}E.mat"

    X_train, y_train = load_epochs(train_file, None)
    X_eval, _ = load_epochs(eval_file, mat_file)
    y_true = load_left_right_true_labels(mat_file)

    X_train, X_eval = normalize_epochs(X_train, X_eval)

    X_train = prepare_eegnet_input(X_train)
    X_eval = prepare_eegnet_input(X_eval)

    model = train_or_load_eegnet(
        subj,
        X_train,
        y_train,
    )

    y_pred = predict_eegnet(model, X_eval)

    acc = accuracy_score(y_true, y_pred)
    accuracies.append(f"0{subj}: {acc:.4f}")

for text in accuracies:
    print(text)