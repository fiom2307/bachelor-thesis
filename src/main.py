from pathlib import Path
from data_loader import load_epochs, load_left_right_true_labels
from csp import fit_csp, apply_csp
from lda import train_lda, predict
from evaluate import accuracy

DATA_DIR = Path("data/")

res = []

# 9 subjects
for subj in range(1, 10):
    t_file  = DATA_DIR / f"A0{subj}T.gdf"
    e_file  = DATA_DIR / f"A0{subj}E.gdf"
    mat_file = DATA_DIR / f"A0{subj}E.mat"

    if not (t_file.exists() and e_file.exists() and mat_file.exists()):
        print(f"Missing files for subject {subj}: Skiping")
        continue

    # Train
    X_train, y_train = load_epochs(t_file, mat_file)
    if X_train is None:
        continue

    # 4 csp components (between 4-8 normally)
    csp = fit_csp(X_train, y_train, n_components=4)
    X_train_csp = apply_csp(csp, X_train)
    lda = train_lda(X_train_csp, y_train)

    # Eval
    X_eval, _ = load_epochs(e_file, mat_file)
    if X_eval is None:
        continue

    y_true = load_left_right_true_labels(mat_file=mat_file)

    X_eval_csp = apply_csp(csp, X_eval)
    y_pred = predict(lda, X_eval_csp)

    acc = accuracy(y_true, y_pred)

    text = f"A0{subj} -- Accuracy: {acc:.4f} ({acc*100:.1f}%)"
    res.append(text)

for text in res:
    print(text)