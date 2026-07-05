import shap
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf

from src.models.eegnet import (
    train_or_load_eegnet,
    predict_eegnet,
    set_seed,
    create_eegnet_model
)

from src.data.preprocessing import (
    normalize_epochs,
    prepare_eegnet_input
)

from src.models.csp_lda import train_csp_lda, predict_csp_lda

def run_eegnet_for_subject(subj, data):

    X_train, y_train, X_eval, y_eval = data

    X_train, X_eval = normalize_epochs(X_train, X_eval)

    X_train = prepare_eegnet_input(X_train)
    X_eval = prepare_eegnet_input(X_eval)

    model = train_or_load_eegnet(
        subj,
        X_train,
        y_train,
    )

    # rng = np.random.default_rng(42)
    # background_idx = rng.choice(len(X_train), size=50, replace=False)

    # background = X_train[background_idx].astype(np.float32)

    # -------- choose class to explain --------
    # class_index = 0  # 0 = left hand, 1 = right hand
    # class_name = "Left hand"

    # Predict labels for evaluation trials
    # y_pred_probs = model.predict(X_eval)
    # y_pred = np.argmax(y_pred_probs, axis=1)

    # Use only correctly classified trials of the selected class
    # correct_class_mask = (y_eval == class_index) & (y_pred == class_index)

    # X_explain = X_eval[correct_class_mask].astype(np.float32)

    # print(f"Correct {class_name} trials:", X_explain.shape)

    # -------- compute SHAP --------
    # explainer = shap.GradientExplainer(model, background)

    # shap_values = explainer.shap_values(X_explain)

    # print("SHAP computed")
    # print(type(shap_values))
    # print("shap_values shape:", shap_values.shape)

    # Take SHAP values for the selected output class
    # sv = shap_values[..., class_index]

    # print("sv shape before squeeze:", sv.shape)

    # sv = sv.squeeze(-1)

    # print("sv shape after squeeze:", sv.shape)

    # channel_names = [
    #     "Fz",
    #     "FC3", "FC1", "FCz", "FC2", "FC4",
    #     "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    #     "CP3", "CP1", "CPz", "CP2", "CP4",
    #     "P1", "Pz", "P2", "POz"
    # ]

    # channel_importance = np.mean(np.abs(sv), axis=(0, 2))

    # print("channel_importance shape:", channel_importance.shape)

    # plt.figure(figsize=(10, 4))
    # plt.bar(channel_names, channel_importance)
    # plt.xticks(rotation=90)
    # plt.ylabel("Mean absolute SHAP value")
    # plt.title(f"EEGNet SHAP Channel Importance - Correct {class_name}")
    # plt.tight_layout()
    # plt.show()

    y_pred = predict_eegnet(model, X_eval)

    accuracy = accuracy_score(y_eval, y_pred)

    return accuracy


def run_crossval_for_subject(subject, X, y):

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("labels:", np.unique(y, return_counts=True))


    seed = 42 + subject

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    csp_accuracies = []
    eegnet_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"A{subject:02d} fold {fold}/5")

        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]

        X_test_fold = X[test_idx]
        y_test_fold = y[test_idx]

        # -------------------------
        # CSP + LDA
        # -------------------------
        csp, lda = train_csp_lda(X_train_fold, y_train_fold)
        y_pred_csp = predict_csp_lda(csp, lda, X_test_fold)

        csp_acc = accuracy_score(y_test_fold, y_pred_csp)
        csp_accuracies.append(csp_acc)

        # -------------------------
        # EEGNet
        # -------------------------
        tf.keras.backend.clear_session()
        set_seed(seed + fold)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_fold,
            y_train_fold,
            test_size=0.2,
            stratify=y_train_fold,
            random_state=seed + fold
        )

        # Important for TensorFlow
        X_tr = X_tr.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test_eegnet = X_test_fold.astype(np.float32)

        # Optional but very useful for EEGNet
        mean = X_tr.mean(axis=(0, 2), keepdims=True)
        std = X_tr.std(axis=(0, 2), keepdims=True) + 1e-6

        X_tr = (X_tr - mean) / std
        X_val = (X_val - mean) / std
        X_test_eegnet = (X_test_eegnet - mean) / std

        n_classes = 2
        n_channels = X.shape[1]
        n_samples = X.shape[2]

        y_tr_cat = tf.keras.utils.to_categorical(y_tr, num_classes=n_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)

        model = create_eegnet_model(
            n_classes=n_classes,
            n_channels=n_channels,
            n_samples=n_samples
        )

        print("model input shape:", model.input_shape)
        print("X_tr shape:", X_tr.shape)
        print("X_val shape:", X_val.shape)
        print("X_test shape:", X_test_eegnet.shape)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        )

        history = model.fit(
            X_tr,
            y_tr_cat,
            epochs=500,
            batch_size=16,
            validation_data=(X_val, y_val_cat),
            callbacks=[early_stopping],
            verbose=0
        )

        print("epochs trained:", len(history.history["loss"]))
        print("best train acc:", max(history.history["accuracy"]))
        print("best val acc:", max(history.history["val_accuracy"]))

        # Check if EEGNet learned the training data
        train_probabilities = model.predict(X_tr, verbose=0)
        y_pred_train = np.argmax(train_probabilities, axis=1)

        print("EEGNet train pred labels:", np.bincount(y_pred_train.astype(int)))
        print("EEGNet train acc:", accuracy_score(y_tr, y_pred_train))

        # Test fold prediction
        probabilities = model.predict(X_test_eegnet, verbose=0)
        y_pred_eegnet = np.argmax(probabilities, axis=1)

        eegnet_acc = accuracy_score(y_test_fold, y_pred_eegnet)
        eegnet_accuracies.append(eegnet_acc)

        print("EEGNet y_tr labels:", np.bincount(y_tr.astype(int)))
        print("EEGNet y_val labels:", np.bincount(y_val.astype(int)))
        print("test labels:", np.bincount(y_test_fold.astype(int)))
        print("pred labels:", np.bincount(y_pred_eegnet.astype(int)))
        print("EEGNet fold acc:", eegnet_acc)

    mean_csp_acc = np.mean(csp_accuracies)
    mean_eegnet_acc = np.mean(eegnet_accuracies)

    print(f"A{subject:02d} CSP+LDA CV accuracy: {mean_csp_acc:.4f}")
    print(f"A{subject:02d} EEGNet CV accuracy: {mean_eegnet_acc:.4f}")

    return mean_csp_acc, mean_eegnet_acc