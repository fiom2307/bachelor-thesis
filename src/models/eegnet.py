import random

import numpy as np
import tensorflow as tf

from src.models.EEGModels import EEGNet
from src.utils.config import (
    BASE_SEED,
    EEGNET_BATCH_SIZE,
    EEGNET_D,
    EEGNET_DROPOUT_RATE,
    EEGNET_DROPOUT_TYPE,
    EEGNET_EARLY_STOPPING_PATIENCE,
    EEGNET_F1,
    EEGNET_F2,
    EEGNET_KERNEL_LENGTH,
    EEGNET_LEARNING_RATE,
    EEGNET_MAX_EPOCHS,
    N_CLASSES,
    N_FOLDS,
)
from src.utils.cross_validation import (
    average_fold_probabilities,
    get_stratified_folds,
)
from src.utils.paths import (
    get_eegnet_fold_model_path,
    get_subject_name,
)


def set_seed(seed: int) -> None:
    """
    Set the random seeds used by Python, NumPy, and TensorFlow.

    This improves the reproducibility of model training. Deterministic
    TensorFlow operations are enabled when they are supported by the
    current environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def create_eegnet_model(
    n_channels: int,
    n_samples: int,
) -> tf.keras.Model:
    """
    Create and compile the EEGNet-8,2 model.

    The temporal convolution uses a kernel length of 125 samples. At the
    dataset sampling rate of 250 Hz, this corresponds to a 0.5-second
    temporal window and allows the model to learn frequency-related
    patterns in the sensorimotor mu and beta bands.

    A dropout rate of 0.5 is used to reduce overfitting in the
    within-subject setting, where only a limited number of training
    trials is available. During training, dropout randomly disables
    half of the affected units, encouraging the model to learn features
    that do not depend too strongly on individual activations.

    Depthwise convolutions learn spatial filters across the EEG channels.
    """
    model = EEGNet(
        nb_classes=N_CLASSES,
        Chans=n_channels,
        Samples=n_samples,
        dropoutRate=EEGNET_DROPOUT_RATE,
        kernLength=EEGNET_KERNEL_LENGTH,
        F1=EEGNET_F1,
        D=EEGNET_D,
        F2=EEGNET_F2,
        dropoutType=EEGNET_DROPOUT_TYPE,
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=EEGNET_LEARNING_RATE,
        ),
        metrics=["accuracy"],
    )

    return model


def train_or_load_eegnet(
    subject: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> list[tf.keras.Model]:
    """
    Train or load one EEGNet model for each stratified fold.

    Each fold model is trained on 80% of the subject's training trials.
    The remaining 20% is used as validation data for early stopping.
    A fold-specific seed is used to make the training reproducible.

    Previously saved models are loaded instead of being trained again.
    """
    seed = BASE_SEED + subject
    set_seed(seed)

    n_channels = X_train.shape[1]
    n_samples = X_train.shape[2]

    models = []

    for fold, train_idx, val_idx in get_stratified_folds(
        X_train,
        y_train,
        seed,
    ):
        model_path = get_eegnet_fold_model_path(subject, fold)

        if model_path.exists():
            model = tf.keras.models.load_model(model_path)
            models.append(model)
            continue

        print(
            f"Training {get_subject_name(subject)} "
            f"EEGNet fold {fold}/{N_FOLDS}"
        )

        set_seed(seed + fold)

        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        y_tr_cat = tf.keras.utils.to_categorical(
            y_tr,
            num_classes=N_CLASSES,
        )
        y_val_cat = tf.keras.utils.to_categorical(
            y_val,
            num_classes=N_CLASSES,
        )

        model = create_eegnet_model(
            n_channels,
            n_samples,
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EEGNET_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )

        model.fit(
            X_tr,
            y_tr_cat,
            epochs=EEGNET_MAX_EPOCHS,
            batch_size=EEGNET_BATCH_SIZE,
            validation_data=(X_val, y_val_cat),
            callbacks=[early_stopping],
            verbose=0,
        )

        model.save(model_path)
        models.append(model)

    return models


def predict_eegnet(
    models: list[tf.keras.Model],
    X_eval: np.ndarray,
) -> np.ndarray:
    """
    Predict the final classes using the EEGNet fold ensemble.

    Every fold model predicts class probabilities for the evaluation
    trials. The probabilities are averaged, and the class with the
    highest average probability is selected.
    """
    probabilities = []

    for model in models:
        fold_probabilities = model.predict(
            X_eval,
            verbose=0,
        )
        probabilities.append(fold_probabilities)

    return average_fold_probabilities(probabilities)