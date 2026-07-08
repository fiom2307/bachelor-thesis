import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from src.models.EEGModels import EEGNet
from src.utils.paths import EEGNET_MODEL_DIR

from sklearn.model_selection import StratifiedKFold

# epoch = trial aaaa (not in eegnet context)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def get_eegnet_model_path(subject):
    return EEGNET_MODEL_DIR / f"A{subject:02d}_eegnet.keras"

def create_eegnet_model(n_classes, n_channels, n_samples):
    model = EEGNet(
        nb_classes=n_classes,
        Chans=n_channels,
        Samples=n_samples,
        dropoutRate=0.5, # to avoid overfitting, paper, within subject
        kernLength=125, # paper ( allows for capturing frequency information at 2 Hz), sampled with 250Hz (desc data) / 2 =  125 # was 125
        F1=8, # paper eegnet-8,2
        D=2,
        F2=16, # 8*2 paper
        dropoutType="Dropout"
    )

    model.compile(
        loss="categorical_crossentropy", #paper
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # paper adam
        metrics=["accuracy"]
    )
    return model

def get_eegnet_fold_model_path(subject, fold, base_seed):
    return EEGNET_MODEL_DIR / f"A{subject:02d}_eegnet_kfold_seed{base_seed}_fold{fold}.keras"


def train_or_load_eegnet(subject, X_train, y_train):
    base_seed = 42
    seed = base_seed + subject
    set_seed(seed)

    model_path = get_eegnet_model_path(subject)

    n_classes = 2
    n_channels = X_train.shape[1]
    n_samples = X_train.shape[2]

    n_folds = 5

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed
    )

    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        model_path = get_eegnet_fold_model_path(subject, fold, base_seed)

        if model_path.exists():
            model = tf.keras.models.load_model(model_path)
            models.append(model)
            continue

        print(f"Training A{subject:02d} EEGNet fold {fold}/{n_folds}")

        set_seed(seed + fold)

        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        y_tr_cat = tf.keras.utils.to_categorical(y_tr, num_classes=n_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)

        model = create_eegnet_model(
            n_classes,
            n_channels,
            n_samples
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        )

        model.fit(
            X_tr,
            y_tr_cat,
            epochs=500,
            batch_size=16,
            validation_data=(X_val, y_val_cat),
            callbacks=[early_stopping],
            verbose=0
        )

        model.save(model_path)

        models.append(model)

    # y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=n_classes) # into categorical/vectors
    
    # model.fit(
    #     X_train,
    #     y_train_cat,
    #     epochs=500, # looks at the whole training dataset n times, paper
    #     batch_size=16, # se separa en grupos peques, 144 training trials/16=9 batches
    #     validation_split=0.2,
    #     callbacks=[early_stopping],
    #     verbose=0
    # )

    return models


def predict_eegnet(models, X_eval):
    if not isinstance(models, list):
        models = [models]

    probabilities = []

    for model in models:
        fold_probabilities = model.predict(X_eval, verbose=0)
        probabilities.append(fold_probabilities)

    mean_probabilities = np.mean(probabilities, axis=0)
    y_pred = np.argmax(mean_probabilities, axis=1)

    return y_pred