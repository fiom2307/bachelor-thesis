import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from src.models.EEGModels import EEGNet
from src.utils.paths import EEGNET_MODEL_DIR

# epoch = trial aaaa (not in eegnet context)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def get_eegnet_model_path(subject):
    return EEGNET_MODEL_DIR / f"A{subject:02d}_eegnet.keras"


def train_or_load_eegnet(subject, X_train, y_train):
    set_seed(42 + subject)

    model_path = get_eegnet_model_path(subject)

    n_classes = 2
    n_channels = X_train.shape[1]
    n_samples = X_train.shape[2]

    if model_path.exists():
        print(f"Loading saved EEGNet model for A{subject:02d}")
        model = tf.keras.models.load_model(model_path)
        return model

    print(f"Training EEGNet model for A{subject:02d}")

    model = EEGNet(
        nb_classes=n_classes,
        Chans=n_channels,
        Samples=n_samples,
        dropoutRate=0.5, # to avoid overfitting, paper, within subject
        kernLength=64, # paper ( allows for capturing frequency information at 2 Hz), sampled with 250Hz (desc data) / 2 =  125 # was 125
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

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,# logic it says, some rule to avoid stopping too early because of small random
        restore_best_weights=True
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42 + subject,
        shuffle=True
    )

    y_tr_cat = tf.keras.utils.to_categorical(y_tr, num_classes=n_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)

    model.fit(
        X_tr,
        y_tr_cat,
        epochs=500,
        batch_size=16,
        validation_data=(X_val, y_val_cat),
        callbacks=[early_stopping],
        verbose=0
    )
    
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

    model.save(model_path)
    print(f"Saved EEGNet model for A{subject:02d} at {model_path}")

    return model


def predict_eegnet(model, X_eval):
    probabilities = model.predict(X_eval, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    return y_pred