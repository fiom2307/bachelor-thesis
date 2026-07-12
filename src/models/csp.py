import numpy as np
from mne.decoding import CSP

from src.utils.config import (
    CSP_N_COMPONENTS,
    CSP_REG,
)


def fit_csp(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> CSP:
    """
    Fit Common Spatial Patterns (CSP) using the training EEG epochs.

    CSP learns spatial filters that emphasize differences in signal
    variance between motor imagery classes.

    Four components are retained to obtain a compact feature
    representation. Ledoit-Wolf regularization is used to stabilize
    covariance estimation, and the logarithm of the average component
    power is used as input for the LDA classifier.
    """
    csp = CSP(
        n_components=CSP_N_COMPONENTS,
        reg=CSP_REG,
        log=True,
        norm_trace=False,
    )

    csp.fit(X_train, y_train)

    return csp


def apply_csp(csp: CSP, X: np.ndarray) -> np.ndarray:
    """
    Transform EEG epochs into CSP features.

    The fitted spatial filters are applied to the EEG signals, producing
    log-power features that can be used as input for the LDA classifier.
    """
    return csp.transform(X)