import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)


def compute_fold_temporal_relevance(
    csp: CSP,
    lda: LinearDiscriminantAnalysis,
    data: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute class-wise temporal relevance for one CSP+LDA fold.

    Temporal relevance is estimated from the instantaneous power
    of CSP component time series, weighted by the class-specific
    absolute LDA coefficients.
    """
    data = np.asarray(
        data,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    if data.ndim != 3:
        raise ValueError(
            "data must have shape "
            "(n_trials, n_channels, n_times)."
        )

    if labels.shape[0] != data.shape[0]:
        raise ValueError(
            "labels must contain one label per trial."
        )

    n_components = csp.n_components

    # CSP spatial filters:
    # (n_components, n_channels)
    filters = np.asarray(
        csp.filters_[:n_components],
        dtype=np.float64,
    )

    # LDA coefficients:
    # (n_classes, n_components)
    coefficients = np.asarray(
        lda.coef_,
        dtype=np.float64,
    )

    if coefficients.ndim == 1:
        coefficients = coefficients[
            np.newaxis,
            :
        ]

    if coefficients.shape[1] != n_components:
        raise ValueError(
            "Number of LDA features does not match "
            "the number of CSP components."
        )

    # ----------------------------------------------------------
    # Project EEG into CSP component space
    # ----------------------------------------------------------

    # data:
    # (trials, channels, times)
    #
    # filters:
    # (components, channels)
    #
    # result:
    # (trials, components, times)
    csp_signals = np.einsum(
        "kc,nct->nkt",
        filters,
        data,
    )

    # Instantaneous power of each CSP component.
    component_power = csp_signals**2

    # ----------------------------------------------------------
    # Class-wise temporal relevance
    # ----------------------------------------------------------

    n_classes = coefficients.shape[0]
    n_times = data.shape[2]

    temporal_relevance = np.zeros(
        (
            n_classes,
            n_times,
        ),
        dtype=np.float64,
    )

    for class_idx in range(n_classes):
        class_mask = labels == class_idx

        if not np.any(class_mask):
            continue

        class_power = component_power[
            class_mask
        ]

        # Mean CSP-component power across trials:
        #
        # (components, times)
        mean_component_power = np.mean(
            class_power,
            axis=0,
        )

        # Importance of each CSP feature for this class.
        component_weights = np.abs(
            coefficients[class_idx]
        )

        # Weighted sum across CSP components.
        #
        # (components,)
        # @
        # (components, times)
        #
        # ->
        # (times,)
        temporal_relevance[class_idx] = (
            component_weights
            @ mean_component_power
        )

        # Normalize temporal relevance within this class.
        total = np.sum(
            temporal_relevance[class_idx]
        )

        if total > 0:
            temporal_relevance[
                class_idx
            ] /= total

    return temporal_relevance


def compute_temporal_relevance(
    csps: list[CSP],
    ldas: list[LinearDiscriminantAnalysis],
    data: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute mean class-wise temporal relevance across CSP+LDA folds.
    """
    if len(csps) != len(ldas):
        raise ValueError(
            "csps and ldas must contain the same "
            "number of models."
        )

    if len(csps) == 0:
        raise ValueError(
            "At least one CSP+LDA model is required."
        )

    fold_relevances = []

    for csp, lda in zip(
        csps,
        ldas,
    ):
        relevance = (
            compute_fold_temporal_relevance(
                csp=csp,
                lda=lda,
                data=data,
                labels=labels,
            )
        )

        fold_relevances.append(
            relevance
        )

    fold_relevances = np.stack(
        fold_relevances,
        axis=0,
    )

    temporal_relevance = np.mean(
        fold_relevances,
        axis=0,
    )

    # Re-normalize each class after averaging folds.
    class_sums = np.sum(
        temporal_relevance,
        axis=1,
        keepdims=True,
    )

    class_sums = np.maximum(
        class_sums,
        np.finfo(np.float64).eps,
    )

    temporal_relevance = (
        temporal_relevance
        / class_sums
    )

    return temporal_relevance