import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)


def compute_fold_channel_relevance(
    csp: CSP,
    lda: LinearDiscriminantAnalysis,
) -> np.ndarray:
    """
    Compute class-wise channel relevance for one CSP+LDA fold.

    CSP spatial patterns describe how each CSP component is
    expressed across EEG channels. LDA coefficients describe
    how strongly each CSP feature contributes to each class.

    Returns
    -------
    channel_relevance : np.ndarray
        Class-wise normalized channel relevance.
        Shape: (n_classes, n_channels).
    """
    n_components = csp.n_components

    patterns = np.asarray(
        csp.patterns_[:n_components],
        dtype=np.float64,
    )

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
    # CSP spatial relevance
    # ----------------------------------------------------------

    # Square patterns to remove CSP sign ambiguity.
    pattern_relevance = patterns**2

    # Normalize each CSP component across EEG channels.
    pattern_sums = np.sum(
        pattern_relevance,
        axis=1,
        keepdims=True,
    )

    pattern_sums = np.maximum(
        pattern_sums,
        np.finfo(np.float64).eps,
    )

    pattern_relevance = (
        pattern_relevance
        / pattern_sums
    )

    # ----------------------------------------------------------
    # Class-wise LDA component relevance
    # ----------------------------------------------------------

    # Absolute coefficient magnitude:
    # (n_classes, n_components)
    component_relevance = np.abs(
        coefficients
    )

    # ----------------------------------------------------------
    # Project CSP component relevance back to EEG channels
    # ----------------------------------------------------------

    # (n_classes, n_components)
    # @
    # (n_components, n_channels)
    #
    # ->
    # (n_classes, n_channels)
    channel_relevance = (
        component_relevance
        @ pattern_relevance
    )

    # Normalize each class independently across channels.
    class_sums = np.sum(
        channel_relevance,
        axis=1,
        keepdims=True,
    )

    class_sums = np.maximum(
        class_sums,
        np.finfo(np.float64).eps,
    )

    channel_relevance = (
        channel_relevance
        / class_sums
    )

    return channel_relevance


def compute_channel_relevance(
    csps: list[CSP],
    ldas: list[LinearDiscriminantAnalysis],
) -> np.ndarray:
    """
    Compute mean class-wise channel relevance across CSP+LDA folds.

    Each fold is interpreted separately before averaging.

    Parameters
    ----------
    csps : list[CSP]
        CSP models from all folds.

    ldas : list[LinearDiscriminantAnalysis]
        Corresponding LDA models from all folds.

    Returns
    -------
    channel_relevance : np.ndarray
        Mean class-wise normalized channel relevance.
        Shape: (n_classes, n_channels).
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
            compute_fold_channel_relevance(
                csp=csp,
                lda=lda,
            )
        )

        fold_relevances.append(
            relevance
        )

    fold_relevances = np.stack(
        fold_relevances,
        axis=0,
    )

    # Shape before averaging:
    # (n_folds, n_classes, n_channels)
    channel_relevance = np.mean(
        fold_relevances,
        axis=0,
    )

    # Re-normalize each class after fold averaging.
    class_sums = np.sum(
        channel_relevance,
        axis=1,
        keepdims=True,
    )

    class_sums = np.maximum(
        class_sums,
        np.finfo(np.float64).eps,
    )

    channel_relevance = (
        channel_relevance
        / class_sums
    )

    return channel_relevance