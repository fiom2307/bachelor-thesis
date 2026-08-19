from dataclasses import dataclass

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)


@dataclass(frozen=True)
class TrialTemporalRelevanceResult:
    """
    Trial-wise CSP+LDA temporal relevance for one subject.
    """

    values: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray

    @property
    def correct_mask(self) -> np.ndarray:
        """Return correctly classified trials."""
        return self.predictions == self.labels

    @property
    def incorrect_mask(self) -> np.ndarray:
        """Return incorrectly classified trials."""
        return self.predictions != self.labels


def compute_fold_trial_temporal_relevance(
    csp: CSP,
    lda: LinearDiscriminantAnalysis,
    data: np.ndarray,
    labels: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Compute trial-wise temporal relevance for one CSP+LDA fold.

    EEG epochs are projected into CSP component space.
    Instantaneous component power is then weighted by the
    absolute LDA coefficients corresponding to the
    ground-truth class of each evaluation trial.

    Parameters
    ----------
    csp : CSP
        Fitted CSP model.

    lda : LinearDiscriminantAnalysis
        Fitted LDA classifier.

    data : np.ndarray
        Evaluation EEG trials with shape
        (n_trials, n_channels, n_times).

    labels : np.ndarray
        Ground-truth labels with shape
        (n_trials,).

    Returns
    -------
    temporal_relevance : np.ndarray
        Trial-wise temporal relevance.
        Shape: (n_trials, n_times).

    probabilities : np.ndarray
        LDA class probabilities for each evaluation trial.
        Shape: (n_trials, n_classes).
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

    if labels.ndim != 1:
        labels = labels.reshape(-1)

    if labels.shape[0] != data.shape[0]:
        raise ValueError(
            "labels must contain one label per trial."
        )

    n_components = csp.n_components

    # ----------------------------------------------------------
    # CSP spatial filters
    # ----------------------------------------------------------

    filters = np.asarray(
        csp.filters_[:n_components],
        dtype=np.float64,
    )

    if filters.shape[1] != data.shape[1]:
        raise ValueError(
            "Number of CSP filter channels does not match "
            "the number of EEG channels."
        )

    # ----------------------------------------------------------
    # LDA coefficients
    # ----------------------------------------------------------

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

    classes = np.asarray(
        lda.classes_
    )

    if coefficients.shape[0] != len(classes):
        raise ValueError(
            "Number of LDA coefficient rows does not match "
            "the number of classes."
        )

    # ----------------------------------------------------------
    # Project EEG into CSP component space
    # ----------------------------------------------------------

    # data:
    # (n_trials, n_channels, n_times)
    #
    # filters:
    # (n_components, n_channels)
    #
    # csp_signals:
    # (n_trials, n_components, n_times)
    csp_signals = np.einsum(
        "kc,nct->nkt",
        filters,
        data,
    )

    # Instantaneous power of each CSP component:
    #
    # (n_trials, n_components, n_times)
    component_power = (
        csp_signals**2
    )

    # ----------------------------------------------------------
    # Trial-wise temporal relevance
    # ----------------------------------------------------------

    n_trials = data.shape[0]
    n_times = data.shape[2]

    temporal_relevance = np.zeros(
        (
            n_trials,
            n_times,
        ),
        dtype=np.float64,
    )

    for trial_index, label in enumerate(
        labels
    ):
        class_indices = np.flatnonzero(
            classes == label
        )

        if len(class_indices) != 1:
            raise ValueError(
                f"Label {label} was not found uniquely "
                "in lda.classes_."
            )

        class_index = class_indices[0]

        # Importance of each CSP component for the
        # ground-truth class of this trial.
        #
        # Shape:
        # (n_components,)
        component_weights = np.abs(
            coefficients[
                class_index
            ]
        )

        # Instantaneous component power for this trial:
        #
        # (n_components, n_times)
        trial_power = component_power[
            trial_index
        ]

        # Weighted sum across CSP components:
        #
        # (n_components,)
        # @
        # (n_components, n_times)
        #
        # ->
        # (n_times,)
        temporal_relevance[
            trial_index
        ] = (
            component_weights
            @ trial_power
        )

    # ----------------------------------------------------------
    # Fold predictions
    # ----------------------------------------------------------

    # Use the CSP features actually provided to LDA.
    classification_features = np.asarray(
        csp.transform(data),
        dtype=np.float64,
    )

    probabilities = lda.predict_proba(
        classification_features
    )

    return (
        temporal_relevance,
        probabilities,
    )


def compute_trial_temporal_relevance(
    csps: list[CSP],
    ldas: list[LinearDiscriminantAnalysis],
    data: np.ndarray,
    labels: np.ndarray,
) -> TrialTemporalRelevanceResult:
    """
    Compute ensemble trial-wise CSP+LDA temporal relevance.

    Temporal relevance and predicted class probabilities are
    computed independently for each fold model and then averaged
    across folds.

    Parameters
    ----------
    csps : list[CSP]
        CSP models from all folds.

    ldas : list[LinearDiscriminantAnalysis]
        Corresponding LDA models from all folds.

    data : np.ndarray
        Evaluation EEG trials with shape
        (n_trials, n_channels, n_times).

    labels : np.ndarray
        Ground-truth labels with shape
        (n_trials,).

    Returns
    -------
    result : TrialTemporalRelevanceResult
        Trial-wise temporal relevance, ensemble predictions,
        and ground-truth labels.
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

    if labels.ndim != 1:
        labels = labels.reshape(-1)

    if len(data) != len(labels):
        raise ValueError(
            "data and labels must contain the same "
            "number of trials."
        )

    reference_classes = np.asarray(
        ldas[0].classes_
    )

    fold_relevances = []
    fold_probabilities = []

    for csp, lda in zip(
        csps,
        ldas,
        strict=True,
    ):
        if not np.array_equal(
            lda.classes_,
            reference_classes,
        ):
            raise ValueError(
                "All LDA models must use the same "
                "class ordering."
            )

        (
            relevance,
            probabilities,
        ) = compute_fold_trial_temporal_relevance(
            csp=csp,
            lda=lda,
            data=data,
            labels=labels,
        )

        fold_relevances.append(
            relevance
        )

        fold_probabilities.append(
            probabilities
        )

    # ----------------------------------------------------------
    # Average relevance across folds
    # ----------------------------------------------------------

    mean_relevance = np.mean(
        np.stack(
            fold_relevances,
            axis=0,
        ),
        axis=0,
    )

    # ----------------------------------------------------------
    # Ensemble predictions
    # ----------------------------------------------------------

    mean_probabilities = np.mean(
        np.stack(
            fold_probabilities,
            axis=0,
        ),
        axis=0,
    )

    predictions = reference_classes[
        np.argmax(
            mean_probabilities,
            axis=1,
        )
    ]

    return TrialTemporalRelevanceResult(
        values=mean_relevance,
        predictions=predictions,
        labels=labels,
    )


def aggregate_trial_temporal_relevance(
    result: TrialTemporalRelevanceResult,
    mask: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Average trial-wise temporal relevance separately
    for each ground-truth motor imagery class.

    Parameters
    ----------
    result : TrialTemporalRelevanceResult
        Trial-wise temporal relevance.

    mask : np.ndarray
        Boolean trial-selection mask, such as
        result.correct_mask or result.incorrect_mask.

    Returns
    -------
    relevance : np.ndarray
        Mean temporal relevance for each ground-truth class.
        Shape: (n_classes, n_times).

    counts : np.ndarray
        Number of selected trials for each class.
        Shape: (n_classes,).
    """
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if mask.ndim != 1:
        mask = mask.reshape(-1)

    if len(mask) != len(result.labels):
        raise ValueError(
            "mask must contain one value per trial."
        )

    classes = np.unique(
        result.labels
    )

    n_times = result.values.shape[1]

    relevance = np.zeros(
        (
            len(classes),
            n_times,
        ),
        dtype=np.float64,
    )

    counts = np.zeros(
        len(classes),
        dtype=int,
    )

    for class_index, class_id in enumerate(
        classes
    ):
        class_mask = (
            (result.labels == class_id)
            & mask
        )

        counts[class_index] = np.sum(
            class_mask
        )

        if counts[class_index] == 0:
            relevance[class_index] = np.nan
            continue

        relevance[class_index] = np.mean(
            result.values[
                class_mask
            ],
            axis=0,
        )

    return (
        relevance,
        counts,
    )