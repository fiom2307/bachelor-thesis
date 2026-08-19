from dataclasses import dataclass

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)


@dataclass(frozen=True)
class TrialFrequencyRelevanceResult:
    """
    Trial-wise CSP+LDA frequency relevance for one subject.
    """

    values: np.ndarray
    frequencies: np.ndarray
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


def compute_fold_trial_frequency_relevance(
    csp: CSP,
    lda: LinearDiscriminantAnalysis,
    data: np.ndarray,
    labels: np.ndarray,
    sfreq: float,
    fmin: float = 8.0,
    fmax: float = 30.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute trial-wise frequency relevance for one CSP+LDA fold.

    EEG epochs are projected into CSP component space.
    Spectral power is computed for each CSP component and
    weighted using the absolute LDA coefficients corresponding
    to the ground-truth class of each evaluation trial.

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

    sfreq : float
        Sampling frequency in Hz.

    fmin : float
        Minimum frequency of interest.

    fmax : float
        Maximum frequency of interest.

    Returns
    -------
    frequency_relevance : np.ndarray
        Trial-wise frequency relevance with shape
        (n_trials, n_frequencies).

    frequencies : np.ndarray
        Frequency bins retained between fmin and fmax.

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

    if sfreq <= 0:
        raise ValueError(
            "sfreq must be greater than zero."
        )

    if fmin < 0 or fmax <= fmin:
        raise ValueError(
            "Invalid frequency range."
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

    n_times = csp_signals.shape[-1]

    # ----------------------------------------------------------
    # Frequency decomposition
    # ----------------------------------------------------------

    # Apply a Hann window to reduce spectral leakage.
    window = np.hanning(
        n_times
    )

    windowed_signals = (
        csp_signals
        * window[
            np.newaxis,
            np.newaxis,
            :
        ]
    )

    spectrum = np.fft.rfft(
        windowed_signals,
        axis=-1,
    )

    frequencies = np.fft.rfftfreq(
        n_times,
        d=1.0 / sfreq,
    )

    # Spectral power:
    # (n_trials, n_components, n_frequencies)
    spectral_power = (
        np.abs(spectrum) ** 2
    )

    # Keep only frequencies within the classification band.
    frequency_mask = (
        (frequencies >= fmin)
        & (frequencies <= fmax)
    )

    frequencies = frequencies[
        frequency_mask
    ]

    spectral_power = spectral_power[
        ...,
        frequency_mask
    ]

    # ----------------------------------------------------------
    # Trial-wise frequency relevance
    # ----------------------------------------------------------

    n_trials = data.shape[0]
    n_frequencies = len(frequencies)

    frequency_relevance = np.zeros(
        (
            n_trials,
            n_frequencies,
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

        # Trial spectral power:
        # (n_components, n_frequencies)
        trial_power = spectral_power[
            trial_index
        ]

        # Weighted sum across CSP components:
        #
        # (n_components,)
        # @
        # (n_components, n_frequencies)
        #
        # ->
        # (n_frequencies,)
        frequency_relevance[
            trial_index
        ] = (
            component_weights
            @ trial_power
        )

    # ----------------------------------------------------------
    # Fold predictions
    # ----------------------------------------------------------

    # Use the actual CSP features employed by the classifier.
    classification_features = np.asarray(
        csp.transform(data),
        dtype=np.float64,
    )

    probabilities = lda.predict_proba(
        classification_features
    )

    return (
        frequency_relevance,
        frequencies,
        probabilities,
    )


def compute_trial_frequency_relevance(
    csps: list[CSP],
    ldas: list[LinearDiscriminantAnalysis],
    data: np.ndarray,
    labels: np.ndarray,
    sfreq: float,
    fmin: float = 8.0,
    fmax: float = 30.0,
) -> TrialFrequencyRelevanceResult:
    """
    Compute ensemble trial-wise CSP+LDA frequency relevance.

    Frequency relevance is computed independently for each
    fold model and evaluation trial. Relevance values and
    predicted class probabilities are then averaged across
    folds.

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

    sfreq : float
        Sampling frequency in Hz.

    fmin : float
        Minimum frequency of interest.

    fmax : float
        Maximum frequency of interest.

    Returns
    -------
    result : TrialFrequencyRelevanceResult
        Trial-wise frequency relevance, frequency bins,
        ensemble predictions, and ground-truth labels.
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

    reference_frequencies = None

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
            frequencies,
            probabilities,
        ) = compute_fold_trial_frequency_relevance(
            csp=csp,
            lda=lda,
            data=data,
            labels=labels,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
        )

        if reference_frequencies is None:
            reference_frequencies = frequencies

        elif not np.allclose(
            reference_frequencies,
            frequencies,
        ):
            raise ValueError(
                "Frequency bins differ between folds."
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

    return TrialFrequencyRelevanceResult(
        values=mean_relevance,
        frequencies=reference_frequencies,
        predictions=predictions,
        labels=labels,
    )


def aggregate_trial_frequency_relevance(
    result: TrialFrequencyRelevanceResult,
    mask: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Average trial-wise frequency relevance separately
    for each ground-truth motor imagery class.

    Parameters
    ----------
    result : TrialFrequencyRelevanceResult
        Trial-wise frequency relevance.

    mask : np.ndarray
        Boolean trial-selection mask, such as
        result.correct_mask or result.incorrect_mask.

    Returns
    -------
    relevance : np.ndarray
        Mean frequency relevance for each ground-truth class.
        Shape: (n_classes, n_frequencies).

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

    n_frequencies = result.values.shape[1]

    relevance = np.zeros(
        (
            len(classes),
            n_frequencies,
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