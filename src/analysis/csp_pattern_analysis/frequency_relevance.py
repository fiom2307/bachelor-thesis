import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)


def compute_fold_frequency_relevance(
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
]:
    """
    Compute class-wise frequency relevance for one CSP+LDA fold.

    EEG epochs are projected into CSP component space.
    The spectral power of each CSP component is then weighted
    by the class-specific absolute LDA coefficients.
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
    # csp_signals:
    # (trials, components, times)
    csp_signals = np.einsum(
        "kc,nct->nkt",
        filters,
        data,
    )

    n_times = csp_signals.shape[-1]

    # ----------------------------------------------------------
    # Frequency decomposition
    # ----------------------------------------------------------

    # Hann window reduces spectral leakage.
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
    # (trials, components, frequencies)
    spectral_power = (
        np.abs(spectrum) ** 2
    )

    # Keep only the frequencies relevant for the
    # classification preprocessing.
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
    # Class-wise frequency relevance
    # ----------------------------------------------------------

    n_classes = coefficients.shape[0]
    n_frequencies = len(frequencies)

    frequency_relevance = np.zeros(
        (
            n_classes,
            n_frequencies,
        ),
        dtype=np.float64,
    )

    for class_idx in range(n_classes):
        class_mask = (
            labels == class_idx
        )

        if not np.any(class_mask):
            continue

        # Average spectral power over trials belonging
        # to the current class.
        #
        # Shape:
        # (components, frequencies)
        mean_component_power = np.mean(
            spectral_power[
                class_mask
            ],
            axis=0,
        )

        # Importance of each CSP component for this class.
        #
        # Shape:
        # (components,)
        component_weights = np.abs(
            coefficients[
                class_idx
            ]
        )

        # Weighted sum across CSP components:
        #
        # (components,)
        # @
        # (components, frequencies)
        #
        # ->
        # (frequencies,)
        frequency_relevance[
            class_idx
        ] = (
            component_weights
            @ mean_component_power
        )

        # Normalize within the class.
        total = np.sum(
            frequency_relevance[
                class_idx
            ]
        )

        if total > 0:
            frequency_relevance[
                class_idx
            ] /= total

    return (
        frequency_relevance,
        frequencies,
    )


def compute_frequency_relevance(
    csps: list[CSP],
    ldas: list[
        LinearDiscriminantAnalysis
    ],
    data: np.ndarray,
    labels: np.ndarray,
    sfreq: float,
    fmin: float = 8.0,
    fmax: float = 30.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Compute mean class-wise frequency relevance
    across CSP+LDA folds.
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
    reference_frequencies = None

    for csp, lda in zip(
        csps,
        ldas,
    ):
        (
            relevance,
            frequencies,
        ) = compute_fold_frequency_relevance(
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

    fold_relevances = np.stack(
        fold_relevances,
        axis=0,
    )

    # Shape before averaging:
    #
    # (
    #     n_folds,
    #     n_classes,
    #     n_frequencies,
    # )
    frequency_relevance = np.mean(
        fold_relevances,
        axis=0,
    )

    # Normalize each class after averaging folds.
    class_sums = np.sum(
        frequency_relevance,
        axis=1,
        keepdims=True,
    )

    class_sums = np.maximum(
        class_sums,
        np.finfo(np.float64).eps,
    )

    frequency_relevance = (
        frequency_relevance
        / class_sums
    )

    return (
        frequency_relevance,
        reference_frequencies,
    )