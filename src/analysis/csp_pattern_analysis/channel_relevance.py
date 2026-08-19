import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)
from dataclasses import dataclass


@dataclass(frozen=True)
class TrialChannelRelevanceResult:
    """
    Trial-wise CSP+LDA channel relevance for one subject.
    """

    values: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray

    @property
    def correct_mask(self) -> np.ndarray:
        return self.predictions == self.labels

    @property
    def incorrect_mask(self) -> np.ndarray:
        return self.predictions != self.labels


def compute_trial_channel_relevance(
    csps: list[CSP],
    ldas: list[LinearDiscriminantAnalysis],
    data: np.ndarray,
    labels: np.ndarray,
) -> TrialChannelRelevanceResult:
    """
    Compute trial-wise CSP+LDA spatial relevance.

    The same CSP spatial patterns and LDA coefficients used in
    the model-level channel relevance are retained, while the
    contribution of each CSP component is additionally weighted
    by its activation in each evaluation trial.
    """
    data = np.asarray(
        data,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    fold_relevances = []
    fold_probabilities = []

    for csp, lda in zip(
        csps,
        ldas,
        strict=True,
    ):
        n_components = csp.n_components

        # ------------------------------------------------------
        # CSP spatial patterns
        # Same computation as in your original analysis
        # ------------------------------------------------------

        patterns = np.asarray(
            csp.patterns_[:n_components],
            dtype=np.float64,
        )

        pattern_relevance = patterns**2

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

        # ------------------------------------------------------
        # Trial CSP features
        # ------------------------------------------------------

        features = np.asarray(
            csp.transform(data),
            dtype=np.float64,
        )

        # ------------------------------------------------------
        # LDA coefficients
        # ------------------------------------------------------

        coefficients = np.asarray(
            lda.coef_,
            dtype=np.float64,
        )

        if coefficients.ndim == 1:
            coefficients = coefficients[
                np.newaxis,
                :
            ]

        classes = np.asarray(
            lda.classes_
        )

        trial_relevance = np.zeros(
            (
                len(data),
                data.shape[1],
            ),
            dtype=np.float64,
        )

        # ------------------------------------------------------
        # Compute relevance for true class of each trial
        # ------------------------------------------------------

        for trial_index, label in enumerate(
            labels
        ):
            class_index = np.flatnonzero(
                classes == label
            )[0]

            component_relevance = np.abs(
                features[trial_index]
                * coefficients[class_index]
            )

            trial_relevance[
                trial_index
            ] = (
                component_relevance
                @ pattern_relevance
            )

        fold_relevances.append(
            trial_relevance
        )

        fold_probabilities.append(
            lda.predict_proba(
                features
            )
        )

    # ----------------------------------------------------------
    # Average across folds
    # ----------------------------------------------------------

    mean_relevance = np.mean(
        np.stack(
            fold_relevances,
            axis=0,
        ),
        axis=0,
    )

    mean_probabilities = np.mean(
        np.stack(
            fold_probabilities,
            axis=0,
        ),
        axis=0,
    )

    classes = np.asarray(
        ldas[0].classes_
    )

    predictions = classes[
        np.argmax(
            mean_probabilities,
            axis=1,
        )
    ]

    return TrialChannelRelevanceResult(
        values=mean_relevance,
        predictions=predictions,
        labels=labels,
    )


def aggregate_trial_channel_relevance(
    result: TrialChannelRelevanceResult,
    mask: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Average trial relevance separately for each true class.
    """
    classes = np.unique(
        result.labels
    )

    n_channels = result.values.shape[1]

    relevance = np.zeros(
        (
            len(classes),
            n_channels,
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
            result.values[class_mask],
            axis=0,
        )

    return relevance, counts