from dataclasses import dataclass

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)

from src.analysis.csp_lda_occlusion_analysis._utils import (
    compute_model_occlusion,
)
from src.data.labels import CLASS_LABELS


CSPFoldModel = tuple[
    CSP,
    LinearDiscriminantAnalysis,
]


@dataclass(frozen=True)
class CSPAnalysisResult:
    """
    CSP+LDA occlusion results for one subject.
    """

    values: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray
    times: np.ndarray

    @property
    def correct_mask(self) -> np.ndarray:
        """Return correctly classified trials."""
        return self.predictions == self.labels

    @property
    def incorrect_mask(self) -> np.ndarray:
        """Return incorrectly classified trials."""
        return self.predictions != self.labels


def compute_occlusion_reference(
    data: np.ndarray,
) -> np.ndarray:
    """
    Compute the mean training epoch used for occlusion.
    """
    return np.mean(
        data,
        axis=0,
    )


def compute_csp_lda_ensemble_occlusion(
    models: list[CSPFoldModel],
    reference: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    sfreq: float,
    tmin: float,
    window_duration: float = 0.5,
    window_step: float = 0.1,
) -> CSPAnalysisResult:
    """
    Compute and average occlusion relevance across CSP+LDA folds.
    """
    if not models:
        raise ValueError(
            "No CSP+LDA fold models were provided."
        )

    fold_values = []
    fold_probabilities = []

    window_times = None

    for csp, lda in models:
        values, probabilities, times = (
            compute_model_occlusion(
                csp=csp,
                lda=lda,
                reference=reference,
                data=data,
                labels=labels,
                sfreq=sfreq,
                tmin=tmin,
                window_duration=window_duration,
                window_step=window_step,
            )
        )

        fold_values.append(
            values
        )

        fold_probabilities.append(
            probabilities
        )

        if window_times is None:
            window_times = times
        elif not np.array_equal(
            window_times,
            times,
        ):
            raise ValueError(
                "Occlusion-window times differ "
                "between CSP+LDA folds."
            )

    mean_values = np.mean(
        fold_values,
        axis=0,
    )

    mean_probabilities = np.mean(
        fold_probabilities,
        axis=0,
    )

    class_labels = np.asarray(
        CLASS_LABELS
    )

    predictions = class_labels[
        mean_probabilities.argmax(axis=1)
    ]

    if window_times is None:
        raise ValueError(
            "Occlusion-window times were not computed."
        )

    return CSPAnalysisResult(
        values=mean_values,
        probabilities=mean_probabilities,
        predictions=predictions,
        labels=np.asarray(
            labels
        ),
        times=window_times,
    )