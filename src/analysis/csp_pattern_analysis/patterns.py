from dataclasses import dataclass

import numpy as np
from mne.decoding import CSP
from scipy.optimize import linear_sum_assignment
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)


CSPFoldModel = tuple[
    CSP,
    LinearDiscriminantAnalysis,
]


@dataclass(frozen=True)
class CSPPatternResult:
    """
    CSP spatial patterns for one subject.

    Attributes
    ----------
    mean_patterns
        Mean aligned CSP patterns across folds, with shape
        (n_components, n_channels).

    fold_patterns
        Aligned CSP patterns for every fold, with shape
        (n_folds, n_components, n_channels).
    """

    mean_patterns: np.ndarray
    fold_patterns: np.ndarray


def compute_subject_csp_patterns(
    models: list[CSPFoldModel],
) -> CSPPatternResult:
    """
    Extract, align, and average CSP spatial patterns across folds.
    """
    fold_patterns = extract_csp_patterns(
        models
    )

    aligned_patterns = align_csp_patterns(
        fold_patterns
    )

    mean_patterns = compute_mean_csp_patterns(
        aligned_patterns
    )

    return CSPPatternResult(
        mean_patterns=mean_patterns,
        fold_patterns=aligned_patterns,
    )


def extract_csp_patterns(
    models: list[CSPFoldModel],
) -> np.ndarray:
    """
    Extract the CSP spatial patterns used by each fold model.

    Returns
    -------
    np.ndarray
        Array with shape:

        (n_folds, n_components, n_channels)
    """
    if not models:
        raise ValueError(
            "No CSP+LDA fold models were provided."
        )

    extracted_patterns = []

    expected_shape = None

    for fold_index, (csp, _) in enumerate(
        models
    ):
        if not hasattr(
            csp,
            "patterns_",
        ):
            raise ValueError(
                f"CSP model from fold {fold_index} "
                "has not been fitted."
            )

        patterns = np.asarray(
            csp.patterns_,
            dtype=float,
        )

        if patterns.ndim != 2:
            raise ValueError(
                "CSP patterns must have shape "
                "(components, channels)."
            )

        n_components = int(
            csp.n_components
        )

        if patterns.shape[0] < n_components:
            raise ValueError(
                f"Fold {fold_index} contains only "
                f"{patterns.shape[0]} CSP patterns, "
                f"but {n_components} components were requested."
            )

        # MNE may store a complete set of spatial patterns.
        # Only the components used during CSP transformation
        # are retained here.
        selected_patterns = patterns[
            :n_components
        ].copy()

        if expected_shape is None:
            expected_shape = (
                selected_patterns.shape
            )
        elif selected_patterns.shape != expected_shape:
            raise ValueError(
                "CSP pattern shapes differ between folds: "
                f"expected {expected_shape}, received "
                f"{selected_patterns.shape} in fold "
                f"{fold_index}."
            )

        extracted_patterns.append(
            selected_patterns
        )

    return np.stack(
        extracted_patterns,
        axis=0,
    )


def align_csp_patterns(
    fold_patterns: np.ndarray,
) -> np.ndarray:
    """
    Align CSP component order and signs across folds.

    The first fold is used as the reference. Components from every
    subsequent fold are matched to the reference components using
    absolute spatial correlation. Sign inversions are then corrected.

    Parameters
    ----------
    fold_patterns
        CSP patterns with shape:

        (n_folds, n_components, n_channels)

    Returns
    -------
    np.ndarray
        Aligned CSP patterns with the same shape as the input.
    """
    fold_patterns = np.asarray(
        fold_patterns,
        dtype=float,
    )

    if fold_patterns.ndim != 3:
        raise ValueError(
            "fold_patterns must have shape "
            "(n_folds, n_components, n_channels)."
        )

    n_folds = fold_patterns.shape[0]

    if n_folds == 0:
        raise ValueError(
            "At least one fold is required."
        )

    aligned_patterns = np.empty_like(
        fold_patterns
    )

    reference_patterns = fold_patterns[
        0
    ].copy()

    aligned_patterns[0] = (
        reference_patterns
    )

    normalized_reference = (
        _normalize_patterns(
            reference_patterns
        )
    )

    for fold_index in range(
        1,
        n_folds,
    ):
        current_patterns = fold_patterns[
            fold_index
        ]

        normalized_current = (
            _normalize_patterns(
                current_patterns
            )
        )

        correlation_matrix = (
            normalized_reference
            @ normalized_current.T
        )

        reference_indices, current_indices = (
            linear_sum_assignment(
                -np.abs(
                    correlation_matrix
                )
            )
        )

        aligned_fold = np.empty_like(
            current_patterns
        )

        for (
            reference_index,
            current_index,
        ) in zip(
            reference_indices,
            current_indices,
            strict=True,
        ):
            pattern = current_patterns[
                current_index
            ].copy()

            correlation = correlation_matrix[
                reference_index,
                current_index,
            ]

            if correlation < 0:
                pattern *= -1.0

            aligned_fold[
                reference_index
            ] = pattern

        aligned_patterns[
            fold_index
        ] = aligned_fold

    return aligned_patterns


def compute_mean_csp_patterns(
    aligned_patterns: np.ndarray,
) -> np.ndarray:
    """
    Compute the mean CSP spatial patterns across aligned folds.

    Parameters
    ----------
    aligned_patterns
        Aligned patterns with shape:

        (n_folds, n_components, n_channels)

    Returns
    -------
    np.ndarray
        Mean CSP patterns with shape:

        (n_components, n_channels)
    """
    aligned_patterns = np.asarray(
        aligned_patterns,
        dtype=float,
    )

    if aligned_patterns.ndim != 3:
        raise ValueError(
            "aligned_patterns must have shape "
            "(n_folds, n_components, n_channels)."
        )

    if aligned_patterns.shape[0] == 0:
        raise ValueError(
            "At least one fold is required "
            "to compute mean CSP patterns."
        )

    return np.mean(
        aligned_patterns,
        axis=0,
    )


def _normalize_patterns(
    patterns: np.ndarray,
) -> np.ndarray:
    """
    Standardize CSP patterns for spatial-correlation matching.

    Each component is centered and normalized independently.
    """
    patterns = np.asarray(
        patterns,
        dtype=float,
    )

    centered_patterns = (
        patterns
        - patterns.mean(
            axis=1,
            keepdims=True,
        )
    )

    norms = np.linalg.norm(
        centered_patterns,
        axis=1,
        keepdims=True,
    )

    if np.any(
        np.isclose(
            norms,
            0.0,
        )
    ):
        raise ValueError(
            "A CSP pattern has zero spatial variance "
            "and cannot be aligned."
        )

    return centered_patterns / norms