import joblib
import numpy as np

from src.analysis.csp_pattern_analysis.channel_relevance import (
    compute_channel_relevance,
)
from src.utils.paths import (
    get_csp_channel_rankings_path,
    get_csp_channel_relevance_path,
    get_csp_fold_model_path,
    get_lda_fold_model_path,
    get_subject_name,
)
from src.visualization.csp_pattern_plots import (
    plot_csp_channel_rankings,
    plot_csp_channel_relevance,
)


SUBJECTS = range(1, 10)

N_FOLDS = 5

CHANNEL_NAMES = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]

CLASS_NAMES = [
    "Left hand",
    "Right hand",
    "Feet",
    "Tongue",
]


def _get_fold_numbers(
    subject: int,
) -> list[int]:
    """
    Determine whether saved folds are numbered 0-4 or 1-5.
    """
    possible_fold_sets = [
        list(range(1, N_FOLDS + 1)),
        list(range(N_FOLDS)),
    ]

    for folds in possible_fold_sets:
        models_exist = all(
            get_csp_fold_model_path(
                subject,
                fold,
            ).exists()
            and get_lda_fold_model_path(
                subject,
                fold,
            ).exists()
            for fold in folds
        )

        if models_exist:
            return folds

    subject_name = get_subject_name(
        subject
    )

    raise FileNotFoundError(
        "Could not find all CSP+LDA fold models "
        f"for {subject_name}."
    )


def _load_subject_models(
    subject: int,
):
    """
    Load all CSP and LDA fold models for one subject.
    """
    csps = []
    ldas = []

    folds = _get_fold_numbers(
        subject
    )

    for fold in folds:
        csp_path = get_csp_fold_model_path(
            subject,
            fold,
        )

        lda_path = get_lda_fold_model_path(
            subject,
            fold,
        )

        csp = joblib.load(
            csp_path
        )

        lda = joblib.load(
            lda_path
        )

        csps.append(csp)
        ldas.append(lda)

    return csps, ldas


def plot_subject_analysis(
    subject: int,
) -> np.ndarray:
    """
    Compute and plot CSP+LDA channel relevance for one subject.

    Returns
    -------
    channel_relevance : np.ndarray
        Class-wise channel relevance with shape
        (n_classes, n_channels).
    """
    subject_name = get_subject_name(
        subject
    )

    print(
        f"Computing CSP pattern analysis for "
        f"{subject_name}..."
    )

    csps, ldas = _load_subject_models(
        subject
    )

    channel_relevance = (
        compute_channel_relevance(
            csps=csps,
            ldas=ldas,
        )
    )

    expected_shape = (
        len(CLASS_NAMES),
        len(CHANNEL_NAMES),
    )

    if channel_relevance.shape != expected_shape:
        raise ValueError(
            "Expected class-wise channel relevance "
            f"with shape {expected_shape}, but received "
            f"{channel_relevance.shape}."
        )

    relevance_path = (
        get_csp_channel_relevance_path(
            subject
        )
    )

    plot_csp_channel_relevance(
        channel_relevance=channel_relevance,
        channel_names=CHANNEL_NAMES,
        class_names=CLASS_NAMES,
        subject=subject_name,
        output_dir=relevance_path.parent,
    )

    rankings_path = (
        get_csp_channel_rankings_path(
            subject
        )
    )

    plot_csp_channel_rankings(
        channel_relevance=channel_relevance,
        channel_names=CHANNEL_NAMES,
        class_names=CLASS_NAMES,
        subject=subject_name,
        output_dir=rankings_path.parent,
        top_n=10,
    )

    return channel_relevance


def plot_global_analysis(
    subject_relevances: list[np.ndarray],
) -> None:
    """
    Plot mean CSP+LDA channel relevance across all subjects.
    """
    mean_channel_relevance = np.mean(
        np.stack(
            subject_relevances,
            axis=0,
        ),
        axis=0,
    )

    relevance_path = (
        get_csp_channel_relevance_path(
            None
        )
    )

    plot_csp_channel_relevance(
        channel_relevance=mean_channel_relevance,
        channel_names=CHANNEL_NAMES,
        class_names=CLASS_NAMES,
        subject="all_mean",
        output_dir=relevance_path.parent,
    )

    rankings_path = (
        get_csp_channel_rankings_path(
            None
        )
    )

    plot_csp_channel_rankings(
        channel_relevance=mean_channel_relevance,
        channel_names=CHANNEL_NAMES,
        class_names=CLASS_NAMES,
        subject="all_mean",
        output_dir=rankings_path.parent,
        top_n=10,
    )


def main() -> None:
    """
    Generate subject-wise and global CSP pattern analysis plots.
    """
    subject_relevances = []

    for subject in SUBJECTS:
        relevance = plot_subject_analysis(
            subject
        )

        subject_relevances.append(
            relevance
        )

    print(
        "Computing global mean CSP pattern analysis..."
    )

    plot_global_analysis(
        subject_relevances
    )

    print(
        "CSP pattern analysis completed."
    )


if __name__ == "__main__":
    main()