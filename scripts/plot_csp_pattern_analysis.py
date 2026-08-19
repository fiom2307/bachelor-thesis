import joblib
import numpy as np

from src.analysis.csp_pattern_analysis.channel_relevance import (
    aggregate_trial_channel_relevance,
    compute_trial_channel_relevance,
)
from src.analysis.csp_pattern_analysis.frequency_relevance import (
    aggregate_trial_frequency_relevance,
    compute_trial_frequency_relevance,
)
from src.analysis.csp_pattern_analysis.temporal_relevance import (
    aggregate_trial_temporal_relevance,
    compute_trial_temporal_relevance,
)
from src.data.dataset import get_data_for_subject
from src.utils.paths import (
    get_csp_channel_rankings_path,
    get_csp_channel_relevance_path,
    get_csp_fold_model_path,
    get_csp_frequency_relevance_path,
    get_csp_temporal_relevance_path,
    get_csp_topographies_path,
    get_lda_fold_model_path,
    get_subject_name,
)
from src.visualization.csp_pattern_plots import (
    plot_csp_channel_rankings,
    plot_csp_channel_relevance,
    plot_csp_frequency_relevance,
    plot_csp_temporal_relevance,
    plot_csp_topographies,
)


SUBJECTS = range(1, 10)

N_FOLDS = 5

SFREQ = 250.0
TMIN = 0.5

FMIN = 8.0
FMAX = 30.0


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

        csps.append(
            csp
        )

        ldas.append(
            lda
        )

    return csps, ldas


def _load_subject_evaluation_data(
    subject: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Load the preprocessed evaluation epochs and labels
    for one subject.
    """
    subject_data = get_data_for_subject(
        subject
    )

    if subject_data is None:
        raise FileNotFoundError(
            "Could not load data for "
            f"{get_subject_name(subject)}."
        )

    (
        _,
        _,
        X_eval,
        y_eval,
    ) = subject_data

    X_eval = np.asarray(
        X_eval,
        dtype=np.float64,
    )

    y_eval = np.asarray(
        y_eval,
        dtype=int,
    )

    if X_eval.ndim != 3:
        raise ValueError(
            "Expected X_eval with shape "
            "(n_trials, n_channels, n_times), "
            f"but received {X_eval.shape}."
        )

    if y_eval.ndim != 1:
        y_eval = y_eval.reshape(-1)

    if X_eval.shape[0] != y_eval.shape[0]:
        raise ValueError(
            "Number of evaluation trials does not "
            "match number of evaluation labels."
        )

    return (
        X_eval,
        y_eval,
    )


def _create_times(
    n_times: int,
) -> np.ndarray:
    """
    Create the time axis for the classification epoch.
    """
    return (
        np.arange(
            n_times,
            dtype=np.float64,
        )
        / SFREQ
        + TMIN
    )


def _plot_spatial_relevance(
    channel_relevance: np.ndarray,
    subject_name: str,
    selection: str,
    subject: int | None,
) -> None:
    """
    Plot channel relevance, channel rankings, and
    scalp topographies for one trial selection.
    """
    expected_shape = (
        len(CLASS_NAMES),
        len(CHANNEL_NAMES),
    )

    if (
        channel_relevance.shape
        != expected_shape
    ):
        raise ValueError(
            "Expected class-wise channel relevance "
            f"with shape {expected_shape}, "
            f"but received "
            f"{channel_relevance.shape}."
        )

    plot_subject = (
        f"{subject_name}_{selection}"
    )

    # ==========================================================
    # CHANNEL RELEVANCE
    # ==========================================================

    channel_relevance_path = (
        get_csp_channel_relevance_path(
            subject
        )
    )

    plot_csp_channel_relevance(
        channel_relevance=channel_relevance,
        channel_names=CHANNEL_NAMES,
        class_names=CLASS_NAMES,
        subject=plot_subject,
        output_dir=(
            channel_relevance_path.parent
        ),
    )

    # ==========================================================
    # CHANNEL RANKINGS
    # ==========================================================

    channel_rankings_path = (
        get_csp_channel_rankings_path(
            subject
        )
    )

    plot_csp_channel_rankings(
        channel_relevance=channel_relevance,
        channel_names=CHANNEL_NAMES,
        class_names=CLASS_NAMES,
        subject=plot_subject,
        output_dir=(
            channel_rankings_path.parent
        ),
        top_n=10,
    )

    # ==========================================================
    # TOPOGRAPHIES
    # ==========================================================

    topographies_path = (
        get_csp_topographies_path(
            subject
        )
    )

    plot_csp_topographies(
        channel_relevance=channel_relevance,
        channel_names=CHANNEL_NAMES,
        class_names=CLASS_NAMES,
        subject=plot_subject,
        output_dir=(
            topographies_path.parent
        ),
        sfreq=SFREQ,
    )


def _plot_temporal_relevance(
    temporal_relevance: np.ndarray,
    times: np.ndarray,
    subject_name: str,
    selection: str,
    subject: int | None,
) -> None:
    """
    Plot temporal relevance for one trial selection.
    """
    expected_shape = (
        len(CLASS_NAMES),
        len(times),
    )

    if (
        temporal_relevance.shape
        != expected_shape
    ):
        raise ValueError(
            "Expected class-wise temporal relevance "
            f"with shape {expected_shape}, "
            f"but received "
            f"{temporal_relevance.shape}."
        )

    temporal_relevance_path = (
        get_csp_temporal_relevance_path(
            subject
        )
    )

    plot_csp_temporal_relevance(
        temporal_relevance=temporal_relevance,
        times=times,
        class_names=CLASS_NAMES,
        subject=(
            f"{subject_name}_{selection}"
        ),
        output_dir=(
            temporal_relevance_path.parent
        ),
    )


def _plot_frequency_relevance(
    frequency_relevance: np.ndarray,
    frequencies: np.ndarray,
    subject_name: str,
    selection: str,
    subject: int | None,
) -> None:
    """
    Plot frequency relevance for one trial selection.
    """
    expected_shape = (
        len(CLASS_NAMES),
        len(frequencies),
    )

    if (
        frequency_relevance.shape
        != expected_shape
    ):
        raise ValueError(
            "Expected class-wise frequency relevance "
            f"with shape {expected_shape}, "
            f"but received "
            f"{frequency_relevance.shape}."
        )

    frequency_relevance_path = (
        get_csp_frequency_relevance_path(
            subject
        )
    )

    plot_csp_frequency_relevance(
        frequency_relevance=frequency_relevance,
        frequencies=frequencies,
        class_names=CLASS_NAMES,
        subject=(
            f"{subject_name}_{selection}"
        ),
        output_dir=(
            frequency_relevance_path.parent
        ),
    )


def plot_subject_analysis(
    subject: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute and plot CSP+LDA relevance for one subject.
    """
    subject_name = get_subject_name(
        subject
    )

    print(
        f"Computing CSP pattern analysis for "
        f"{subject_name}..."
    )

    # ==========================================================
    # LOAD CSP+LDA MODELS
    # ==========================================================

    csps, ldas = _load_subject_models(
        subject
    )

    # ==========================================================
    # LOAD EVALUATION DATA
    # ==========================================================

    X_eval, y_eval = (
        _load_subject_evaluation_data(
            subject
        )
    )

    times = _create_times(
        X_eval.shape[2]
    )

    # ==========================================================
    # TRIAL-WISE SPATIAL RELEVANCE
    # ==========================================================

    trial_channel_result = (
        compute_trial_channel_relevance(
            csps=csps,
            ldas=ldas,
            data=X_eval,
            labels=y_eval,
        )
    )

    (
        correct_channel_relevance,
        correct_counts,
    ) = aggregate_trial_channel_relevance(
        result=trial_channel_result,
        mask=trial_channel_result.correct_mask,
    )

    (
        incorrect_channel_relevance,
        incorrect_counts,
    ) = aggregate_trial_channel_relevance(
        result=trial_channel_result,
        mask=trial_channel_result.incorrect_mask,
    )

    print(
        f"{subject_name} correct counts: "
        f"{correct_counts}"
    )

    print(
        f"{subject_name} incorrect counts: "
        f"{incorrect_counts}"
    )

    # ==========================================================
    # CORRECT SPATIAL RELEVANCE
    # ==========================================================

    _plot_spatial_relevance(
        channel_relevance=(
            correct_channel_relevance
        ),
        subject_name=subject_name,
        selection="correct",
        subject=subject,
    )

    # ==========================================================
    # INCORRECT SPATIAL RELEVANCE
    # ==========================================================

    _plot_spatial_relevance(
        channel_relevance=(
            incorrect_channel_relevance
        ),
        subject_name=subject_name,
        selection="incorrect",
        subject=subject,
    )

    # ==========================================================
    # TRIAL-WISE TEMPORAL RELEVANCE
    # ==========================================================

    trial_temporal_result = (
        compute_trial_temporal_relevance(
            csps=csps,
            ldas=ldas,
            data=X_eval,
            labels=y_eval,
        )
    )

    (
        correct_temporal_relevance,
        correct_temporal_counts,
    ) = aggregate_trial_temporal_relevance(
        result=trial_temporal_result,
        mask=trial_temporal_result.correct_mask,
    )

    (
        incorrect_temporal_relevance,
        incorrect_temporal_counts,
    ) = aggregate_trial_temporal_relevance(
        result=trial_temporal_result,
        mask=trial_temporal_result.incorrect_mask,
    )

    # ==========================================================
    # CHECK TEMPORAL COUNTS
    # ==========================================================

    if not np.array_equal(
        correct_counts,
        correct_temporal_counts,
    ):
        raise ValueError(
            "Correct trial counts differ between "
            "spatial and temporal relevance."
        )

    if not np.array_equal(
        incorrect_counts,
        incorrect_temporal_counts,
    ):
        raise ValueError(
            "Incorrect trial counts differ between "
            "spatial and temporal relevance."
        )

    # ==========================================================
    # CORRECT TEMPORAL RELEVANCE
    # ==========================================================

    _plot_temporal_relevance(
        temporal_relevance=(
            correct_temporal_relevance
        ),
        times=times,
        subject_name=subject_name,
        selection="correct",
        subject=subject,
    )

    # ==========================================================
    # INCORRECT TEMPORAL RELEVANCE
    # ==========================================================

    _plot_temporal_relevance(
        temporal_relevance=(
            incorrect_temporal_relevance
        ),
        times=times,
        subject_name=subject_name,
        selection="incorrect",
        subject=subject,
    )

    # ==========================================================
    # TRIAL-WISE FREQUENCY RELEVANCE
    # ==========================================================

    trial_frequency_result = (
        compute_trial_frequency_relevance(
            csps=csps,
            ldas=ldas,
            data=X_eval,
            labels=y_eval,
            sfreq=SFREQ,
            fmin=FMIN,
            fmax=FMAX,
        )
    )

    (
        correct_frequency_relevance,
        correct_frequency_counts,
    ) = aggregate_trial_frequency_relevance(
        result=trial_frequency_result,
        mask=trial_frequency_result.correct_mask,
    )

    (
        incorrect_frequency_relevance,
        incorrect_frequency_counts,
    ) = aggregate_trial_frequency_relevance(
        result=trial_frequency_result,
        mask=trial_frequency_result.incorrect_mask,
    )

    frequencies = (
        trial_frequency_result.frequencies
    )

    # ==========================================================
    # CHECK FREQUENCY COUNTS
    # ==========================================================

    if not np.array_equal(
        correct_counts,
        correct_frequency_counts,
    ):
        raise ValueError(
            "Correct trial counts differ between "
            "spatial and frequency relevance."
        )

    if not np.array_equal(
        incorrect_counts,
        incorrect_frequency_counts,
    ):
        raise ValueError(
            "Incorrect trial counts differ between "
            "spatial and frequency relevance."
        )

    # ==========================================================
    # CORRECT FREQUENCY RELEVANCE
    # ==========================================================

    _plot_frequency_relevance(
        frequency_relevance=(
            correct_frequency_relevance
        ),
        frequencies=frequencies,
        subject_name=subject_name,
        selection="correct",
        subject=subject,
    )

    # ==========================================================
    # INCORRECT FREQUENCY RELEVANCE
    # ==========================================================

    _plot_frequency_relevance(
        frequency_relevance=(
            incorrect_frequency_relevance
        ),
        frequencies=frequencies,
        subject_name=subject_name,
        selection="incorrect",
        subject=subject,
    )

    return (
        correct_channel_relevance,
        incorrect_channel_relevance,
        correct_temporal_relevance,
        incorrect_temporal_relevance,
        correct_frequency_relevance,
        incorrect_frequency_relevance,
        frequencies,
    )


def plot_global_analysis(
    subject_correct_channel_relevances: list[
        np.ndarray
    ],
    subject_incorrect_channel_relevances: list[
        np.ndarray
    ],
    subject_correct_temporal_relevances: list[
        np.ndarray
    ],
    subject_incorrect_temporal_relevances: list[
        np.ndarray
    ],
    subject_correct_frequency_relevances: list[
        np.ndarray
    ],
    subject_incorrect_frequency_relevances: list[
        np.ndarray
    ],
    subject_frequencies: list[
        np.ndarray
    ],
) -> None:
    """
    Plot mean CSP+LDA relevance across all subjects.
    """

    # ==========================================================
    # GLOBAL CORRECT SPATIAL RELEVANCE
    # ==========================================================

    mean_correct_channel_relevance = np.nanmean(
        np.stack(
            subject_correct_channel_relevances,
            axis=0,
        ),
        axis=0,
    )

    _plot_spatial_relevance(
        channel_relevance=(
            mean_correct_channel_relevance
        ),
        subject_name="all_mean",
        selection="correct",
        subject=None,
    )

    # ==========================================================
    # GLOBAL INCORRECT SPATIAL RELEVANCE
    # ==========================================================

    mean_incorrect_channel_relevance = np.nanmean(
        np.stack(
            subject_incorrect_channel_relevances,
            axis=0,
        ),
        axis=0,
    )

    _plot_spatial_relevance(
        channel_relevance=(
            mean_incorrect_channel_relevance
        ),
        subject_name="all_mean",
        selection="incorrect",
        subject=None,
    )

    # ==========================================================
    # CHECK TEMPORAL SHAPES
    # ==========================================================

    correct_temporal_shapes = {
        relevance.shape
        for relevance
        in subject_correct_temporal_relevances
    }

    if len(correct_temporal_shapes) != 1:
        raise ValueError(
            "All subjects must have correct temporal "
            "relevance arrays with the same shape."
        )

    incorrect_temporal_shapes = {
        relevance.shape
        for relevance
        in subject_incorrect_temporal_relevances
    }

    if len(incorrect_temporal_shapes) != 1:
        raise ValueError(
            "All subjects must have incorrect temporal "
            "relevance arrays with the same shape."
        )

    # ==========================================================
    # GLOBAL CORRECT TEMPORAL RELEVANCE
    # ==========================================================

    mean_correct_temporal_relevance = np.nanmean(
        np.stack(
            subject_correct_temporal_relevances,
            axis=0,
        ),
        axis=0,
    )

    times = _create_times(
        mean_correct_temporal_relevance.shape[1]
    )

    _plot_temporal_relevance(
        temporal_relevance=(
            mean_correct_temporal_relevance
        ),
        times=times,
        subject_name="all_mean",
        selection="correct",
        subject=None,
    )

    # ==========================================================
    # GLOBAL INCORRECT TEMPORAL RELEVANCE
    # ==========================================================

    mean_incorrect_temporal_relevance = np.nanmean(
        np.stack(
            subject_incorrect_temporal_relevances,
            axis=0,
        ),
        axis=0,
    )

    _plot_temporal_relevance(
        temporal_relevance=(
            mean_incorrect_temporal_relevance
        ),
        times=times,
        subject_name="all_mean",
        selection="incorrect",
        subject=None,
    )

    # ==========================================================
    # CHECK FREQUENCY BINS
    # ==========================================================

    reference_frequencies = (
        subject_frequencies[0]
    )

    for frequencies in subject_frequencies[1:]:
        if not np.allclose(
            reference_frequencies,
            frequencies,
        ):
            raise ValueError(
                "Frequency bins differ between subjects."
            )

    # ==========================================================
    # CHECK FREQUENCY SHAPES
    # ==========================================================

    correct_frequency_shapes = {
        relevance.shape
        for relevance
        in subject_correct_frequency_relevances
    }

    if len(correct_frequency_shapes) != 1:
        raise ValueError(
            "All subjects must have correct frequency "
            "relevance arrays with the same shape."
        )

    incorrect_frequency_shapes = {
        relevance.shape
        for relevance
        in subject_incorrect_frequency_relevances
    }

    if len(incorrect_frequency_shapes) != 1:
        raise ValueError(
            "All subjects must have incorrect frequency "
            "relevance arrays with the same shape."
        )

    # ==========================================================
    # GLOBAL CORRECT FREQUENCY RELEVANCE
    # ==========================================================

    mean_correct_frequency_relevance = np.nanmean(
        np.stack(
            subject_correct_frequency_relevances,
            axis=0,
        ),
        axis=0,
    )

    _plot_frequency_relevance(
        frequency_relevance=(
            mean_correct_frequency_relevance
        ),
        frequencies=reference_frequencies,
        subject_name="all_mean",
        selection="correct",
        subject=None,
    )

    # ==========================================================
    # GLOBAL INCORRECT FREQUENCY RELEVANCE
    # ==========================================================

    mean_incorrect_frequency_relevance = np.nanmean(
        np.stack(
            subject_incorrect_frequency_relevances,
            axis=0,
        ),
        axis=0,
    )

    _plot_frequency_relevance(
        frequency_relevance=(
            mean_incorrect_frequency_relevance
        ),
        frequencies=reference_frequencies,
        subject_name="all_mean",
        selection="incorrect",
        subject=None,
    )


def main() -> None:
    """
    Generate subject-wise and global
    CSP pattern analysis plots.
    """
    subject_correct_channel_relevances = []
    subject_incorrect_channel_relevances = []

    subject_correct_temporal_relevances = []
    subject_incorrect_temporal_relevances = []

    subject_correct_frequency_relevances = []
    subject_incorrect_frequency_relevances = []

    subject_frequencies = []

    for subject in SUBJECTS:
        (
            correct_channel_relevance,
            incorrect_channel_relevance,
            correct_temporal_relevance,
            incorrect_temporal_relevance,
            correct_frequency_relevance,
            incorrect_frequency_relevance,
            frequencies,
        ) = plot_subject_analysis(
            subject
        )

        subject_correct_channel_relevances.append(
            correct_channel_relevance
        )

        subject_incorrect_channel_relevances.append(
            incorrect_channel_relevance
        )

        subject_correct_temporal_relevances.append(
            correct_temporal_relevance
        )

        subject_incorrect_temporal_relevances.append(
            incorrect_temporal_relevance
        )

        subject_correct_frequency_relevances.append(
            correct_frequency_relevance
        )

        subject_incorrect_frequency_relevances.append(
            incorrect_frequency_relevance
        )

        subject_frequencies.append(
            frequencies
        )

    print(
        "Computing global mean CSP pattern analysis..."
    )

    plot_global_analysis(
        subject_correct_channel_relevances=(
            subject_correct_channel_relevances
        ),
        subject_incorrect_channel_relevances=(
            subject_incorrect_channel_relevances
        ),
        subject_correct_temporal_relevances=(
            subject_correct_temporal_relevances
        ),
        subject_incorrect_temporal_relevances=(
            subject_incorrect_temporal_relevances
        ),
        subject_correct_frequency_relevances=(
            subject_correct_frequency_relevances
        ),
        subject_incorrect_frequency_relevances=(
            subject_incorrect_frequency_relevances
        ),
        subject_frequencies=(
            subject_frequencies
        ),
    )

    print(
        "CSP pattern analysis completed."
    )


if __name__ == "__main__":
    main()