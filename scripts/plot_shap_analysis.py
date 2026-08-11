from collections.abc import Mapping
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.shap_analysis import (
    compute_channel_shap_relevance,
    compute_eegnet_ensemble_shap,
    compute_eegnet_frequency_shap,
    compute_frequency_shap_relevance,
    compute_temporal_shap_relevance,
    compute_topographic_shap_relevance,
    load_frequency_domain_shap_result,
    load_time_domain_shap_result,
    rank_shap_channels,
    save_frequency_domain_shap_result,
    save_time_domain_shap_result,
    select_shap_background,
)
from src.data.dataset import (
    get_data_for_subject,
    get_evaluation_erd_epochs_for_subject,
)
from src.data.labels import CLASS_LABELS
from src.data.preprocessing import (
    normalize_epochs,
    prepare_eegnet_input,
)
from src.models.eegnet import (
    train_or_load_eegnet,
)
from src.utils.paths import (
    get_frequency_domain_shap_values_path,
    get_shap_channel_rankings_path,
    get_shap_channel_relevance_path,
    get_shap_frequency_relevance_path,
    get_shap_temporal_relevance_path,
    get_shap_topographies_path,
    get_time_domain_shap_values_path,
)
from src.visualization.common import save_figure
from src.visualization.shap_plots import (
    plot_channel_rankings,
    plot_channel_relevance,
    plot_frequency_relevance,
    plot_temporal_relevance,
    plot_topographies,
)


TrialSelection = Literal[
    "correct",
    "incorrect",
]

ClassRelevance = Mapping[
    int,
    np.ndarray,
]

VectorRelevance = Mapping[
    int,
    np.ndarray,
]

TrialCounts = Mapping[
    int,
    int,
]


N_BACKGROUND_SAMPLES = 40

TIME_SHAP_NSAMPLES = 200
FREQUENCY_SHAP_NSAMPLES = 256

EPOCH_TMIN = 0.5
IMAGERY_WINDOW = (0.5, 4.0)

TRIAL_SELECTIONS: tuple[
    TrialSelection,
    ...,
] = (
    "correct",
    "incorrect",
)


def plot_shap_for_subject(
    subject: int,
) -> dict[
    TrialSelection,
    tuple[
        ClassRelevance,
        VectorRelevance,
        TrialCounts,
    ],
] | None:
    """
    Compute and save EEGNet SHAP plots for one subject.

    Returns time-domain class relevance, frequency-domain relevance,
    and trial counts for each trial selection so they can later be
    aggregated into global mean plots.
    """
    data = get_data_for_subject(
        subject
    )

    if data is None:
        print(
            f"Skipping A{subject:02d}: "
            "data unavailable."
        )
        return None

    X_train, y_train, X_eval, y_eval = data

    X_train, X_eval = normalize_epochs(
        X_train,
        X_eval,
    )

    models = train_or_load_eegnet(
        subject,
        prepare_eegnet_input(
            X_train
        ),
        y_train,
    )

    epochs, _ = (
        get_evaluation_erd_epochs_for_subject(
            subject
        )
    )

    sfreq = float(
        epochs.info["sfreq"]
    )

    times = (
        np.arange(
            X_eval.shape[-1]
        )
        / sfreq
        + EPOCH_TMIN
    )

    # ------------------------------------------------------------------
    # Time-domain SHAP
    # ------------------------------------------------------------------

    time_shap_file = (
        get_time_domain_shap_values_path(
            subject
        )
    )

    if time_shap_file.exists():
        print(
            "Loading saved time-domain SHAP values: "
            f"{time_shap_file}"
        )

        time_result = (
            load_time_domain_shap_result(
                time_shap_file
            )
        )
    else:
        background = select_shap_background(
            data=X_train,
            labels=y_train,
            n_samples=N_BACKGROUND_SAMPLES,
        )

        background_sets = [
            background
            for _ in models
        ]

        print(
            "Computing time-domain SHAP values for "
            f"A{subject:02d}..."
        )

        time_result = (
            compute_eegnet_ensemble_shap(
                models=models,
                background_sets=background_sets,
                data=X_eval,
                labels=y_eval,
                nsamples=TIME_SHAP_NSAMPLES,
            )
        )

        save_time_domain_shap_result(
            result=time_result,
            output_file=time_shap_file,
        )

    # ------------------------------------------------------------------
    # Frequency-domain SHAP
    # ------------------------------------------------------------------

    frequency_shap_file = (
        get_frequency_domain_shap_values_path(
            subject
        )
    )

    if frequency_shap_file.exists():
        print(
            "Loading saved frequency-domain SHAP values: "
            f"{frequency_shap_file}"
        )

        frequency_result = (
            load_frequency_domain_shap_result(
                frequency_shap_file
            )
        )
    else:
        print(
            "Computing frequency-domain SHAP values for "
            f"A{subject:02d}..."
        )

        frequency_result = (
            compute_eegnet_frequency_shap(
                models=models,
                data=X_eval,
                labels=y_eval,
                sfreq=sfreq,
                nsamples=FREQUENCY_SHAP_NSAMPLES,
            )
        )

        save_frequency_domain_shap_result(
            result=frequency_result,
            output_file=frequency_shap_file,
        )

    # ------------------------------------------------------------------
    # Relevance analyses
    # ------------------------------------------------------------------

    subject_results: dict[
        TrialSelection,
        tuple[
            ClassRelevance,
            VectorRelevance,
            TrialCounts,
        ],
    ] = {}

    for trial_selection in TRIAL_SELECTIONS:
        time_trial_mask = _get_trial_mask(
            result=time_result,
            trial_selection=trial_selection,
        )

        frequency_trial_mask = _get_trial_mask(
            result=frequency_result,
            trial_selection=trial_selection,
        )

        class_relevance = (
            _compute_class_shap_relevance(
                shap_values=time_result.values,
                labels=time_result.labels,
                trial_mask=time_trial_mask,
            )
        )

        trial_counts = _count_trials_by_class(
            labels=time_result.labels,
            trial_mask=time_trial_mask,
        )

        if not class_relevance:
            print(
                f"A{subject:02d}: "
                f"no {trial_selection} trials available."
            )
            continue

        frequency_relevance = (
            compute_frequency_shap_relevance(
                shap_values=frequency_result.values,
                labels=frequency_result.labels,
                trial_mask=frequency_trial_mask,
            )
        )

        channel_relevance = (
            compute_channel_shap_relevance(
                class_relevance
            )
        )

        channel_rankings = rank_shap_channels(
            channel_relevance=channel_relevance,
            channel_names=epochs.ch_names,
        )

        temporal_relevance = (
            compute_temporal_shap_relevance(
                class_relevance
            )
        )

        topographic_relevance = (
            compute_topographic_shap_relevance(
                class_relevance=class_relevance,
                times=times,
                imagery_window=IMAGERY_WINDOW,
            )
        )

        # --------------------------------------------------------------
        # Subject-wise plots
        # --------------------------------------------------------------

        channel_relevance_figure = (
            plot_channel_relevance(
                channel_relevance=channel_relevance,
                channel_names=epochs.ch_names,
                trial_selection=trial_selection,
                subject=subject,
                trial_counts=trial_counts,
            )
        )

        channel_rankings_figure = (
            plot_channel_rankings(
                channel_rankings=channel_rankings,
                trial_selection=trial_selection,
                subject=subject,
                top_n=10,
                trial_counts=trial_counts,
            )
        )

        temporal_figure = (
            plot_temporal_relevance(
                temporal_relevance=temporal_relevance,
                times=times,
                trial_selection=trial_selection,
                subject=subject,
                imagery_window=IMAGERY_WINDOW,
                trial_counts=trial_counts,
            )
        )

        frequency_figure = (
            plot_frequency_relevance(
                frequency_relevance=frequency_relevance,
                frequency_bands=(
                    frequency_result.frequency_bands
                ),
                trial_selection=trial_selection,
                subject=subject,
                trial_counts=trial_counts,
            )
        )

        topographies_figure = (
            plot_topographies(
                topographic_relevance=(
                    topographic_relevance
                ),
                info=epochs.info,
                trial_selection=trial_selection,
                subject=subject,
                imagery_window=IMAGERY_WINDOW,
                trial_counts=trial_counts,
            )
        )

        save_figure(
            channel_relevance_figure,
            get_shap_channel_relevance_path(
                subject=subject,
                trial_selection=trial_selection,
            ),
        )

        save_figure(
            channel_rankings_figure,
            get_shap_channel_rankings_path(
                subject=subject,
                trial_selection=trial_selection,
            ),
        )

        save_figure(
            temporal_figure,
            get_shap_temporal_relevance_path(
                subject=subject,
                trial_selection=trial_selection,
            ),
        )

        save_figure(
            frequency_figure,
            get_shap_frequency_relevance_path(
                subject=subject,
                trial_selection=trial_selection,
            ),
        )

        save_figure(
            topographies_figure,
            get_shap_topographies_path(
                subject=subject,
                trial_selection=trial_selection,
            ),
        )

        plt.close(
            channel_relevance_figure
        )

        plt.close(
            channel_rankings_figure
        )

        plt.close(
            temporal_figure
        )

        plt.close(
            frequency_figure
        )

        plt.close(
            topographies_figure
        )

        subject_results[
            trial_selection
        ] = (
            class_relevance,
            frequency_relevance,
            trial_counts,
        )

        print(
            f"A{subject:02d} "
            f"({trial_selection}): "
            f"SHAP trials="
            f"{sum(trial_counts.values())}"
        )

    accuracy = np.mean(
        time_result.predictions
        == time_result.labels
    )

    print(
        f"A{subject:02d}: "
        f"accuracy={accuracy:.4f}"
    )

    return subject_results


def plot_mean_shap(
    aggregated_class_relevance: dict[
        TrialSelection,
        list[ClassRelevance],
    ],
    aggregated_frequency_relevance: dict[
        TrialSelection,
        list[VectorRelevance],
    ],
    aggregated_trial_counts: dict[
        TrialSelection,
        list[TrialCounts],
    ],
    reference_info,
    reference_times: np.ndarray,
    channel_names: list[str],
    frequency_bands: tuple[
        tuple[float, float],
        ...,
    ],
) -> None:
    """
    Compute and save global mean SHAP plots across subjects.
    """
    for trial_selection in TRIAL_SELECTIONS:
        class_relevance_list = (
            aggregated_class_relevance[
                trial_selection
            ]
        )

        frequency_relevance_list = (
            aggregated_frequency_relevance[
                trial_selection
            ]
        )

        if not class_relevance_list:
            print(
                f"No subject results for "
                f"{trial_selection} mean plot."
            )
            continue

        mean_class_relevance = (
            _mean_class_relevance(
                class_relevance_list
            )
        )

        mean_frequency_relevance = (
            _mean_class_relevance(
                frequency_relevance_list
            )
        )

        mean_trial_counts = (
            _sum_trial_counts(
                aggregated_trial_counts[
                    trial_selection
                ]
            )
        )

        channel_relevance = (
            compute_channel_shap_relevance(
                mean_class_relevance
            )
        )

        channel_rankings = rank_shap_channels(
            channel_relevance=channel_relevance,
            channel_names=channel_names,
        )

        temporal_relevance = (
            compute_temporal_shap_relevance(
                mean_class_relevance
            )
        )

        topographic_relevance = (
            compute_topographic_shap_relevance(
                class_relevance=mean_class_relevance,
                times=reference_times,
                imagery_window=IMAGERY_WINDOW,
            )
        )

        # --------------------------------------------------------------
        # Global mean plots
        # --------------------------------------------------------------

        channel_relevance_figure = (
            plot_channel_relevance(
                channel_relevance=channel_relevance,
                channel_names=channel_names,
                trial_selection=trial_selection,
                subject=None,
                trial_counts=mean_trial_counts,
            )
        )

        channel_rankings_figure = (
            plot_channel_rankings(
                channel_rankings=channel_rankings,
                trial_selection=trial_selection,
                subject=None,
                top_n=10,
                trial_counts=mean_trial_counts,
            )
        )

        temporal_figure = (
            plot_temporal_relevance(
                temporal_relevance=temporal_relevance,
                times=reference_times,
                trial_selection=trial_selection,
                subject=None,
                imagery_window=IMAGERY_WINDOW,
                trial_counts=mean_trial_counts,
            )
        )

        frequency_figure = (
            plot_frequency_relevance(
                frequency_relevance=(
                    mean_frequency_relevance
                ),
                frequency_bands=frequency_bands,
                trial_selection=trial_selection,
                subject=None,
                trial_counts=mean_trial_counts,
            )
        )

        topographies_figure = (
            plot_topographies(
                topographic_relevance=(
                    topographic_relevance
                ),
                info=reference_info,
                trial_selection=trial_selection,
                subject=None,
                imagery_window=IMAGERY_WINDOW,
                trial_counts=mean_trial_counts,
            )
        )

        save_figure(
            channel_relevance_figure,
            get_shap_channel_relevance_path(
                subject=None,
                trial_selection=trial_selection,
            ),
        )

        save_figure(
            channel_rankings_figure,
            get_shap_channel_rankings_path(
                subject=None,
                trial_selection=trial_selection,
            ),
        )

        save_figure(
            temporal_figure,
            get_shap_temporal_relevance_path(
                subject=None,
                trial_selection=trial_selection,
            ),
        )

        save_figure(
            frequency_figure,
            get_shap_frequency_relevance_path(
                subject=None,
                trial_selection=trial_selection,
            ),
        )

        save_figure(
            topographies_figure,
            get_shap_topographies_path(
                subject=None,
                trial_selection=trial_selection,
            ),
        )

        plt.close(
            channel_relevance_figure
        )

        plt.close(
            channel_rankings_figure
        )

        plt.close(
            temporal_figure
        )

        plt.close(
            frequency_figure
        )

        plt.close(
            topographies_figure
        )

        print(
            f"Saved mean SHAP plots "
            f"({trial_selection})."
        )


def _get_trial_mask(
    result,
    trial_selection: TrialSelection,
) -> np.ndarray:
    """
    Return the selected trial mask.
    """
    if trial_selection == "correct":
        return result.correct_mask

    return result.incorrect_mask


def _compute_class_shap_relevance(
    shap_values: np.ndarray,
    labels: np.ndarray,
    trial_mask: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Compute class-wise mean absolute SHAP relevance
    for selected trials.
    """
    class_relevance = {}

    for class_id in CLASS_LABELS:
        class_mask = (
            (labels == class_id)
            & trial_mask
        )

        if not np.any(
            class_mask
        ):
            continue

        class_relevance[
            class_id
        ] = np.abs(
            shap_values[
                class_mask
            ]
        ).mean(
            axis=0
        )

    return class_relevance


def _count_trials_by_class(
    labels: np.ndarray,
    trial_mask: np.ndarray,
) -> dict[int, int]:
    """
    Count selected trials for each motor-imagery class.
    """
    return {
        class_id: int(
            np.sum(
                (labels == class_id)
                & trial_mask
            )
        )
        for class_id in CLASS_LABELS
    }


def _mean_class_relevance(
    relevance_list: list[
        Mapping[
            int,
            np.ndarray,
        ]
    ],
) -> dict[int, np.ndarray]:
    """
    Compute class-wise mean relevance across subjects.

    If a class is missing for some subjects, average only across
    subjects where that class is available.
    """
    mean_relevance = {}

    for class_id in CLASS_LABELS:
        class_arrays = [
            relevance[
                class_id
            ]
            for relevance
            in relevance_list
            if class_id in relevance
        ]

        if not class_arrays:
            continue

        mean_relevance[
            class_id
        ] = np.mean(
            class_arrays,
            axis=0,
        )

    return mean_relevance


def _sum_trial_counts(
    trial_counts_list: list[
        TrialCounts
    ],
) -> dict[int, int]:
    """
    Sum class-wise trial counts across subjects.
    """
    return {
        class_id: int(
            sum(
                counts.get(
                    class_id,
                    0,
                )
                for counts
                in trial_counts_list
            )
        )
        for class_id in CLASS_LABELS
    }


def main() -> None:
    """
    Generate subject-wise and global mean EEGNet SHAP plots.
    """
    aggregated_class_relevance: dict[
        TrialSelection,
        list[ClassRelevance],
    ] = {
        "correct": [],
        "incorrect": [],
    }

    aggregated_frequency_relevance: dict[
        TrialSelection,
        list[VectorRelevance],
    ] = {
        "correct": [],
        "incorrect": [],
    }

    aggregated_trial_counts: dict[
        TrialSelection,
        list[TrialCounts],
    ] = {
        "correct": [],
        "incorrect": [],
    }

    reference_info = None
    reference_times = None
    reference_channel_names = None
    reference_frequency_bands = None

    for subject in range(
        1,
        10,
    ):
        subject_results = (
            plot_shap_for_subject(
                subject
            )
        )

        if subject_results is None:
            continue

        epochs, _ = (
            get_evaluation_erd_epochs_for_subject(
                subject
            )
        )

        data = get_data_for_subject(
            subject
        )

        if data is None:
            continue

        _, _, X_eval, _ = data

        if reference_info is None:
            reference_info = epochs.info

            reference_channel_names = (
                epochs.ch_names
            )

            reference_times = (
                np.arange(
                    X_eval.shape[-1]
                )
                / epochs.info["sfreq"]
                + EPOCH_TMIN
            )

            frequency_file = (
                get_frequency_domain_shap_values_path(
                    subject
                )
            )

            frequency_result = (
                load_frequency_domain_shap_result(
                    frequency_file
                )
            )

            reference_frequency_bands = (
                frequency_result.frequency_bands
            )

        for trial_selection, (
            class_relevance,
            frequency_relevance,
            trial_counts,
        ) in subject_results.items():
            aggregated_class_relevance[
                trial_selection
            ].append(
                class_relevance
            )

            aggregated_frequency_relevance[
                trial_selection
            ].append(
                frequency_relevance
            )

            aggregated_trial_counts[
                trial_selection
            ].append(
                trial_counts
            )

    if (
        reference_info is None
        or reference_times is None
        or reference_channel_names is None
        or reference_frequency_bands is None
    ):
        print(
            "No subject data available for "
            "mean SHAP plots."
        )

        return

    plot_mean_shap(
        aggregated_class_relevance=(
            aggregated_class_relevance
        ),
        aggregated_frequency_relevance=(
            aggregated_frequency_relevance
        ),
        aggregated_trial_counts=(
            aggregated_trial_counts
        ),
        reference_info=reference_info,
        reference_times=reference_times,
        channel_names=reference_channel_names,
        frequency_bands=reference_frequency_bands,
    )


if __name__ == "__main__":
    main()