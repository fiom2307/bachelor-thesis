from collections.abc import Mapping
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.shap_analysis import (
    compute_channel_shap_relevance,
    compute_class_shap_relevance,
    compute_eegnet_ensemble_shap,
    compute_temporal_shap_relevance,
    compute_topographic_shap_relevance,
    count_shap_trials_by_class,
    load_shap_result,
    rank_shap_channels,
    save_shap_result,
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
    get_shap_channel_rankings_path,
    get_shap_channel_relevance_path,
    get_shap_channel_time_path,
    get_shap_temporal_relevance_path,
    get_shap_topographies_path,
    get_shap_values_path,
)
from src.visualization.common import save_figure
from src.visualization.relevance_plots import (
    plot_channel_rankings,
    plot_channel_relevance,
    plot_channel_time_relevance,
    plot_temporal_relevance,
    plot_topographies,
)


TrialSelection = Literal[
    "correct",
    "incorrect",
]

ChannelTimeRelevance = Mapping[
    int,
    np.ndarray,
]

TrialCounts = Mapping[
    int,
    int,
]


N_BACKGROUND_SAMPLES = 40
SHAP_NSAMPLES = 20

EPOCH_TMIN = 0.5
IMAGERY_WINDOW = (0.5, 4.0)

TRIAL_SELECTIONS: tuple[TrialSelection, ...] = (
    "correct",
    "incorrect",
)


def plot_shap_for_subject(
    subject: int,
) -> dict[TrialSelection, tuple[ChannelTimeRelevance, TrialCounts]] | None:
    """
    Compute and save EEGNet SHAP plots for one subject.

    Returns class-wise relevance and trial counts for each trial selection
    so they can later be aggregated into global mean plots.
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

    background = select_shap_background(
        data=X_train,
        labels=y_train,
        n_samples=N_BACKGROUND_SAMPLES,
    )

    background_sets = [
        background
        for _ in models
    ]

    shap_file = get_shap_values_path(
        subject
    )

    if shap_file.exists():
        print(
            f"Loading saved SHAP values: "
            f"{shap_file}"
        )

        shap_result = load_shap_result(
            shap_file
        )
    else:
        print(
            f"Computing SHAP values for "
            f"A{subject:02d}..."
        )

        shap_result = compute_eegnet_ensemble_shap(
            models=models,
            background_sets=background_sets,
            data=X_eval,
            labels=y_eval,
            nsamples=SHAP_NSAMPLES,
        )

        save_shap_result(
            result=shap_result,
            output_file=shap_file,
        )

    epochs, _ = (
        get_evaluation_erd_epochs_for_subject(
            subject
        )
    )

    class_labels = list(
        CLASS_LABELS
    )

    times = (
        np.arange(
            X_eval.shape[-1]
        )
        / epochs.info["sfreq"]
        + EPOCH_TMIN
    )

    subject_results: dict[
        TrialSelection,
        tuple[ChannelTimeRelevance, TrialCounts],
    ] = {}

    for trial_selection in TRIAL_SELECTIONS:
        class_relevance = (
            compute_class_shap_relevance(
                result=shap_result,
                class_labels=class_labels,
                trial_selection=trial_selection,
            )
        )

        trial_counts = count_shap_trials_by_class(
            result=shap_result,
            class_labels=class_labels,
            trial_selection=trial_selection,
        )

        if not class_relevance:
            print(
                f"A{subject:02d}: "
                f"no {trial_selection} trials available."
            )
            continue

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

        channel_time_figure = (
            plot_channel_time_relevance(
                class_relevance=class_relevance,
                times=times,
                channel_names=epochs.ch_names,
                method="shap",
                trial_selection=trial_selection,
                subject=subject,
                imagery_window=IMAGERY_WINDOW,
                trial_counts=trial_counts,
            )
        )

        channel_relevance_figure = (
            plot_channel_relevance(
                channel_relevance=channel_relevance,
                channel_names=epochs.ch_names,
                method="shap",
                trial_selection=trial_selection,
                subject=subject,
                trial_counts=trial_counts,
            )
        )

        channel_rankings_figure = (
            plot_channel_rankings(
                channel_rankings=channel_rankings,
                method="shap",
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
                method="shap",
                trial_selection=trial_selection,
                subject=subject,
                imagery_window=IMAGERY_WINDOW,
                trial_counts=trial_counts,
            )
        )

        topographies_figure = plot_topographies(
            topographic_relevance=topographic_relevance,
            info=epochs.info,
            method="shap",
            trial_selection=trial_selection,
            subject=subject,
            imagery_window=IMAGERY_WINDOW,
            trial_counts=trial_counts,
        )

        save_figure(
            channel_time_figure,
            get_shap_channel_time_path(
                subject=subject,
                trial_selection=trial_selection,
            ),
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
            topographies_figure,
            get_shap_topographies_path(
                subject=subject,
                trial_selection=trial_selection,
            ),
        )

        plt.close(
            channel_time_figure
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
            topographies_figure
        )

        subject_results[
            trial_selection
        ] = (
            class_relevance,
            trial_counts,
        )

        print(
            f"A{subject:02d} "
            f"({trial_selection}): "
            f"SHAP trials={sum(trial_counts.values())}"
        )

    accuracy = np.mean(
        shap_result.predictions
        == shap_result.labels
    )

    print(
        f"A{subject:02d}: "
        f"accuracy={accuracy:.4f}"
    )

    return subject_results


def plot_mean_shap(
    aggregated_class_relevance: dict[
        TrialSelection,
        list[ChannelTimeRelevance],
    ],
    aggregated_trial_counts: dict[
        TrialSelection,
        list[TrialCounts],
    ],
    reference_info,
    reference_times: np.ndarray,
    channel_names: list[str],
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

        mean_trial_counts = _sum_trial_counts(
            aggregated_trial_counts[
                trial_selection
            ]
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

        channel_time_figure = (
            plot_channel_time_relevance(
                class_relevance=mean_class_relevance,
                times=reference_times,
                channel_names=channel_names,
                method="shap",
                trial_selection=trial_selection,
                subject=None,
                imagery_window=IMAGERY_WINDOW,
                trial_counts=mean_trial_counts,
            )
        )

        channel_relevance_figure = (
            plot_channel_relevance(
                channel_relevance=channel_relevance,
                channel_names=channel_names,
                method="shap",
                trial_selection=trial_selection,
                subject=None,
                trial_counts=mean_trial_counts,
            )
        )

        channel_rankings_figure = (
            plot_channel_rankings(
                channel_rankings=channel_rankings,
                method="shap",
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
                method="shap",
                trial_selection=trial_selection,
                subject=None,
                imagery_window=IMAGERY_WINDOW,
                trial_counts=mean_trial_counts,
            )
        )

        topographies_figure = plot_topographies(
            topographic_relevance=topographic_relevance,
            info=reference_info,
            method="shap",
            trial_selection=trial_selection,
            subject=None,
            imagery_window=IMAGERY_WINDOW,
            trial_counts=mean_trial_counts,
        )

        save_figure(
            channel_time_figure,
            get_shap_channel_time_path(
                subject=None,
                trial_selection=trial_selection,
            ),
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
            topographies_figure,
            get_shap_topographies_path(
                subject=None,
                trial_selection=trial_selection,
            ),
        )

        plt.close(
            channel_time_figure
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
            topographies_figure
        )

        print(
            f"Saved mean SHAP plots "
            f"({trial_selection})."
        )


def _mean_class_relevance(
    relevance_list: list[ChannelTimeRelevance],
) -> dict[int, np.ndarray]:
    """
    Compute the mean class-wise channel-time relevance across subjects.

    If a class is missing for some subjects (for example in incorrect
    trials), average only across subjects where that class is available.
    """
    mean_relevance: dict[int, np.ndarray] = {}

    for class_id in CLASS_LABELS:
        class_arrays = [
            relevance[class_id]
            for relevance in relevance_list
            if class_id in relevance
        ]

        if not class_arrays:
            continue

        mean_relevance[class_id] = np.mean(
            class_arrays,
            axis=0,
        )

    return mean_relevance


def _sum_trial_counts(
    trial_counts_list: list[TrialCounts],
) -> dict[int, int]:
    """
    Sum class-wise trial counts across subjects.
    """
    summed_counts: dict[int, int] = {}

    for class_id in CLASS_LABELS:
        summed_counts[class_id] = int(
            sum(
                counts.get(class_id, 0)
                for counts in trial_counts_list
            )
        )

    return summed_counts


def main() -> None:
    """Generate SHAP plots for all subjects and global means."""
    aggregated_class_relevance: dict[
        TrialSelection,
        list[ChannelTimeRelevance],
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

    for subject in range(
        1,
        10,
    ):
        subject_results = plot_shap_for_subject(
            subject
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

        for trial_selection, (
            class_relevance,
            trial_counts,
        ) in subject_results.items():
            aggregated_class_relevance[
                trial_selection
            ].append(class_relevance)

            aggregated_trial_counts[
                trial_selection
            ].append(trial_counts)

    if (
        reference_info is None
        or reference_times is None
        or reference_channel_names is None
    ):
        print(
            "No subject data available for "
            "mean SHAP plots."
        )
        return

    plot_mean_shap(
        aggregated_class_relevance=aggregated_class_relevance,
        aggregated_trial_counts=aggregated_trial_counts,
        reference_info=reference_info,
        reference_times=reference_times,
        channel_names=reference_channel_names,
    )


if __name__ == "__main__":
    main()