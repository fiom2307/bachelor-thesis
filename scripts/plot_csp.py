from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import mne
import numpy as np

from src.analysis.csp_analysis import (
    compute_channel_csp_relevance,
    compute_class_csp_relevance,
    compute_csp_lda_ensemble_occlusion,
    compute_occlusion_reference,
    compute_temporal_csp_relevance,
    compute_topographic_csp_relevance,
    count_csp_trials_by_class,
    load_csp_analysis_result,
    rank_csp_channels,
    save_csp_analysis_result,
)
from src.data.dataset import (
    get_data_for_subject,
    get_evaluation_erd_epochs_for_subject,
)
from src.data.labels import CLASS_LABELS
from src.models.csp_lda import (
    train_or_load_csp_lda,
)
from src.utils.paths import (
    get_csp_channel_rankings_path,
    get_csp_channel_relevance_path,
    get_csp_channel_time_path,
    get_csp_temporal_relevance_path,
    get_csp_topographies_path,
    get_csp_values_path,
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

ClassRelevance = dict[
    int,
    np.ndarray,
]

TrialCounts = dict[
    int,
    int,
]


EPOCH_TMIN = 0.5
IMAGERY_WINDOW = (0.5, 4.0)

OCCLUSION_WINDOW_DURATION = 0.5
OCCLUSION_WINDOW_STEP = 0.1

TRIAL_SELECTIONS: tuple[
    TrialSelection,
    ...,
] = (
    "correct",
    "incorrect",
)


@dataclass(frozen=True)
class CSPSubjectPlotResult:
    """
    CSP relevance data needed for global aggregation.
    """

    class_relevance: dict[
        TrialSelection,
        ClassRelevance,
    ]

    trial_counts: dict[
        TrialSelection,
        TrialCounts,
    ]

    times: np.ndarray
    info: mne.Info
    channel_names: list[str]


def plot_csp_for_subject(
    subject: int,
) -> CSPSubjectPlotResult | None:
    """
    Compute and save CSP+LDA occlusion plots for one subject.
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

    epochs, _ = (
        get_evaluation_erd_epochs_for_subject(
            subject
        )
    )

    csp_file = get_csp_values_path(
        subject
    )

    if csp_file.exists():
        print(
            "Loading saved CSP+LDA "
            f"occlusion values: {csp_file}"
        )

        csp_result = load_csp_analysis_result(
            csp_file
        )
    else:
        print(
            "Computing CSP+LDA occlusion "
            f"values for A{subject:02d}..."
        )

        models = train_or_load_csp_lda(
            subject,
            X_train,
            y_train,
        )

        reference = compute_occlusion_reference(
            X_train
        )

        csp_result = (
            compute_csp_lda_ensemble_occlusion(
                models=models,
                reference=reference,
                data=X_eval,
                labels=y_eval,
                sfreq=float(
                    epochs.info["sfreq"]
                ),
                tmin=EPOCH_TMIN,
                window_duration=(
                    OCCLUSION_WINDOW_DURATION
                ),
                window_step=(
                    OCCLUSION_WINDOW_STEP
                ),
            )
        )

        save_csp_analysis_result(
            result=csp_result,
            output_file=csp_file,
        )

    class_labels = list(
        CLASS_LABELS
    )

    subject_class_relevance: dict[
        TrialSelection,
        ClassRelevance,
    ] = {}

    subject_trial_counts: dict[
        TrialSelection,
        TrialCounts,
    ] = {}

    for trial_selection in TRIAL_SELECTIONS:
        class_relevance = (
            compute_class_csp_relevance(
                result=csp_result,
                class_labels=class_labels,
                trial_selection=trial_selection,
            )
        )

        trial_counts = count_csp_trials_by_class(
            result=csp_result,
            class_labels=class_labels,
            trial_selection=trial_selection,
        )

        if not class_relevance:
            print(
                f"A{subject:02d}: no "
                f"{trial_selection} trials available."
            )
            continue

        channel_relevance = (
            compute_channel_csp_relevance(
                class_relevance
            )
        )

        channel_rankings = rank_csp_channels(
            channel_relevance=channel_relevance,
            channel_names=epochs.ch_names,
        )

        temporal_relevance = (
            compute_temporal_csp_relevance(
                class_relevance
            )
        )

        topographic_relevance = (
            compute_topographic_csp_relevance(
                class_relevance=class_relevance,
                window_times=csp_result.times,
                imagery_window=IMAGERY_WINDOW,
            )
        )

        figures = _create_csp_figures(
            class_relevance=class_relevance,
            channel_relevance=channel_relevance,
            channel_rankings=channel_rankings,
            temporal_relevance=temporal_relevance,
            topographic_relevance=topographic_relevance,
            times=csp_result.times,
            info=epochs.info,
            channel_names=epochs.ch_names,
            trial_selection=trial_selection,
            subject=subject,
            trial_counts=trial_counts,
        )

        _save_csp_figures(
            figures=figures,
            subject=subject,
            trial_selection=trial_selection,
        )

        _close_figures(
            figures
        )

        subject_class_relevance[
            trial_selection
        ] = class_relevance

        subject_trial_counts[
            trial_selection
        ] = trial_counts

        print(
            f"A{subject:02d} "
            f"({trial_selection}): "
            "CSP occlusion trials="
            f"{sum(trial_counts.values())}"
        )

    accuracy = np.mean(
        csp_result.predictions
        == csp_result.labels
    )

    print(
        f"A{subject:02d}: "
        f"accuracy={accuracy:.4f}"
    )

    return CSPSubjectPlotResult(
        class_relevance=subject_class_relevance,
        trial_counts=subject_trial_counts,
        times=csp_result.times,
        info=epochs.info,
        channel_names=list(
            epochs.ch_names
        ),
    )


def plot_mean_csp(
    class_relevance_by_selection: dict[
        TrialSelection,
        list[ClassRelevance],
    ],
    trial_counts_by_selection: dict[
        TrialSelection,
        list[TrialCounts],
    ],
    times: np.ndarray,
    info: mne.Info,
    channel_names: list[str],
) -> None:
    """
    Compute and save mean CSP relevance plots across subjects.
    """
    for trial_selection in TRIAL_SELECTIONS:
        subject_relevances = (
            class_relevance_by_selection[
                trial_selection
            ]
        )

        if not subject_relevances:
            print(
                "No CSP results available for "
                f"{trial_selection} mean plots."
            )
            continue

        mean_class_relevance = (
            _compute_mean_class_relevance(
                subject_relevances
            )
        )

        total_trial_counts = (
            _sum_trial_counts(
                trial_counts_by_selection[
                    trial_selection
                ]
            )
        )

        channel_relevance = (
            compute_channel_csp_relevance(
                mean_class_relevance
            )
        )

        channel_rankings = rank_csp_channels(
            channel_relevance=channel_relevance,
            channel_names=channel_names,
        )

        temporal_relevance = (
            compute_temporal_csp_relevance(
                mean_class_relevance
            )
        )

        topographic_relevance = (
            compute_topographic_csp_relevance(
                class_relevance=mean_class_relevance,
                window_times=times,
                imagery_window=IMAGERY_WINDOW,
            )
        )

        figures = _create_csp_figures(
            class_relevance=mean_class_relevance,
            channel_relevance=channel_relevance,
            channel_rankings=channel_rankings,
            temporal_relevance=temporal_relevance,
            topographic_relevance=topographic_relevance,
            times=times,
            info=info,
            channel_names=channel_names,
            trial_selection=trial_selection,
            subject=None,
            trial_counts=total_trial_counts,
        )

        _save_csp_figures(
            figures=figures,
            subject=None,
            trial_selection=trial_selection,
        )

        _close_figures(
            figures
        )

        print(
            "Saved mean CSP occlusion plots "
            f"({trial_selection})."
        )


def _create_csp_figures(
    class_relevance: ClassRelevance,
    channel_relevance: dict[int, np.ndarray],
    channel_rankings: dict[
        int,
        list[tuple[str, float]],
    ],
    temporal_relevance: dict[int, np.ndarray],
    topographic_relevance: dict[int, np.ndarray],
    times: np.ndarray,
    info: mne.Info,
    channel_names: list[str],
    trial_selection: TrialSelection,
    subject: int | None,
    trial_counts: TrialCounts,
) -> dict[str, plt.Figure]:
    """
    Create all CSP relevance figures.
    """
    return {
        "channel_time": (
            plot_channel_time_relevance(
                class_relevance=class_relevance,
                times=times,
                channel_names=channel_names,
                method="csp",
                trial_selection=trial_selection,
                subject=subject,
                imagery_window=IMAGERY_WINDOW,
                trial_counts=trial_counts,
            )
        ),
        "channel_relevance": (
            plot_channel_relevance(
                channel_relevance=channel_relevance,
                channel_names=channel_names,
                method="csp",
                trial_selection=trial_selection,
                subject=subject,
                trial_counts=trial_counts,
            )
        ),
        "channel_rankings": (
            plot_channel_rankings(
                channel_rankings=channel_rankings,
                method="csp",
                trial_selection=trial_selection,
                subject=subject,
                top_n=10,
                trial_counts=trial_counts,
            )
        ),
        "temporal": (
            plot_temporal_relevance(
                temporal_relevance=temporal_relevance,
                times=times,
                method="csp",
                trial_selection=trial_selection,
                subject=subject,
                imagery_window=IMAGERY_WINDOW,
                trial_counts=trial_counts,
            )
        ),
        "topographies": (
            plot_topographies(
                topographic_relevance=(
                    topographic_relevance
                ),
                info=info,
                method="csp",
                trial_selection=trial_selection,
                subject=subject,
                imagery_window=IMAGERY_WINDOW,
                trial_counts=trial_counts,
            )
        ),
    }


def _save_csp_figures(
    figures: dict[str, plt.Figure],
    subject: int | None,
    trial_selection: TrialSelection,
) -> None:
    """
    Save all CSP relevance figures.
    """
    save_figure(
        figures["channel_time"],
        get_csp_channel_time_path(
            subject=subject,
            trial_selection=trial_selection,
        ),
    )

    save_figure(
        figures["channel_relevance"],
        get_csp_channel_relevance_path(
            subject=subject,
            trial_selection=trial_selection,
        ),
    )

    save_figure(
        figures["channel_rankings"],
        get_csp_channel_rankings_path(
            subject=subject,
            trial_selection=trial_selection,
        ),
    )

    save_figure(
        figures["temporal"],
        get_csp_temporal_relevance_path(
            subject=subject,
            trial_selection=trial_selection,
        ),
    )

    save_figure(
        figures["topographies"],
        get_csp_topographies_path(
            subject=subject,
            trial_selection=trial_selection,
        ),
    )


def _close_figures(
    figures: dict[str, plt.Figure],
) -> None:
    """
    Close all relevance figures.
    """
    for figure in figures.values():
        plt.close(
            figure
        )


def _compute_mean_class_relevance(
    relevance_results: list[ClassRelevance],
) -> ClassRelevance:
    """
    Compute subject-balanced mean relevance for each class.

    A class is averaged only across subjects that contain at least one
    selected trial for that class.
    """
    mean_relevance = {}

    for class_id in CLASS_LABELS:
        class_arrays = [
            relevance[class_id]
            for relevance in relevance_results
            if class_id in relevance
        ]

        if not class_arrays:
            continue

        reference_shape = (
            class_arrays[0].shape
        )

        if any(
            array.shape != reference_shape
            for array in class_arrays
        ):
            raise ValueError(
                "CSP relevance shapes differ "
                f"between subjects for class {class_id}."
            )

        mean_relevance[class_id] = np.mean(
            class_arrays,
            axis=0,
        )

    return mean_relevance


def _sum_trial_counts(
    trial_count_results: list[TrialCounts],
) -> TrialCounts:
    """
    Sum selected trial counts across subjects.
    """
    return {
        class_id: int(
            sum(
                counts.get(
                    class_id,
                    0,
                )
                for counts
                in trial_count_results
            )
        )
        for class_id in CLASS_LABELS
    }


def main() -> None:
    """
    Generate subject-wise and mean CSP+LDA occlusion plots.
    """
    class_relevance_by_selection: dict[
        TrialSelection,
        list[ClassRelevance],
    ] = {
        "correct": [],
        "incorrect": [],
    }

    trial_counts_by_selection: dict[
        TrialSelection,
        list[TrialCounts],
    ] = {
        "correct": [],
        "incorrect": [],
    }

    reference_times = None
    reference_info = None
    reference_channel_names = None

    for subject in range(
        1,
        10,
    ):
        subject_result = plot_csp_for_subject(
            subject
        )

        if subject_result is None:
            continue

        if reference_times is None:
            reference_times = (
                subject_result.times
            )

            reference_info = (
                subject_result.info
            )

            reference_channel_names = (
                subject_result.channel_names
            )
        else:
            if not np.array_equal(
                reference_times,
                subject_result.times,
            ):
                raise ValueError(
                    "CSP occlusion times differ "
                    "between subjects."
                )

            if (
                reference_channel_names
                != subject_result.channel_names
            ):
                raise ValueError(
                    "EEG channel order differs "
                    "between subjects."
                )

        for trial_selection in TRIAL_SELECTIONS:
            class_relevance = (
                subject_result
                .class_relevance
                .get(trial_selection)
            )

            trial_counts = (
                subject_result
                .trial_counts
                .get(trial_selection)
            )

            if (
                class_relevance is None
                or trial_counts is None
            ):
                continue

            class_relevance_by_selection[
                trial_selection
            ].append(
                class_relevance
            )

            trial_counts_by_selection[
                trial_selection
            ].append(
                trial_counts
            )

    if (
        reference_times is None
        or reference_info is None
        or reference_channel_names is None
    ):
        print(
            "No CSP subject data available "
            "for mean plots."
        )
        return

    plot_mean_csp(
        class_relevance_by_selection=(
            class_relevance_by_selection
        ),
        trial_counts_by_selection=(
            trial_counts_by_selection
        ),
        times=reference_times,
        info=reference_info,
        channel_names=reference_channel_names,
    )


if __name__ == "__main__":
    main()