import matplotlib.pyplot as plt
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


EPOCH_TMIN = 0.5
IMAGERY_WINDOW = (0.5, 4.0)

OCCLUSION_WINDOW_DURATION = 0.5
OCCLUSION_WINDOW_STEP = 0.1


def plot_csp_for_subject(
    subject: int,
) -> None:
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
        return

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
            f"Loading saved CSP+LDA "
            f"occlusion values: {csp_file}"
        )

        csp_result = load_csp_analysis_result(
            csp_file
        )
    else:
        print(
            f"Computing CSP+LDA occlusion "
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

    class_relevance = (
        compute_class_csp_relevance(
            result=csp_result,
            class_labels=class_labels,
            correct_only=True,
        )
    )

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

    trial_counts = count_csp_trials_by_class(
        result=csp_result,
        class_labels=class_labels,
        correct_only=True,
    )

    channel_time_figure = (
        plot_channel_time_relevance(
            class_relevance=class_relevance,
            times=csp_result.times,
            channel_names=epochs.ch_names,
            method="csp",
            imagery_window=IMAGERY_WINDOW,
            trial_counts=trial_counts,
        )
    )

    channel_relevance_figure = (
        plot_channel_relevance(
            channel_relevance=channel_relevance,
            channel_names=epochs.ch_names,
            method="csp",
            trial_counts=trial_counts,
        )
    )

    channel_rankings_figure = (
        plot_channel_rankings(
            channel_rankings=channel_rankings,
            method="csp",
            top_n=10,
            trial_counts=trial_counts,
        )
    )

    temporal_figure = (
        plot_temporal_relevance(
            temporal_relevance=temporal_relevance,
            times=csp_result.times,
            method="csp",
            imagery_window=IMAGERY_WINDOW,
            trial_counts=trial_counts,
        )
    )

    topographies_figure = plot_topographies(
        topographic_relevance=(
            topographic_relevance
        ),
        info=epochs.info,
        method="csp",
        imagery_window=IMAGERY_WINDOW,
        trial_counts=trial_counts,
    )

    save_figure(
        channel_time_figure,
        get_csp_channel_time_path(
            subject
        ),
    )

    save_figure(
        channel_relevance_figure,
        get_csp_channel_relevance_path(
            subject
        ),
    )

    save_figure(
        channel_rankings_figure,
        get_csp_channel_rankings_path(
            subject
        ),
    )

    save_figure(
        temporal_figure,
        get_csp_temporal_relevance_path(
            subject
        ),
    )

    save_figure(
        topographies_figure,
        get_csp_topographies_path(
            subject
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

    accuracy = np.mean(
        csp_result.predictions
        == csp_result.labels
    )

    print(
        f"A{subject:02d}: "
        f"accuracy={accuracy:.4f}, "
        f"CSP occlusion trials="
        f"{sum(trial_counts.values())}"
    )


def main() -> None:
    """
    Generate CSP+LDA occlusion plots for all subjects.
    """
    for subject in range(
        1,
        10,
    ):
        plot_csp_for_subject(
            subject
        )


if __name__ == "__main__":
    main()