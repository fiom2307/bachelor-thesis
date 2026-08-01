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
from src.data.preprocessing import normalize_epochs, prepare_eegnet_input
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
from src.visualization.shap_plots import (
    plot_shap_channel_rankings,
    plot_shap_channel_relevance,
    plot_shap_channel_time,
    plot_shap_temporal_relevance,
    plot_shap_topographies,
)


N_BACKGROUND_SAMPLES = 40
SHAP_NSAMPLES = 20

EPOCH_TMIN = 0.5
IMAGERY_WINDOW = (0.5, 4)


def plot_shap_for_subject(
    subject: int,
) -> None:
    """Compute and save EEGNet SHAP plots for one subject."""
    data = get_data_for_subject(subject)

    if data is None:
        print(f"Skipping A{subject:02d}: data unavailable.")
        return

    X_train, y_train, X_eval, y_eval = data

    X_train, X_eval = normalize_epochs(
        X_train,
        X_eval,
    )

    models = train_or_load_eegnet(
        subject,
        prepare_eegnet_input(X_train),
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

    shap_file = get_shap_values_path(subject)

    if shap_file.exists():
        print(f"Loading saved SHAP values: {shap_file}")

        shap_result = load_shap_result(
            shap_file
        )
    else:
        print(
            f"Computing SHAP values for A{subject:02d}..."
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

    epochs, _ = get_evaluation_erd_epochs_for_subject(
        subject
    )

    class_labels = list(CLASS_LABELS)

    class_relevance = compute_class_shap_relevance(
        result=shap_result,
        class_labels=class_labels,
        correct_only=True,
    )

    channel_relevance = compute_channel_shap_relevance(
        class_relevance
    )

    channel_rankings = rank_shap_channels(
        channel_relevance=channel_relevance,
        channel_names=epochs.ch_names,
    )

    temporal_relevance = compute_temporal_shap_relevance(
        class_relevance,
    )

    times = (
        np.arange(X_eval.shape[-1])
        / epochs.info["sfreq"]
        + EPOCH_TMIN
    )

    topographic_relevance = compute_topographic_shap_relevance(
        class_relevance=class_relevance,
        times=times,
        imagery_window=IMAGERY_WINDOW,
    )

    trial_counts = count_shap_trials_by_class(
        result=shap_result,
        class_labels=class_labels,
        correct_only=True,
    )

    channel_time_figure = plot_shap_channel_time(
        class_relevance=class_relevance,
        times=times,
        channel_names=epochs.ch_names,
        imagery_window=IMAGERY_WINDOW,
        trial_counts=trial_counts,
    )

    channel_relevance_figure = plot_shap_channel_relevance(
        channel_relevance=channel_relevance,
        channel_names=epochs.ch_names,
        trial_counts=trial_counts,
    )

    channel_rankings_figure = plot_shap_channel_rankings(
        channel_rankings=channel_rankings,
        top_n=10,
        trial_counts=trial_counts,
    )

    temporal_figure = plot_shap_temporal_relevance(
        temporal_relevance=temporal_relevance,
        times=times,
        imagery_window=IMAGERY_WINDOW,
        trial_counts=trial_counts,
    )

    topographies_figure = plot_shap_topographies(
        topographic_relevance=topographic_relevance,
        info=epochs.info,
        imagery_window=IMAGERY_WINDOW,
        trial_counts=trial_counts,
    )

    save_figure(
        channel_time_figure,
        get_shap_channel_time_path(subject),
    )

    save_figure(
        channel_relevance_figure,
        get_shap_channel_relevance_path(subject),
    )

    save_figure(
        channel_rankings_figure,
        get_shap_channel_rankings_path(subject),
    )

    save_figure(
        temporal_figure,
        get_shap_temporal_relevance_path(subject),
    )

    save_figure(
        topographies_figure,
        get_shap_topographies_path(subject),
    )

    plt.close(channel_time_figure)
    plt.close(channel_relevance_figure)
    plt.close(channel_rankings_figure)
    plt.close(temporal_figure)
    plt.close(topographies_figure)

    accuracy = np.mean(
        shap_result.predictions
        == shap_result.labels
    )

    print(
        f"A{subject:02d}: "
        f"accuracy={accuracy:.4f}, "
        f"SHAP trials={sum(trial_counts.values())}"
    )


def main() -> None:
    """Generate SHAP plots for all subjects."""
    for subject in range(1, 2):
        plot_shap_for_subject(subject)


if __name__ == "__main__":
    main()