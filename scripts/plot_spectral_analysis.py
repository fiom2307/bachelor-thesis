import matplotlib.pyplot as plt
import mne
import numpy as np

from src.analysis.spectral import (
    compute_all_erd_results,
    compute_channel_psd,
    compute_channel_tfr,
)
from src.data.dataset import (
    get_evaluation_erd_epochs_for_subject,
)
from src.utils.paths import (
    get_erd_topographies_path,
    get_psd_path,
    get_tfr_path,
)
from src.visualization.common import (
    save_figure,
)
from src.visualization.spectral import (
    create_channel_psd_figure,
    create_channel_tfr_figure,
    create_erd_topomap_figure,
)


# Subjects A01–A09.
SUBJECTS = range(1, 2)

# C3: right-hand imagery.
# Cz: feet imagery.
# C4: left-hand imagery.
CHANNELS = (
    "C3",
    "Cz",
    "C4",
    "CP4",
    "CP3",
)

FREQUENCY_BANDS = {
    "Mu (8–13 Hz)": (8.0, 13.0),
    "Beta (13–30 Hz)": (13.0, 30.0),
}

TFR_FREQUENCIES = np.arange(
    8.0,
    31.0,
    1.0,
)

BASELINE = (-1.5, -0.5)
IMAGERY_WINDOW = (0.5, 4.0)
IMAGERY_WINDOW_2 = (0.5, 3.5)

# Internal class labels.
CLASS_IDS = {
    "Left hand": 0,
    "Right hand": 1,
    "Feet": 2,
    "Tongue": 3,
}

TFR_DECIM = 2

# False: skip figures that already exist.
# True: regenerate and overwrite every figure.
OVERWRITE = False


def generate_topographies(
    subject: int,
    epochs: mne.Epochs,
    labels: np.ndarray,
) -> None:
    """
    Generate the class-wise mu and beta ERD/ERS topographies
    for one subject.
    """
    output_path = get_erd_topographies_path(
        subject
    )

    if output_path.exists() and not OVERWRITE:
        print(
            "Topographies already exist:"
            f"\n  {output_path}"
        )
        return

    results = compute_all_erd_results(
        epochs=epochs,
        labels=labels,
        frequency_bands=FREQUENCY_BANDS,
        baseline=BASELINE,
        imagery_window=IMAGERY_WINDOW_2,
    )

    figure = create_erd_topomap_figure(
        epochs=epochs,
        results=results,
        band_names=list(FREQUENCY_BANDS),
        subject=subject,
        baseline=BASELINE,
        imagery_window=IMAGERY_WINDOW_2,
    )

    try:
        output_file = save_figure(
            figure=figure,
            output_file=output_path,
        )
    finally:
        plt.close(figure)

    print(
        "Saved ERD/ERS topographies:"
        f"\n  {output_file}"
    )


def generate_tfr(
    subject: int,
    epochs: mne.Epochs,
    labels: np.ndarray,
    channel: str,
) -> None:
    """
    Generate the four-class baseline-normalized TFR
    for one channel and subject.
    """
    output_path = get_tfr_path(
        subject=subject,
        channel=channel,
    )

    if output_path.exists() and not OVERWRITE:
        print(
            f"TFR at {channel} already exists:"
            f"\n  {output_path}"
        )
        return

    result = compute_channel_tfr(
        epochs=epochs,
        labels=labels,
        class_ids=CLASS_IDS,
        channel=channel,
        frequencies=TFR_FREQUENCIES,
        baseline=BASELINE,
        decim=TFR_DECIM,
    )

    figure = create_channel_tfr_figure(
        result=result,
        subject=subject,
        imagery_window=IMAGERY_WINDOW,
    )

    try:
        output_file = save_figure(
            figure=figure,
            output_file=output_path,
        )
    finally:
        plt.close(figure)

    print(
        f"Saved TFR at {channel}:"
        f"\n  {output_file}"
    )


def generate_psd(
    subject: int,
    epochs: mne.Epochs,
    labels: np.ndarray,
    channel: str,
) -> None:
    """
    Generate the four-class and baseline PSD
    for one channel and subject.
    """
    output_path = get_psd_path(
        subject=subject,
        channel=channel,
    )

    if output_path.exists() and not OVERWRITE:
        print(
            f"PSD at {channel} already exists:"
            f"\n  {output_path}"
        )
        return

    result = compute_channel_psd(
        epochs=epochs,
        labels=labels,
        class_ids=CLASS_IDS,
        channel=channel,
        baseline=BASELINE,
        imagery_window=IMAGERY_WINDOW,
        fmin=8.0,
        fmax=30.0,
    )

    figure = create_channel_psd_figure(
        result=result,
        subject=subject,
    )

    try:
        output_file = save_figure(
            figure=figure,
            output_file=output_path,
        )
    finally:
        plt.close(figure)

    print(
        f"Saved PSD at {channel}:"
        f"\n  {output_file}"
    )


def validate_labels(
    labels: np.ndarray,
) -> None:
    """
    Check that the labels match the expected four internal classes.
    """
    expected_labels = set(CLASS_IDS.values())
    observed_labels = set(
        np.unique(labels).tolist()
    )

    unexpected_labels = (
        observed_labels - expected_labels
    )

    if unexpected_labels:
        raise ValueError(
            "Unexpected class labels found. "
            f"Expected a subset of {sorted(expected_labels)}, "
            f"but found {sorted(observed_labels)}."
        )


def validate_channels(
    epochs: mne.Epochs,
) -> None:
    """
    Check that all requested analysis channels are available.
    """
    missing_channels = [
        channel
        for channel in CHANNELS
        if channel not in epochs.ch_names
    ]

    if missing_channels:
        raise ValueError(
            "The following requested channels are missing: "
            f"{missing_channels}."
        )


def process_subject(
    subject: int,
) -> None:
    """
    Load and generate all spectral figures for one subject.
    """
    subject_name = f"A{subject:02d}"

    print(
        "\n"
        + "=" * 60
        + f"\nProcessing {subject_name}"
        + "\n"
        + "=" * 60
    )

    epochs, labels = (
        get_evaluation_erd_epochs_for_subject(
            subject
        )
    )

    labels = np.asarray(labels)

    if len(epochs) != len(labels):
        raise ValueError(
            f"{subject_name}: the number of epochs "
            f"({len(epochs)}) does not match the number "
            f"of labels ({len(labels)})."
        )

    validate_labels(labels)
    validate_channels(epochs)

    generate_topographies(
        subject=subject,
        epochs=epochs,
        labels=labels,
    )

    for channel in CHANNELS:
        print(f"\nChannel: {channel}")

        generate_tfr(
            subject=subject,
            epochs=epochs,
            labels=labels,
            channel=channel,
        )

        generate_psd(
            subject=subject,
            epochs=epochs,
            labels=labels,
            channel=channel,
        )


def main() -> None:
    for subject in SUBJECTS:
        process_subject(subject)

    print(
        "\n"
        + "=" * 60
        + "\nSpectral analysis completed."
        + "\n"
        + "=" * 60
    )


if __name__ == "__main__":
    main()