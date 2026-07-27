import numpy as np

from src.analysis.erd import compute_all_erd_results
from src.data.dataset import (
    get_evaluation_erd_epochs_for_subject,
)
from src.utils.paths import get_erd_topomaps_path
from src.utils.plots import (
    create_erd_topomap_figure,
    save_figure,
)


SUBJECT = 1

FREQUENCY_BANDS = {
    "Mu (8–13 Hz)": (8.0, 13.0),
    "Beta (13–30 Hz)": (13.0, 30.0),
}

BASELINE = (-1.5, -0.5)
IMAGERY_WINDOW = (0.5, 3.5)


def main() -> None:
    epochs, labels = (
        get_evaluation_erd_epochs_for_subject(
            SUBJECT
        )
    )

    print(f"Subject: {SUBJECT:02d}")
    print(f"Epoch shape: {epochs.get_data().shape}")
    print(
        f"Epoch interval: "
        f"{epochs.tmin:.1f} to {epochs.tmax:.1f} s"
    )
    print(f"Channels: {epochs.ch_names}")
    print(
        "Trials per class: "
        f"{np.bincount(labels, minlength=4)}"
    )

    output_path = get_erd_topomaps_path(SUBJECT)
    
    if output_path.exists():
        print(
            "\nThe all-subject confusion matrices already exist:"
            f"\n{output_path}"
        )
        return

    results = compute_all_erd_results(
        epochs=epochs,
        labels=labels,
        frequency_bands=FREQUENCY_BANDS,
        baseline=BASELINE,
        imagery_window=IMAGERY_WINDOW,
    )

    figure = create_erd_topomap_figure(
        epochs=epochs,
        results=results,
        band_names=list(FREQUENCY_BANDS),
        subject=SUBJECT,
        baseline=BASELINE,
        imagery_window=IMAGERY_WINDOW,
    )

    output_file = save_figure(
        figure=figure,
        output_file=output_path,
    )

    print(f"Saved ERD figure to: {output_file}")


if __name__ == "__main__":
    main()