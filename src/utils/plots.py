from pathlib import Path
import mne

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.analysis.erd import ERDResult

from src.data.labels import (
    CLASS_LABELS,
    CLASS_NAMES,
)


def _draw_confusion_matrix(
    axis: Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
) -> None:
    """Draw one normalized confusion matrix on an axis."""
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        normalize="true",
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        values_format=".2f",
        xticks_rotation=30,
        im_kw={
            "vmin": 0,
            "vmax": 1,
        },
    )

    axis.set_title(title)


def create_confusion_matrix_comparison(
    y_true: np.ndarray,
    csp_predictions: np.ndarray,
    eegnet_predictions: np.ndarray,
    subject_name: str = "All subjects",
) -> Figure:
    """Create normalized CSP+LDA and EEGNet confusion matrices."""
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    _draw_confusion_matrix(
        axis=axes[0],
        y_true=y_true,
        y_pred=csp_predictions,
        title=f"CSP+LDA — {subject_name}",
    )

    _draw_confusion_matrix(
        axis=axes[1],
        y_true=y_true,
        y_pred=eegnet_predictions,
        title=f"EEGNet — {subject_name}",
    )

    figure.tight_layout()

    return figure


def create_erd_topomap_figure(
    epochs: mne.Epochs,
    results: dict[tuple[str, int], ERDResult],
    band_names: list[str],
    subject: int,
    baseline: tuple[float, float],
    imagery_window: tuple[float, float],
) -> Figure:
    """
    Create class-wise ERD/ERS topomaps for several frequency bands.

    Each row represents one frequency band and each column represents
    one motor-imagery class.
    """
    figure, axes = plt.subplots(
        nrows=len(band_names),
        ncols=len(CLASS_LABELS),
        figsize=(14, 7),
        constrained_layout=True,
    )

    for row_index, band_name in enumerate(band_names):
        band_topographies = [
            results[(band_name, class_id)].topography
            for class_id in CLASS_LABELS
        ]

        # Use the same color scale for all classes within one band.
        band_limit = max(
            float(np.max(np.abs(topography)))
            for topography in band_topographies
        )

        row_image = None

        for column_index, class_id in enumerate(CLASS_LABELS):
            axis = axes[row_index, column_index]

            result = results[(band_name, class_id)]

            row_image, _ = mne.viz.plot_topomap(
                result.topography,
                epochs.info,
                axes=axis,
                show=False,
                sensors=True,
                contours=6,
                cmap="RdBu_r",
                vlim=(-band_limit, band_limit),
            )

            if row_index == 0:
                axis.set_title(
                    CLASS_NAMES[class_id],
                    fontsize=12,
                )

            axis.text(
                0.5,
                -0.12,
                f"n = {result.n_trials}",
                transform=axis.transAxes,
                horizontalalignment="center",
                fontsize=9,
            )

        axes[row_index, 0].text(
            -0.35,
            0.5,
            band_name,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=12,
            fontweight="bold",
        )

        if row_image is not None:
            colorbar = figure.colorbar(
                row_image,
                ax=axes[row_index, :].tolist(),
                shrink=0.75,
            )

            colorbar.set_label(
                "Power change from baseline (%)"
            )

    figure.suptitle(
        f"Subject {subject:02d}: class-wise ERD/ERS topographies\n"
        f"Baseline: {baseline[0]:.1f} to {baseline[1]:.1f} s | "
        f"Imagery average: "
        f"{imagery_window[0]:.1f} to "
        f"{imagery_window[1]:.1f} s",
        fontsize=14,
    )

    return figure


def save_figure(
    figure: Figure,
    output_file: str | Path,
) -> Path:
    """Save a figure as a high-resolution PNG."""
    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    return output_file


def show_figure() -> None:
    """Display the current Matplotlib figure."""
    plt.show()