from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


CLASS_LABELS = [0, 1, 2, 3]

CLASS_NAMES = [
    "Left hand",
    "Right hand",
    "Feet",
    "Tongue",
]


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