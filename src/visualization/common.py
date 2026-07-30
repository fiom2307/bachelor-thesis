from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


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