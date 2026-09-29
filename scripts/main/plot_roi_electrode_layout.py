from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import mne
from mne.channels.layout import _find_topomap_coords

from src.utils.paths import (
    get_electrode_layout_pdf_path,
    get_electrode_layout_png_path,
)

PDF_OUTPUT = get_electrode_layout_pdf_path()
PNG_OUTPUT = get_electrode_layout_png_path()

MONTAGE_NAME = "biosemi64"
SFREQ = 250.0

ROI_CHANNELS = {
    "Left ROI": [
        "FC3",
        "FC1",
        "C5",
        "C3",
        "C1",
        "CP3",
        "CP1",
        "P1",
    ],
    "Midline ROI": [
        "Fz",
        "FCz",
        "Cz",
        "CPz",
        "Pz",
        "POz",
    ],
    "Right ROI": [
        "FC2",
        "FC4",
        "C2",
        "C4",
        "C6",
        "CP2",
        "CP4",
        "P2",
    ],
}

ROI_COLORS = {
    "Left ROI": "#8FBBD9",
    "Midline ROI": "#E7C66B",
    "Right ROI": "#D99A8B",
}

CHANNEL_NAMES = [
    channel
    for channels in ROI_CHANNELS.values()
    for channel in channels
]


def _validate_roi_definitions() -> None:
    """Ensure that the ROI definitions contain exactly the intended channels."""
    unique_channels = set(CHANNEL_NAMES)

    if len(CHANNEL_NAMES) != 22:
        raise ValueError(
            f"Expected 22 ROI channel entries, found {len(CHANNEL_NAMES)}."
        )

    if len(unique_channels) != len(CHANNEL_NAMES):
        raise ValueError("ROI channel definitions contain duplicate channels.")

    if set(ROI_CHANNELS) != set(ROI_COLORS):
        raise ValueError("Every ROI must have exactly one plotting color.")


def _create_mne_info() -> mne.Info:
    """Create an MNE Info object using channel locations from a standard montage."""
    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    missing_channels = [
        channel
        for channel in CHANNEL_NAMES
        if channel not in montage.ch_names
    ]

    if missing_channels:
        raise ValueError(
            f"Channels missing from {MONTAGE_NAME}: {missing_channels}"
        )

    info = mne.create_info(
        ch_names=CHANNEL_NAMES,
        sfreq=SFREQ,
        ch_types="eeg",
    )
    info.set_montage(
        montage,
        match_case=True,
    )

    return info


def _get_channel_positions(info: mne.Info) -> dict[str, tuple[float, float]]:
    """Project MNE montage coordinates into the standard 2D topomap plane."""
    picks = list(range(len(CHANNEL_NAMES)))
    coords = _find_topomap_coords(
        info,
        picks=picks,
        sphere=(0.0, 0.0, 0.0, 0.095),
    )

    if coords.shape != (22, 2):
        raise ValueError(
            "Expected projected coordinates with shape (22, 2), "
            f"but received {coords.shape}."
        )

    return {
        channel: tuple(coords[index])
        for index, channel in enumerate(CHANNEL_NAMES)
    }


def _draw_mne_head_outline(
    ax: plt.Axes,
    info: mne.Info,
) -> None:
    """Draw the same head outline style used by MNE topography plots."""
    mne.viz.plot_topomap(
        data=np.zeros(len(CHANNEL_NAMES)),
        pos=info,
        axes=ax,
        show=False,
        sensors=False,
        contours=0,
        cmap=ListedColormap(["white"]),
        vlim=(-1.0, 1.0),
    )


def _plot_electrodes(
    ax: plt.Axes,
    positions: dict[str, tuple[float, float]],
) -> None:
    """Plot ROI-colored electrode markers with labels inside each marker."""
    plotted_channels = []
    marker_radius = 0.0073

    for roi_name, channels in ROI_CHANNELS.items():
        for channel in channels:
            x_coord, y_coord = positions[channel]

            ax.add_patch(
                Circle(
                    (x_coord, y_coord),
                    marker_radius,
                    facecolor=ROI_COLORS[roi_name],
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=3,
                )
            )
            ax.text(
                x_coord,
                y_coord,
                channel,
                ha="center",
                va="center",
                color="black",
                fontsize=7.8,
                zorder=4,
            )

            plotted_channels.append(channel)

    if plotted_channels != CHANNEL_NAMES:
        raise ValueError("Plotted channel order does not match ROI definitions.")

    if len(plotted_channels) != 22:
        raise ValueError(
            f"Expected to plot 22 electrodes, plotted {len(plotted_channels)}."
        )


def _add_legend(ax: plt.Axes) -> None:
    """Add a compact ROI legend."""
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=7,
            label=roi_name,
        )
        for roi_name, color in ROI_COLORS.items()
    ]

    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=3,
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.2,
        borderaxespad=0.0,
    )


def create_roi_electrode_layout() -> plt.Figure:
    """Create the ROI electrode-layout figure."""
    _validate_roi_definitions()

    info = _create_mne_info()
    positions = _get_channel_positions(info)

    fig, ax = plt.subplots(
        figsize=(4.2, 4.25),
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")

    _draw_mne_head_outline(
        ax=ax,
        info=info,
    )
    _plot_electrodes(
        ax=ax,
        positions=positions,
    )
    _add_legend(ax)

    ax.set_xlim(-0.105, 0.105)
    ax.set_ylim(-0.098, 0.124)
    ax.axis("off")

    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = create_roi_electrode_layout()
    figure.savefig(
        PDF_OUTPUT,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    figure.savefig(
        PNG_OUTPUT,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(figure)

    print("ROI electrode layout saved to:")
    print(PDF_OUTPUT)
    print(PNG_OUTPUT)
    print(f"Plotted channels: {len(CHANNEL_NAMES)}")
    for roi_name, channels in ROI_CHANNELS.items():
        print(f"{roi_name}: {', '.join(channels)}")
    print(f"Montage: {MONTAGE_NAME}")


if __name__ == "__main__":
    main()
