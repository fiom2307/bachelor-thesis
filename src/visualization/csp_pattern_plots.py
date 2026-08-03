import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.figure import Figure


def plot_csp_patterns(
    patterns: np.ndarray,
    info: mne.Info,
    subject: int | None = None,
) -> Figure:
    """
    Plot the mean CSP spatial pattern for each component.

    Parameters
    ----------
    patterns
        CSP spatial patterns with shape:

        (n_components, n_channels)

    info
        MNE information containing channel names, channel types,
        and electrode locations.

    subject
        Subject number. If None, no subject identifier is included
        in the figure title.

    Returns
    -------
    Figure
        Figure containing one topography per CSP component.
    """
    patterns = np.asarray(
        patterns,
        dtype=float,
    )

    _validate_patterns(
        patterns=patterns,
        info=info,
    )

    n_components = patterns.shape[0]

    n_columns = min(
        n_components,
        4,
    )

    n_rows = int(
        np.ceil(
            n_components / n_columns
        )
    )

    figure, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_columns,
        figsize=(
            3.2 * n_columns,
            3.5 * n_rows,
        ),
        squeeze=False,
        constrained_layout=True,
    )

    flat_axes = axes.ravel()

    value_limit = _compute_symmetric_limit(
        patterns
    )

    image = None

    for component_index in range(
        n_components
    ):
        axis = flat_axes[
            component_index
        ]

        image, _ = mne.viz.plot_topomap(
            data=patterns[
                component_index
            ],
            pos=info,
            axes=axis,
            show=False,
            sensors=True,
            contours=6,
            cmap="RdBu_r",
            vlim=(
                -value_limit,
                value_limit,
            ),
        )

        axis.set_title(
            f"Component {component_index + 1}"
        )

    for unused_axis in flat_axes[
        n_components:
    ]:
        unused_axis.set_visible(
            False
        )

    if image is None:
        raise ValueError(
            "No CSP components were available "
            "for plotting."
        )

    figure.colorbar(
        image,
        ax=[
            axis
            for axis in flat_axes[
                :n_components
            ]
        ],
        shrink=0.75,
        label="CSP pattern amplitude",
    )

    if subject is None:
        title = (
            "Mean CSP spatial patterns "
            "across folds"
        )
    else:
        title = (
            f"A{subject:02d} — "
            "mean CSP spatial patterns "
            "across folds"
        )

    figure.suptitle(
        title,
        fontsize=14,
    )

    return figure


def _validate_patterns(
    patterns: np.ndarray,
    info: mne.Info,
) -> None:
    """
    Validate CSP pattern dimensions and values.
    """
    if patterns.ndim != 2:
        raise ValueError(
            "patterns must have shape "
            "(n_components, n_channels)."
        )

    n_components, n_channels = (
        patterns.shape
    )

    if n_components == 0:
        raise ValueError(
            "At least one CSP component "
            "is required."
        )

    if n_channels != len(
        info["ch_names"]
    ):
        raise ValueError(
            "The number of CSP pattern channels "
            "does not match the MNE Info object: "
            f"{n_channels} pattern channels versus "
            f"{len(info['ch_names'])} Info channels."
        )

    if not np.all(
        np.isfinite(
            patterns
        )
    ):
        raise ValueError(
            "CSP patterns contain non-finite values."
        )


def _compute_symmetric_limit(
    patterns: np.ndarray,
) -> float:
    """
    Compute a common symmetric color limit for all components.
    """
    value_limit = float(
        np.max(
            np.abs(
                patterns
            )
        )
    )

    if np.isclose(
        value_limit,
        0.0,
    ):
        return 1.0

    return value_limit