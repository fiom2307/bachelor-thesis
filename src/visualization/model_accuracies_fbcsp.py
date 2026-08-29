import matplotlib.pyplot as plt
import numpy as np


def create_fbcsp_accuracy_comparison(
    subjects: list[str],
    csp_lda_accuracies: list[float],
    fbcsp_lda_accuracies: list[float],
    eegnet_accuracies: list[float],
):
    """
    Create a grouped bar plot comparing CSP+LDA,
    FBCSP+LDA, and EEGNet accuracies.
    """
    subjects_with_mean = subjects + ["Mean"]

    csp_values = np.asarray(
        csp_lda_accuracies,
        dtype=float,
    )

    fbcsp_values = np.asarray(
        fbcsp_lda_accuracies,
        dtype=float,
    )

    eegnet_values = np.asarray(
        eegnet_accuracies,
        dtype=float,
    )

    csp_values = np.append(
        csp_values,
        np.mean(csp_values),
    )

    fbcsp_values = np.append(
        fbcsp_values,
        np.mean(fbcsp_values),
    )

    eegnet_values = np.append(
        eegnet_values,
        np.mean(eegnet_values),
    )

    x = np.arange(
        len(subjects_with_mean)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(13, 6),
    )

    bars_csp = ax.bar(
        x - width,
        csp_values,
        width,
        label="CSP+LDA",
    )

    bars_fbcsp = ax.bar(
        x,
        fbcsp_values,
        width,
        label="FBCSP+LDA",
    )

    bars_eegnet = ax.bar(
        x + width,
        eegnet_values,
        width,
        label="EEGNet",
    )

    ax.set_title(
        "Classification accuracy: "
        "CSP+LDA vs FBCSP+LDA vs EEGNet"
    )

    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy")

    ax.set_xticks(x)
    ax.set_xticklabels(
        subjects_with_mean
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.legend()

    def add_accuracy_labels(
        bars,
    ) -> None:
        """
        Add accuracy labels above bars.

        Labels are placed above the bars instead of directly
        on top of the plotted values to avoid overlap.
        """
        for bar in bars:
            height = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    add_accuracy_labels(
        bars_csp,
    )

    add_accuracy_labels(
        bars_fbcsp,
    )

    add_accuracy_labels(
        bars_eegnet,
    )

    fig.tight_layout()

    return fig