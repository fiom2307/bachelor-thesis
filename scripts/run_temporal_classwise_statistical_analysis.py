import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )

from src.analysis.temporal_statistical_analysis import (
    run_classwise_csp_lda_vs_eegnet_correct_temporal_analysis,
    run_csp_lda_classwise_temporal_statistical_analysis,
)
from src.utils.paths import (
    get_temporal_classwise_csp_vs_eegnet_correct_results_path,
    get_temporal_classwise_statistical_results_path,
)
from src.visualization.temporal_statistical_analysis import (
    plot_classwise_csp_lda_vs_eegnet_correct_temporal_summaries,
    plot_csp_lda_classwise_temporal_statistical_summaries,
)


def main() -> None:
    """
    Run the class-wise temporal relevance follow-up analyses.
    """
    _, csp_condition_rows = (
        run_csp_lda_classwise_temporal_statistical_analysis()
    )

    csp_condition_plot_paths = (
        plot_csp_lda_classwise_temporal_statistical_summaries(
            csp_condition_rows
        )
    )

    _, model_comparison_rows = (
        run_classwise_csp_lda_vs_eegnet_correct_temporal_analysis()
    )

    model_comparison_plot_paths = (
        plot_classwise_csp_lda_vs_eegnet_correct_temporal_summaries(
            model_comparison_rows
        )
    )

    print()
    print("Saved class-wise temporal statistical analysis:")
    print(
        get_temporal_classwise_statistical_results_path()
    )
    print(
        get_temporal_classwise_csp_vs_eegnet_correct_results_path()
    )

    for plot_path in (
        csp_condition_plot_paths
        + model_comparison_plot_paths
    ):
        print(
            plot_path
        )


if __name__ == "__main__":
    main()
