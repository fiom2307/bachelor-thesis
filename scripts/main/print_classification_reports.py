from src.pipelines.comparison_pipeline import (
    collect_all_predictions,
)
from src.utils.results import (
    print_classification_report_comparison,
)


def main() -> None:
    y_true, csp_predictions, eegnet_predictions = (
        collect_all_predictions()
    )

    print_classification_report_comparison(
        y_true=y_true,
        csp_predictions=csp_predictions,
        eegnet_predictions=eegnet_predictions,
    )


if __name__ == "__main__":
    main()