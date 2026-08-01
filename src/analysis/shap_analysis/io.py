from pathlib import Path

import numpy as np

from src.analysis.shap_analysis.eegnet import SHAPResult


def save_shap_result(
    result: SHAPResult,
    output_file: Path,
) -> None:
    """Save computed SHAP results."""
    np.savez_compressed(
        output_file,
        values=result.values,
        probabilities=result.probabilities,
        predictions=result.predictions,
        labels=result.labels,
    )


def load_shap_result(
    input_file: Path,
) -> SHAPResult:
    """Load previously computed SHAP results."""
    with np.load(input_file) as data:
        return SHAPResult(
            values=data["values"],
            probabilities=data["probabilities"],
            predictions=data["predictions"],
            labels=data["labels"],
        )