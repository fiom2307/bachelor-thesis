from pathlib import Path

import numpy as np

from src.analysis.csp_analysis.csp import (
    CSPAnalysisResult,
)


def save_csp_analysis_result(
    result: CSPAnalysisResult,
    output_file: Path,
) -> None:
    """
    Save computed CSP+LDA occlusion results.
    """
    np.savez_compressed(
        output_file,
        values=result.values,
        probabilities=result.probabilities,
        predictions=result.predictions,
        labels=result.labels,
        times=result.times,
    )


def load_csp_analysis_result(
    input_file: Path,
) -> CSPAnalysisResult:
    """
    Load previously computed CSP+LDA occlusion results.
    """
    with np.load(input_file) as data:
        return CSPAnalysisResult(
            values=data["values"],
            probabilities=data["probabilities"],
            predictions=data["predictions"],
            labels=data["labels"],
            times=data["times"],
        )