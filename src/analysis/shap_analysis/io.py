from pathlib import Path

import numpy as np

from src.analysis.shap_analysis.frequency_domain.shap_analysis import (
    FrequencySHAPResult,
)
from src.analysis.shap_analysis.time_domain.shap_analysis import (
    TimeDomainSHAPResult,
)


def save_time_domain_shap_result(
    result: TimeDomainSHAPResult,
    output_file: str | Path,
) -> Path:
    """
    Save time-domain SHAP results.
    """
    output_file = Path(output_file)

    np.savez_compressed(
        output_file,
        values=result.values,
        predictions=result.predictions,
        labels=result.labels,
    )

    return output_file


def load_time_domain_shap_result(
    input_file: str | Path,
) -> TimeDomainSHAPResult:
    """
    Load time-domain SHAP results.
    """
    input_file = Path(input_file)

    with np.load(
        input_file,
    ) as data:
        return TimeDomainSHAPResult(
            values=data["values"],
            predictions=data["predictions"],
            labels=data["labels"],
        )


def save_frequency_domain_shap_result(
    result: FrequencySHAPResult,
    output_file: str | Path,
) -> Path:
    """
    Save frequency-domain SHAP results.
    """
    output_file = Path(output_file)

    np.savez_compressed(
        output_file,
        values=result.values,
        predictions=result.predictions,
        labels=result.labels,
        frequency_bands=np.asarray(
            result.frequency_bands,
            dtype=float,
        ),
    )

    return output_file


def load_frequency_domain_shap_result(
    input_file: str | Path,
) -> FrequencySHAPResult:
    """
    Load frequency-domain SHAP results.
    """
    input_file = Path(input_file)

    with np.load(
        input_file,
    ) as data:
        frequency_bands = tuple(
            (
                float(low),
                float(high),
            )
            for low, high in data[
                "frequency_bands"
            ]
        )

        return FrequencySHAPResult(
            values=data["values"],
            predictions=data["predictions"],
            labels=data["labels"],
            frequency_bands=frequency_bands,
        )