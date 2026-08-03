from pathlib import Path

import numpy as np

from src.analysis.csp_pattern_analysis.patterns import (
    CSPPatternResult,
)


def save_csp_pattern_result(
    result: CSPPatternResult,
    output_file: str | Path,
) -> Path:
    """
    Save CSP spatial-pattern results as a compressed NPZ file.
    """
    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_file,
        mean_patterns=result.mean_patterns,
        fold_patterns=result.fold_patterns,
    )

    return output_file


def load_csp_pattern_result(
    input_file: str | Path,
) -> CSPPatternResult:
    """
    Load CSP spatial-pattern results from an NPZ file.
    """
    input_file = Path(
        input_file
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"CSP pattern file does not exist: "
            f"{input_file}"
        )

    with np.load(
        input_file,
        allow_pickle=False,
    ) as data:
        required_keys = {
            "mean_patterns",
            "fold_patterns",
        }

        missing_keys = (
            required_keys
            - set(
                data.files
            )
        )

        if missing_keys:
            raise ValueError(
                "Invalid CSP pattern file. "
                f"Missing arrays: "
                f"{sorted(missing_keys)}"
            )

        mean_patterns = np.asarray(
            data["mean_patterns"],
            dtype=float,
        )

        fold_patterns = np.asarray(
            data["fold_patterns"],
            dtype=float,
        )

    if mean_patterns.ndim != 2:
        raise ValueError(
            "Saved mean_patterns must have shape "
            "(n_components, n_channels)."
        )

    if fold_patterns.ndim != 3:
        raise ValueError(
            "Saved fold_patterns must have shape "
            "(n_folds, n_components, n_channels)."
        )

    if (
        fold_patterns.shape[1:]
        != mean_patterns.shape
    ):
        raise ValueError(
            "Saved mean and fold CSP patterns "
            "have incompatible shapes."
        )

    return CSPPatternResult(
        mean_patterns=mean_patterns,
        fold_patterns=fold_patterns,
    )