import numpy as np
import shap
from tensorflow import keras


def compute_model_shap(
    model: keras.Model,
    background_data: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    nsamples: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute true-class SHAP values for one EEGNet model.
    """
    background_data = np.asarray(
        background_data,
        dtype=np.float32,
    )

    data = np.asarray(
        data,
        dtype=np.float32,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    model_background, added_axis = prepare_model_input(
        model,
        background_data,
    )

    model_data, _ = prepare_model_input(
        model,
        data,
    )

    probabilities = model.predict(
        model_data,
        batch_size=batch_size,
        verbose=0,
    )

    shap_values = np.empty_like(
        model_data,
        dtype=np.float32,
    )

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(
            labels == class_id
        )

        class_output = model.output[:, int(class_id)]

        explainer = shap.GradientExplainer(
            (
                model.inputs[0],
                class_output,
            ),
            model_background,
        )

        for start in range(
            0,
            len(class_indices),
            batch_size,
        ):
            batch_indices = class_indices[
                start : start + batch_size
            ]

            batch_data = model_data[
                batch_indices
            ]

            batch_shap_values = explainer.shap_values(
                batch_data,
                nsamples=nsamples,
                rseed=(
                    seed
                    + int(class_id) * 1000
                    + start
                ),
            )

            shap_values[batch_indices] = (
                unwrap_shap_values(
                    batch_shap_values,
                    expected_shape=batch_data.shape,
                )
            )

    if added_axis is not None:
        shap_values = np.squeeze(
            shap_values,
            axis=added_axis,
        )

    return (
        shap_values,
        np.asarray(probabilities),
    )


def prepare_model_input(
    model: keras.Model,
    data: np.ndarray,
) -> tuple[np.ndarray, int | None]:
    """
    Add EEGNet's singleton input dimension when necessary.
    """
    expected_shape = tuple(
        model.input_shape[1:]
    )

    if data.shape[1:] == expected_shape:
        return data, None

    for axis in range(
        1,
        data.ndim + 1,
    ):
        candidate = np.expand_dims(
            data,
            axis=axis,
        )

        candidate_shape = candidate.shape[1:]

        matches = all(
            expected is None
            or actual == expected
            for actual, expected in zip(
                candidate_shape,
                expected_shape,
                strict=True,
            )
        )

        if matches:
            return candidate, axis

    raise ValueError(
        f"Cannot adapt data shape {data.shape} "
        f"to model input shape {model.input_shape}."
    )


def unwrap_shap_values(
    values: object,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Remove SHAP's additional single-output dimension."""
    if isinstance(values, list):
        values = values[0]

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.shape == (*expected_shape, 1):
        values = values[..., 0]

    return values