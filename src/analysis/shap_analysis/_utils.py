import numpy as np
from tensorflow import keras


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