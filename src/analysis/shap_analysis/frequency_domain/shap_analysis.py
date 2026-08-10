from dataclasses import dataclass

import numpy as np
import shap
from tensorflow import keras

from src.analysis.shap_analysis._utils import (
    prepare_model_input,
)
from src.utils.config import BASE_SEED


FrequencyBand = tuple[float, float]


FREQUENCY_BANDS: tuple[
    FrequencyBand,
    ...,
] = (
    (8.0, 10.0),
    (10.0, 12.0),
    (12.0, 14.0),
    (14.0, 16.0),
    (16.0, 18.0),
    (18.0, 20.0),
    (20.0, 22.0),
    (22.0, 24.0),
    (24.0, 26.0),
    (26.0, 28.0),
    (28.0, 30.0),
)


@dataclass(frozen=True)
class FrequencySHAPResult:
    """
    Frequency-domain SHAP results for one subject.
    """

    values: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray
    frequency_bands: tuple[
        FrequencyBand,
        ...,
    ]

    @property
    def correct_mask(self) -> np.ndarray:
        """Return correctly classified trials."""
        return self.predictions == self.labels

    @property
    def incorrect_mask(self) -> np.ndarray:
        """Return incorrectly classified trials."""
        return self.predictions != self.labels


def compute_eegnet_frequency_shap(
    models: list[keras.Model],
    data: np.ndarray,
    labels: np.ndarray,
    sfreq: float,
    frequency_bands: tuple[
        FrequencyBand,
        ...,
    ] = FREQUENCY_BANDS,
    nsamples: int = 256,
    batch_size: int = 16,
    seed: int = BASE_SEED,
) -> FrequencySHAPResult:
    """
    Compute frequency-band SHAP values for an EEGNet ensemble.
    """
    data = np.asarray(
        data,
        dtype=np.float32,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    fold_probabilities = []

    for model in models:
        model_data, _ = prepare_model_input(
            model,
            data,
        )

        probabilities = model.predict(
            model_data,
            batch_size=batch_size,
            verbose=0,
        )

        fold_probabilities.append(
            probabilities
        )

    mean_probabilities = np.mean(
        fold_probabilities,
        axis=0,
    )

    predictions = mean_probabilities.argmax(
        axis=1
    )

    n_bands = len(
        frequency_bands
    )

    shap_values = np.empty(
        (
            len(data),
            n_bands,
        ),
        dtype=np.float32,
    )

    background_mask = np.zeros(
        (
            1,
            n_bands,
        ),
        dtype=np.float32,
    )

    full_mask = np.ones(
        (
            1,
            n_bands,
        ),
        dtype=np.float32,
    )

    random_state = np.random.get_state()

    np.random.seed(
        seed
    )

    try:
        for trial_index, (
            trial,
            class_id,
        ) in enumerate(
            zip(
                data,
                labels,
                strict=True,
            )
        ):
            band_components, residual = (
                decompose_frequency_bands(
                    trial=trial,
                    sfreq=sfreq,
                    frequency_bands=frequency_bands,
                )
            )

            def predict_from_masks(
                masks: np.ndarray,
            ) -> np.ndarray:
                return predict_ensemble_from_masks(
                    models=models,
                    masks=masks,
                    band_components=band_components,
                    residual=residual,
                    batch_size=batch_size,
                )

            explainer = shap.KernelExplainer(
                predict_from_masks,
                background_mask,
                link="identity",
            )

            values = explainer.shap_values(
                full_mask,
                nsamples=nsamples,
                l1_reg=0,
                silent=True,
            )

            shap_values[
                trial_index
            ] = extract_class_shap_values(
                values=values,
                class_id=int(class_id),
            )

    finally:
        np.random.set_state(
            random_state
        )

    return FrequencySHAPResult(
        values=shap_values,
        predictions=predictions,
        labels=labels,
        frequency_bands=frequency_bands,
    )


def decompose_frequency_bands(
    trial: np.ndarray,
    sfreq: float,
    frequency_bands: tuple[
        FrequencyBand,
        ...,
    ],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose one EEG trial into frequency-band components.
    """
    n_times = trial.shape[-1]

    frequencies = np.fft.rfftfreq(
        n_times,
        d=1.0 / sfreq,
    )

    spectrum = np.fft.rfft(
        trial,
        axis=-1,
    )

    band_components = []

    for band_index, (
        low,
        high,
    ) in enumerate(
        frequency_bands
    ):
        if band_index == len(
            frequency_bands
        ) - 1:
            frequency_mask = (
                (frequencies >= low)
                & (frequencies <= high)
            )
        else:
            frequency_mask = (
                (frequencies >= low)
                & (frequencies < high)
            )

        band_spectrum = np.zeros_like(
            spectrum
        )

        band_spectrum[
            :,
            frequency_mask,
        ] = spectrum[
            :,
            frequency_mask,
        ]

        band_component = np.fft.irfft(
            band_spectrum,
            n=n_times,
            axis=-1,
        )

        band_components.append(
            band_component.astype(
                np.float32
            )
        )

    band_components = np.stack(
        band_components,
        axis=0,
    )

    residual = (
        trial
        - band_components.sum(
            axis=0
        )
    ).astype(
        np.float32
    )

    return (
        band_components,
        residual,
    )


def predict_ensemble_from_masks(
    models: list[keras.Model],
    masks: np.ndarray,
    band_components: np.ndarray,
    residual: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """
    Reconstruct masked EEG signals and predict with the ensemble.
    """
    masks = np.asarray(
        masks,
        dtype=np.float32,
    )

    reconstructed_data = (
        residual[
            np.newaxis,
            :,
            :,
        ]
        + np.einsum(
            "mb,bct->mct",
            masks,
            band_components,
        )
    ).astype(
        np.float32
    )

    fold_probabilities = []

    for model in models:
        model_data, _ = prepare_model_input(
            model,
            reconstructed_data,
        )

        probabilities = model.predict(
            model_data,
            batch_size=batch_size,
            verbose=0,
        )

        fold_probabilities.append(
            probabilities
        )

    return np.mean(
        fold_probabilities,
        axis=0,
    )


def extract_class_shap_values(
    values: object,
    class_id: int,
) -> np.ndarray:
    """
    Extract SHAP values for the true class.
    """
    if isinstance(
        values,
        list,
    ):
        values = values[
            class_id
        ]

        return np.asarray(
            values[0],
            dtype=np.float32,
        )

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    return values[
        0,
        :,
        class_id,
    ]