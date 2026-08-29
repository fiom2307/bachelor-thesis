import numpy as np
from mne.decoding import CSP
from mne.filter import filter_data

from src.models.csp import apply_csp, fit_csp
from src.utils.config import CSP_N_COMPONENTS


DEFAULT_FREQUENCY_BANDS = (
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 30.0),
)

DEFAULT_SFREQ = 250.0


class FBCSP:
    """
    Filter Bank Common Spatial Pattern.

    The EEG epochs are filtered into several frequency bands.
    A separate CSP model is fitted for every frequency band.

    During transformation, the CSP log-power features from all
    frequency bands are concatenated into one feature vector.
    """

    def __init__(
        self,
        frequency_bands: tuple[
            tuple[float, float],
            ...
        ] = DEFAULT_FREQUENCY_BANDS,
        n_components: int = CSP_N_COMPONENTS,
        sfreq: float = DEFAULT_SFREQ,
    ) -> None:
        self.frequency_bands = frequency_bands
        self.n_components = n_components
        self.sfreq = sfreq

        self.csps: list[CSP] = []

    def _filter_band(
        self,
        X: np.ndarray,
        fmin: float,
        fmax: float,
    ) -> np.ndarray:
        """
        Band-pass filter EEG epochs for one filter-bank band.

        X must have shape:
            (n_trials, n_channels, n_times)
        """
        X = np.asarray(
            X,
            dtype=np.float64,
        )

        return filter_data(
            X,
            sfreq=self.sfreq,
            l_freq=fmin,
            h_freq=fmax,
            method="iir",
            iir_params={
                "order": 4,
                "ftype": "butter",
            },
            phase="zero",
            verbose=False,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "FBCSP":
        """
        Fit one CSP model for each frequency band.
        """
        self.csps = []

        for fmin, fmax in self.frequency_bands:
            X_band = self._filter_band(
                X,
                fmin,
                fmax,
            )

            csp = fit_csp(
                X_band,
                y,
                self.n_components,
            )

            self.csps.append(csp)

        return self

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Transform EEG epochs into concatenated FBCSP features.

        If there are B frequency bands and C CSP components,
        the output has B * C features per trial.
        """
        if len(self.csps) == 0:
            raise RuntimeError(
                "FBCSP must be fitted before transform()."
            )

        if len(self.csps) != len(self.frequency_bands):
            raise RuntimeError(
                "The number of fitted CSP models does not match "
                "the number of frequency bands."
            )

        features = []

        for (
            (fmin, fmax),
            csp,
        ) in zip(
            self.frequency_bands,
            self.csps,
        ):
            X_band = self._filter_band(
                X,
                fmin,
                fmax,
            )

            X_csp = apply_csp(
                csp,
                X_band,
            )

            features.append(X_csp)

        return np.concatenate(
            features,
            axis=1,
        )

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Fit the complete filter bank and transform the data.
        """
        self.fit(X, y)

        return self.transform(X)