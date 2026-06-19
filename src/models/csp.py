import numpy as np
from mne.decoding import CSP

def fit_csp(X_train: np.ndarray, y_train: np.ndarray) -> CSP:
    csp = CSP(
        n_components=4, #to check
        reg="ledoit_wolf", 
        log=True, 
        norm_trace=False
        )
    
    # learns spatial filters
    csp.fit(X_train, y_train)
    
    return csp


def apply_csp(csp: CSP, X: np.ndarray) -> np.ndarray:
    return csp.transform(X) # applies spatial filters