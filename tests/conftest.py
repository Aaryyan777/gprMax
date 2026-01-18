
import os
import pytest
import h5py
import numpy as np

@pytest.fixture(scope="session")
def basic_models_path():
    return os.path.join(os.path.dirname(__file__), 'models_basic')

def calculate_max_diff(dataref, datatest):
    """Utility to calculate the max difference in dB between two datasets."""
    datadiffs = np.zeros(datatest.shape, dtype=np.float64)
    for i in range(datatest.shape[1]):
        max_val = np.amax(np.abs(dataref[:, i]))
        # Relative difference
        diff = np.divide(np.abs(dataref[:, i] - datatest[:, i]), max_val, 
                         out=np.zeros_like(dataref[:, i]), where=max_val != 0)
        
        # Calculate power (dB)
        with np.errstate(divide='ignore'):
            datadiffs[:, i] = 20 * np.log10(diff)
        
        # Replace -inf (perfect match) with -200dB
        datadiffs[:, i][np.isneginf(datadiffs[:, i])] = -200.0
        # Replace any other non-finite values (NaNs from 0/0) with -200dB
        datadiffs[:, i][np.invert(np.isfinite(datadiffs[:, i]))] = -200.0
        
    return np.amax(datadiffs)
