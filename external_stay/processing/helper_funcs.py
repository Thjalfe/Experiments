import numpy as np


def rolling_average(data: np.ndarray, window_size: int):
    cumsum = np.nancumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1 :] / window_size


def dBm_to_mW(dBm: np.ndarray | float) -> np.ndarray | float:
    return 10 ** (dBm / 10)


def mW_to_dBm(mW: np.ndarray | float) -> np.ndarray | float:
    return 10 * np.log10(mW)
