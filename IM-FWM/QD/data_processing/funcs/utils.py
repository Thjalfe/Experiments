import numpy as np
from scipy.signal import butter, filtfilt


def find_idler_wl_approx(sig_wl: float, pump_wls: tuple[float, float]) -> float:
    return 1 / (1 / sig_wl + 1 / pump_wls[0] - 1 / pump_wls[1])


def pump_wls_to_thz_sep(pump_wls: tuple[float, float], c=299792458) -> float:
    pump_wl1 = pump_wls[0] * 10**-9
    pump_wl2 = pump_wls[1] * 10**-9
    pump_freq1 = c / pump_wl1
    pump_freq2 = c / pump_wl2
    return (pump_freq2 - pump_freq1) * 10**-12


def sort_by_ascending_x_ax(
    x: np.ndarray, y: np.ndarray, y_err: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idxs = np.argsort(x)
    return x[idxs], y[idxs], y_err[idxs]


def get_exp_fit(
    x: np.ndarray,
    y: np.ndarray,
    new_x_ax: np.ndarray,
    idxs_to_ignore: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if idxs_to_ignore is not None:
        x = np.delete(x, idxs_to_ignore)
        y = np.delete(y, idxs_to_ignore)
    exp_fit = np.polyfit(x, np.log(y), 1)
    exp_fit_fn = np.exp(exp_fit[1]) * np.exp(exp_fit[0] * new_x_ax)
    return exp_fit_fn


def rolling_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="same")


def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def apply_butter_lowpass_filter(data, cutoff, fs, order=4, axis=0):
    b, a = butter_lowpass(cutoff, fs, order=order)
    filtered_data = filtfilt(b, a, data, axis=axis)
    return filtered_data
