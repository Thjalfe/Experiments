import numpy as np

try:
    from .utils import find_idler_wl_approx
except ImportError:
    from utils import find_idler_wl_approx


def mean_std_data(data_dict: dict) -> dict:
    keys = data_dict.keys()
    mean_std_dict = {key: {"wl": None, "mean": None, "std": None} for key in keys}
    for key, value in data_dict.items():
        mean_std_dict[key]["wl"] = value[0]
        mean_std_dict[key]["mean"] = np.mean(value[1:], axis=0)
        mean_std_dict[key]["std"] = np.std(value[1:], axis=0)
    return mean_std_dict


def find_idler_loc(
    data: np.ndarray,
    wl_arr: np.ndarray,
    pump_wls: tuple[float, float],
    idler_tolerance: float = 0.5,
    min_counts_idler: float = 50,
) -> tuple[int, float, float]:
    possible_idler_locs_tmp = np.where(data > min_counts_idler)[0]
    sig_wl = wl_arr[np.argmax(data)]
    idler_wl_approx = find_idler_wl_approx(sig_wl, pump_wls)
    possible_idler_locs = possible_idler_locs_tmp[
        np.abs(wl_arr[possible_idler_locs_tmp] - idler_wl_approx) < idler_tolerance
    ]
    idler_loc = possible_idler_locs[np.argmax(data[possible_idler_locs])]
    idler_wl = float(wl_arr[idler_loc])
    return idler_loc, idler_wl, sig_wl


def calc_ce_from_peak_values(
    data: np.ndarray, ref: np.ndarray, idler_loc: int, duty_cycle: float
) -> float:
    sig_loc = np.argmax(data)
    idler_val = data[idler_loc] - ref[idler_loc]
    sig_val = data[sig_loc]
    ce = idler_val / sig_val
    return ce / duty_cycle


def calc_ce_std_from_peak_values(
    data: np.ndarray,
    ref: np.ndarray,
    idler_loc: int,
    duty_cycle: float,
    data_std: np.ndarray,
) -> float:
    sig_loc = np.argmax(data)
    idler_val = data[idler_loc] - ref[idler_loc]
    sig_val = data[sig_loc]
    ce = idler_val / sig_val / duty_cycle
    ce_std = ce * np.sqrt(
        (data_std[idler_loc] / idler_val) ** 2 + (data_std[sig_loc] / sig_val) ** 2
    )
    return ce_std


def calc_multiple_ces_from_peak_values(
    data: dict, ref: dict, idler_locs: np.ndarray, duty_cycles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ces = np.zeros(len(data))
    std_combined = np.zeros(len(data))
    for i, key in enumerate(data.keys()):
        ces[i] = calc_ce_from_peak_values(
            data[key]["mean"], ref[key]["mean"], idler_locs[i], duty_cycles[i]
        )
        std_combined[i] = calc_ce_std_from_peak_values(
            data[key]["mean"],
            ref[key]["mean"],
            idler_locs[i],
            duty_cycles[i],
            data[key]["std"],
        )
    return ces, std_combined


def find_multiple_idler_locs(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idler_locs = np.zeros(len(data), dtype=int)
    idler_wls = np.zeros(len(data))
    sig_wls = np.zeros(len(data))
    for i, key in enumerate(data.keys()):
        idler_loc, idler_wl, sig_wl = find_idler_loc(
            data[key]["mean"], data[key]["wl"], key
        )
        idler_locs[i] = idler_loc
        idler_wls[i] = idler_wl
        sig_wls[i] = sig_wl

    return idler_locs, idler_wls, sig_wls
