from typing import cast
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from funcs.load_data import get_data_filenames, load_data_as_dict
from funcs.processing import (
    find_multiple_idler_locs,
    mean_std_data,
)
from funcs.utils import pump_wls_to_thz_sep

plt.style.use("custom")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # type: ignore

data_dir = "../../data/"


def lorentzian_fit(x: np.ndarray, amp: float, wid: float, cen: float) -> np.ndarray:
    return amp * (wid**2 / ((x - cen) ** 2 + wid**2))


ref_file_names, data_file_names = get_data_filenames(data_dir)
ref_data, duty_cycles = load_data_as_dict(ref_file_names)
data, duty_cycles = load_data_as_dict(data_file_names)
pump_wls_tuple = list(data.keys())
thz_sep = np.array(
    [pump_wls_to_thz_sep(pump_wls_tuple[i]) for i in range(len(pump_wls_tuple))]
)
pump_wls_arr = np.array(pump_wls_tuple)
ref_data_mean_std = mean_std_data(ref_data)
data_mean_std = mean_std_data(data)
idler_locs, idler_wls, sig_wls = find_multiple_idler_locs(data_mean_std)


def get_lorentzian_fit(
    wls: np.ndarray,
    counts: np.ndarray,
    center_loc: int | np.int64,
    factor_around_center: float,
    nm_around_center_peak: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    def get_lorentzian_fit_params(
        wls: np.ndarray,
        counts: np.ndarray,
        center_loc: int | np.int64,
        nm_around_center_peak: float,
    ) -> np.ndarray:
        valid_idxs = np.where(
            (wls >= wls[center_loc] - nm_around_center_peak)
            & (wls <= wls[center_loc] + nm_around_center_peak)
        )

        center = wls[center_loc]
        amp = counts[center_loc]
        wid = 0.05
        wls = wls[valid_idxs]
        counts = counts[valid_idxs]

        popt, _ = curve_fit(
            lorentzian_fit,
            wls,
            counts,
            p0=[amp, wid, center],
        )
        return popt

    popt = get_lorentzian_fit_params(wls, counts, center_loc, nm_around_center_peak)
    center_wl = popt[2]
    wl_ax_fit = np.linspace(
        center_wl - factor_around_center * popt[1],
        center_wl + factor_around_center * popt[1],
        1000,
    )
    fit = lorentzian_fit(wl_ax_fit, *popt)
    return wl_ax_fit, fit, popt


pump_wl_idx = 0
idler_loc = idler_locs[pump_wl_idx]
data_local = data_mean_std[pump_wls_tuple[pump_wl_idx]]
wls = data_local["wl"]
counts = data_local["mean"]
# wls = data[pump_wls_tuple[pump_wl_idx]][0, :]
# counts = data[pump_wls_tuple[pump_wl_idx]][3, :]

signal_loc = np.argmax(counts)
# fit = lorentzian_fit(wl_ax_fit, *popt)
idler_wl_ax_fit, idler_fit, idler_popt = get_lorentzian_fit(
    wls, counts, idler_loc, 4, 0.05
)
signal_wl_ax_fit, signal_fit, signal_popt = get_lorentzian_fit(
    wls, counts, signal_loc, 15, 0.05
)
idler_wl_lims = idler_popt[2] + np.array([-1, 1]) * idler_popt[1]
signal_wl_lims = signal_popt[2] + np.array([-1, 1]) * signal_popt[1]
idler_loc_lims = np.where(
    (idler_wl_ax_fit >= idler_wl_lims[0]) & (idler_wl_ax_fit <= idler_wl_lims[1])
)
signal_loc_lims = np.where(
    (signal_wl_ax_fit >= signal_wl_lims[0]) & (signal_wl_ax_fit <= signal_wl_lims[1])
)
idler_int = np.trapz(idler_fit[idler_loc_lims], idler_wl_ax_fit[idler_loc_lims])
signal_int = np.trapz(signal_fit[signal_loc_lims], signal_wl_ax_fit[signal_loc_lims])
ce_from_fit = idler_int / signal_int / duty_cycles[pump_wl_idx]
idler_loc_lims_raw = np.where((wls >= idler_wl_lims[0]) & (wls <= idler_wl_lims[1]))
signal_loc_lims_raw = np.where((wls >= signal_wl_lims[0]) & (wls <= signal_wl_lims[1]))
idler_pow = np.sum(counts[idler_loc_lims_raw]) / len(idler_loc_lims_raw)
signal_pow = np.sum(counts[signal_loc_lims_raw]) / len(signal_loc_lims_raw)
ce = idler_pow / signal_pow / duty_cycles[pump_wl_idx]
print(f"ce_from_fit: {ce_from_fit}")
print(f"ce: {ce}")
fig, ax = plt.subplots()
ax = cast(Axes, ax)
ax.plot(data_local["wl"], data_local["mean"])
ax.fill_between(
    data_local["wl"],
    data_local["mean"] - data_local["std"],
    data_local["mean"] + data_local["std"],
    alpha=0.5,
    color="C0",
)
ax.plot(idler_wl_ax_fit, idler_fit)
ax.plot(wls[idler_loc_lims_raw], counts[idler_loc_lims_raw], "ro")
ax.plot(signal_wl_ax_fit, signal_fit)
ax.plot(wls[signal_loc_lims_raw], counts[signal_loc_lims_raw], "ro")
plt.show()


# |%%--%%| <3EyXXoIr1y|ACiIZFFznX>


def get_lorentzian_fit_from_multiple_spectra(
    data_dict: dict,
    pump_wls: list[tuple[float, float]],
    sig_or_idler: str,
    factor_around_center_for_wl_ax: float,
    nm_around_center_peak: float,
) -> dict:
    if sig_or_idler not in ["sig", "idler"]:
        raise ValueError(f"Invalid value for sig_or_idler: {sig_or_idler}")

    fit_dict = {
        pump_wl: {"wl_ax": np.array([]), "fit": np.array([]), "popt": np.array([])}
        for pump_wl in pump_wls
    }

    for pump_wl in pump_wls:
        data_local = data_dict[pump_wl]
        wls = data_local["wl"]
        counts = data_local["mean"]

        loc = None
        if sig_or_idler == "idler":
            loc = idler_locs[pump_wls.index(pump_wl)]
        elif sig_or_idler == "sig":
            loc = np.argmax(counts)

        if loc is None:
            raise ValueError(f"Location could not be determined for pump_wl: {pump_wl}")

        wl_ax, fit, popt = get_lorentzian_fit(
            wls, counts, loc, factor_around_center_for_wl_ax, nm_around_center_peak
        )

        fit_dict[pump_wl]["wl_ax"] = wl_ax
        fit_dict[pump_wl]["fit"] = fit
        fit_dict[pump_wl]["popt"] = popt

    return fit_dict


sig_fit_dict = get_lorentzian_fit_from_multiple_spectra(
    data_mean_std, pump_wls_tuple, "sig", 4, 0.10
)
idler_fit_dict = get_lorentzian_fit_from_multiple_spectra(
    data_mean_std, pump_wls_tuple, "idler", 4, 0.10
)


def calc_ce_from_lorentzian_fit(
    idler_fit: np.ndarray,
    signal_fit: np.ndarray,
    idler_fit_params: np.ndarray,
    signal_fit_params: np.ndarray,
    idler_wl_ax_fit: np.ndarray,
    signal_wl_ax_fit: np.ndarray,
    raw_data: np.ndarray,
    raw_wl_ax: np.ndarray,
    raw_std: np.ndarray,
    ref_data: np.ndarray,
    duty_cycle: float,
) -> tuple:
    idler_wl_lims = idler_fit_params[2] + np.array([-1, 1]) * idler_fit_params[1]
    signal_wl_lims = signal_fit_params[2] + np.array([-1, 1]) * signal_fit_params[1]
    idler_loc_lims = np.where(
        (idler_wl_ax_fit >= idler_wl_lims[0]) & (idler_wl_ax_fit <= idler_wl_lims[1])
    )
    signal_loc_lims = np.where(
        (signal_wl_ax_fit >= signal_wl_lims[0])
        & (signal_wl_ax_fit <= signal_wl_lims[1])
    )
    idler_int = np.trapezoid(idler_fit[idler_loc_lims], idler_wl_ax_fit[idler_loc_lims])
    signal_int = np.trapezoid(
        signal_fit[signal_loc_lims], signal_wl_ax_fit[signal_loc_lims]
    )
    idler_loc_lims_raw = np.where(
        (raw_wl_ax >= idler_wl_lims[0]) & (raw_wl_ax <= idler_wl_lims[1])
    )[0]
    signal_loc_lims_raw = np.where(
        (raw_wl_ax >= signal_wl_lims[0]) & (raw_wl_ax <= signal_wl_lims[1])
    )[0]
    i_pts = list(idler_loc_lims_raw)
    s_pts = list(signal_loc_lims_raw)
    while len(s_pts) > len(i_pts):
        # drop the lowest‐value s word:
        drop = s_pts[np.argmin(raw_data[s_pts])]
        s_pts.remove(drop)
    while len(i_pts) > len(s_pts):
        drop = i_pts[np.argmin(raw_data[i_pts] - ref_data[i_pts])]
        i_pts.remove(drop)
    idler_counts = np.sum(raw_data[i_pts] - ref_data[i_pts])
    signal_counts = np.sum(raw_data[s_pts])
    ratio = idler_counts / signal_counts
    ce_raw = ratio / (duty_cycle * (1 + ratio))
    idler_var = np.sum(raw_std[idler_loc_lims_raw] ** 2)
    signal_var = np.sum(raw_std[signal_loc_lims_raw] ** 2)
    n_idler = len(idler_loc_lims_raw)
    n_signal = len(signal_loc_lims_raw)

    idler_pow_std = np.sqrt(idler_var) / n_idler
    signal_pow_std = np.sqrt(signal_var) / n_signal
    rel_idler_std = idler_pow_std / idler_pow
    rel_signal_std = signal_pow_std / signal_pow

    ce_std = ce_raw * np.sqrt(rel_idler_std**2 + rel_signal_std**2)
    idler_peak_wl = raw_wl_ax[
        np.argmax(raw_data[idler_loc_lims_raw]) + idler_loc_lims_raw[0]
    ]

    return idler_int / signal_int / duty_cycle, ce_raw, ce_std, idler_peak_wl


def calc_ce_from_lorentzian_fit_nomean_data(
    idler_fit_params: np.ndarray,
    signal_fit_params: np.ndarray,
    raw_data: np.ndarray,
    raw_wl_ax: np.ndarray,
    ref_data: np.ndarray,
    duty_cycle: float,
) -> tuple:
    idler_wl_lims = idler_fit_params[2] + np.array([-1, 1]) * idler_fit_params[1]
    signal_wl_lims = signal_fit_params[2] + np.array([-1, 1]) * signal_fit_params[1]
    idler_loc_lims_raw = np.where(
        (raw_wl_ax >= np.min(idler_wl_lims)) & (raw_wl_ax <= np.max(idler_wl_lims))
    )[0]
    signal_loc_lims_raw = np.where(
        (raw_wl_ax >= np.min(signal_wl_lims)) & (raw_wl_ax <= np.max(signal_wl_lims))
    )[0]
    i_pts = list(idler_loc_lims_raw)
    s_pts = list(signal_loc_lims_raw)
    while len(s_pts) > len(i_pts):
        # drop the lowest‐value s word:
        drop = s_pts[np.argmin(raw_data[s_pts])]
        s_pts.remove(drop)
    while len(i_pts) > len(s_pts):
        drop = i_pts[np.argmin(raw_data[i_pts] - ref_data[i_pts])]
        i_pts.remove(drop)
    idler_counts = np.sum(raw_data[i_pts] - ref_data[i_pts])
    signal_counts = np.sum(raw_data[s_pts])
    ratio = idler_counts / signal_counts
    ce_raw = ratio / (duty_cycle * (1 + ratio))

    idler_peak_wl = raw_wl_ax[
        np.argmax(raw_data[idler_loc_lims_raw]) + idler_loc_lims_raw[0]
    ]

    return ce_raw, idler_peak_wl


def calc_multiple_ces_from_lotentzian_fit_nomean_data(
    data_raw: dict,
    ref_data: dict,
    idler_locs_init: np.ndarray,
    duty_cycles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ces_raw = np.zeros(len(idler_fit_dict))
    ces_std = np.zeros(len(idler_fit_dict))
    idler_wls = np.zeros(len(idler_fit_dict))
    for i, pump_wl in enumerate(data_raw.keys()):
        sub_data = data_raw[pump_wl]
        wl_raw_tmp = sub_data[0]
        ces_raw_tmp, idler_wls_tmp = [], []
        idler_loc_init = idler_locs_init[i]
        ref_data_tmp = ref_data[pump_wl]["mean"]
        for j in range(1, len(sub_data)):
            data_raw_tmp = sub_data[j]
            signal_loc = np.argmax(data_raw_tmp)
            idler_loc = (
                np.argmax(data_raw_tmp[idler_loc_init - 3 : idler_loc_init + 3])
                + idler_loc_init
            )
            idler_wl_ax_fit, idler_fit, idler_popt = get_lorentzian_fit(
                wl_raw_tmp, data_raw_tmp, idler_loc, 4, 0.07
            )

            signal_wl_ax_fit, signal_fit, signal_popt = get_lorentzian_fit(
                wl_raw_tmp, data_raw_tmp, signal_loc, 4, 0.07
            )
            ce_raw_tmp, idler_wl_tmp = calc_ce_from_lorentzian_fit_nomean_data(
                idler_popt,
                signal_popt,
                data_raw_tmp,
                wl_raw_tmp,
                ref_data_tmp,
                duty_cycles[i],
            )
            ces_raw_tmp.append(ce_raw_tmp)
            idler_wls_tmp.append(idler_wl_tmp)
        ces_raw[i] = np.mean(ces_raw_tmp)
        ces_std[i] = np.std(ces_raw_tmp)
        idler_wls[i] = np.mean(idler_wls_tmp)
    return ces_raw, ces_std, idler_wls


def calc_multiple_ces_from_lotentzian_fit(
    idler_fit_dict: dict,
    signal_fit_dict: dict,
    data_raw: dict,
    ref_data: dict,
    duty_cycles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ces = np.zeros(len(idler_fit_dict))
    ces_raw = np.zeros(len(idler_fit_dict))
    ces_std = np.zeros(len(idler_fit_dict))
    idler_wls = np.zeros(len(idler_fit_dict))
    for i, pump_wl in enumerate(idler_fit_dict.keys()):
        idler_fit = idler_fit_dict[pump_wl]["fit"]
        signal_fit = signal_fit_dict[pump_wl]["fit"]
        idler_popt = idler_fit_dict[pump_wl]["popt"]
        signal_popt = signal_fit_dict[pump_wl]["popt"]
        idler_wl_ax_fit = idler_fit_dict[pump_wl]["wl_ax"]
        signal_wl_ax_fit = signal_fit_dict[pump_wl]["wl_ax"]
        wl_raw_tmp = data_raw[pump_wl]["wl"]
        data_raw_tmp = data_raw[pump_wl]["mean"]
        data_raw_std_tmp = data_raw[pump_wl]["std"]
        ref_data_tmp = ref_data[pump_wl]["mean"]
        ces[i], ces_raw[i], ces_std[i], idler_wls[i] = calc_ce_from_lorentzian_fit(
            idler_fit,
            signal_fit,
            idler_popt,
            signal_popt,
            idler_wl_ax_fit,
            signal_wl_ax_fit,
            data_raw_tmp,
            wl_raw_tmp,
            data_raw_std_tmp,
            ref_data_tmp,
            duty_cycles[i],
        )
    return ces, ces_raw, ces_std, idler_wls


ces, ces_raw, ces_std, idler_wls = calc_multiple_ces_from_lotentzian_fit(
    idler_fit_dict, sig_fit_dict, data_mean_std, ref_data_mean_std, duty_cycles
)
ces_raw2, ces_std2, idler_wls2 = calc_multiple_ces_from_lotentzian_fit_nomean_data(
    data, ref_data_mean_std, idler_locs, duty_cycles
)
blue_idxs = np.where(thz_sep < 0)
red_idxs = np.where(thz_sep > 0)
# red_idxs = red_idxs[0][2:]
blue_ces = ces[blue_idxs]
blue_ces_raw = ces_raw[blue_idxs]
# blue_stds = stds[blue_idxs]
red_ces = ces[red_idxs]
red_ces_raw = ces_raw[red_idxs]
# red_stds = stds[red_idxs]
ces_raw = ces_raw * 100
print(ces_raw)
print(ces_raw2 * 100)
stds = ces_std * 100
data_to_save = {
    "thz_sep": thz_sep,
    "idler_wls": idler_wls,
    "ces": ces_raw,
    "stds": stds,
    "sig_wl": 971.2885,
}

with open("../../data/ce_data_from_qd_calc-from-lorentz-fit.pkl", "wb") as f:
    pickle.dump(data_to_save, f)

# |%%--%%| <ACiIZFFznX|AsOU5HX192>
# idx = 2
# fig, ax = plt.subplots()
# ax = cast(Axes, ax)
# ax.plot(
#     data_mean_std[pump_wls_tuple[idx]]["wl"], data_mean_std[pump_wls_tuple[idx]]["mean"]
# )
# ax.plot(
#     idler_fit_dict[pump_wls_tuple[idx]]["wl_ax"],
#     idler_fit_dict[pump_wls_tuple[idx]]["fit"],
# )
# plt.show()
thz_sep_tmp = np.array([0.402, 0.841, 1.329, 1.784, 2.24])
ce_tmp = np.array([-0.848, -1.476, -3.718, -8.8, -8.582])
fig, ax = plt.subplots()
fig = cast(Figure, fig)
ax = cast(Axes, ax)
fig.tight_layout()
# ax.plot(thz_sep_tmp, ce_tmp, "-o", color="k", label="Classical")
ax.plot(
    np.abs(thz_sep[red_idxs]),
    # 10 * np.log10(red_ces_raw),
    red_ces_raw * 100,
    "o",
    color=colors[3],
    label="Red-shifted idler",
)
ax.plot(
    np.abs(thz_sep[blue_idxs]),
    # 10 * np.log10(blue_ces),
    # 10 * np.log10(blue_ces_raw),
    blue_ces_raw * 100,
    "o",
    label="Blue-shifted idler",
    color=colors[0],
)
ax.set_xlabel(r"$|\nu_\mathrm{s}-\nu_\mathrm{i}|$ [THz]")
ax.set_ylabel(r"$\eta_\mathrm{peak}$ [dB]")
ax.legend()
plt.show()
