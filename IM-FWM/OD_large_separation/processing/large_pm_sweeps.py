import glob
from matplotlib.ticker import MaxNLocator

import os
import pickle
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np

plt.style.use("custom")
data_super_dir = "../data/"
data_files = glob.glob(f"{data_super_dir}**/*.pkl", recursive=True)
sub_str_req = os.path.join(data_super_dir, "telecom-wl")
data_filenames = [
    pathname
    for pathname in data_files
    if sub_str_req in pathname and "old" not in pathname
]
data_list = []
for filename in data_filenames:
    with open(filename, "rb") as f:
        data_list.append(pickle.load(f))
fig_super_dir = (
    "/home/thjalfe/Documents/PhD/logbook/2025/march/figs/large_sep_distant_bs"
)


def merge_two_datasets(data1: dict, data2: dict):
    data_merged = {}
    nonempty_specs_1_count = 0
    nonempty_specs_2_count = 0
    if np.mean(data2["tisa_wl_array"]) > np.mean(data1["tisa_wl_array"]):
        # orders by tisa wl
        data1, data2 = data2, data1
    for key, val in data1.items():
        # check if val is dict
        if key == "spectra":
            # delete all empty entries for data2
            dummy_data1_specs = {}
            dummy_data2_specs = {}
            nonempty_specs_1_count = 0
            nonempty_specs_2_count = 0
            for key2, val2 in val.items():
                if len(val2) > 0:
                    dummy_data1_specs[key2] = val2
                    nonempty_specs_1_count += 1
            for key2, val2 in data2[key].items():
                if len(val2) > 0:
                    dummy_data2_specs[key2] = val2
                    nonempty_specs_2_count += 1
            data_merged[key] = {**dummy_data1_specs, **dummy_data2_specs}
        elif key == "optimum_vals_dict":
            data_merged[key] = {}
            for key2, val2 in val.items():
                data_merged[key][key2] = val2 + data2[key][key2]
        elif key == "toptica_wls":
            data_merged[key] = val + data2[key]
        elif key == "tisa_wl_array":
            data_merged[key] = np.concatenate(
                (val[:nonempty_specs_1_count], data2[key][:nonempty_specs_2_count])
            )
        elif key == "telecom_wl":
            data_merged[key] = val
    return data_merged


def plot_raw_spectra_at_specific_idler(
    measured_data: dict,
    which_spec: str,
    idler_pow_thresh=-68,
    fig_super_dir: str | None = None,
):
    extra_dir_str = f"telecom_wl={measured_data['telecom_wl']:.1f}nm"
    tisa_wls = measured_data["tisa_wl_array"]
    idler_idx = None
    if which_spec == "max_idler_pow":
        idler_idx = np.argmax(measured_data["optimum_vals_dict"]["idler_powers"])
    elif which_spec == "max_sep":
        idler_powers = np.array(
            measured_data["optimum_vals_dict"]["idler_powers"]
        ).astype(float)
        idler_idxs_within_thresh = np.where(idler_powers > idler_pow_thresh)[0]
        idler_idx = idler_idxs_within_thresh[-1]
    spectra_max_idler = measured_data["spectra"][tisa_wls[idler_idx]]
    fig, ax = plt.subplots()
    ax = cast(Axes, ax)
    for i, spec in enumerate(spectra_max_idler):
        spec[1, spec[1] < -80] = np.nan
        ax.plot(spec[0], spec[1])
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Power [dBm]")
    ax.set_title(
        rf"$\lambda_q,\lambda_p$={tisa_wls[idler_idx]}, {measured_data['telecom_wl']:.1f} nm"
    )
    if fig_super_dir is not None:
        fig_dir = os.path.join(fig_super_dir, extra_dir_str)
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(
            os.path.join(fig_dir, f"raw_spectra_{which_spec}.pdf"), bbox_inches="tight"
        )
        plt.close(fig)


import numpy as np


def plot_relevant_results(measured_data: dict, fig_super_dir: str | None = None):
    def tisa_to_idler(x):
        return np.interp(
            x,
            measured_data["tisa_wl_array"],
            measured_data["optimum_vals_dict"]["idler_wls"],
        )

    def idler_to_tisa(x):
        return np.interp(
            x,
            measured_data["optimum_vals_dict"]["idler_wls"],
            measured_data["tisa_wl_array"],
        )

    extra_dir_str = f"telecom_wl={measured_data['telecom_wl']:.1f}nm"
    fig_dir = None
    if fig_super_dir is not None:
        fig_dir = os.path.join(fig_super_dir, extra_dir_str)
        os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax = cast(Axes, ax)
    ax.plot(
        measured_data["tisa_wl_array"],
        measured_data["optimum_vals_dict"]["toptica_wls"],
    )
    ax.set_xlabel(r"$\lambda_\mathrm{q}$ [nm]")
    ax.set_ylabel(r"$\lambda_\mathrm{s}$ [nm]")
    ax.set_title(
        rf"Telecom wl: {measured_data['telecom_wl']:.1f} nm, $\lambda_\mathrm{{s}}$ maximizing idler"
    )
    if fig_dir is not None:
        fig.savefig(os.path.join(fig_dir, "q_s_phase-match.pdf"), bbox_inches="tight")
    fig, ax = plt.subplots()
    ax = cast(Axes, ax)
    ax.plot(
        measured_data["tisa_wl_array"],
        measured_data["optimum_vals_dict"]["idler_powers"],
    )
    ax.set_xlim(measured_data["tisa_wl_array"][-1], measured_data["tisa_wl_array"][0])
    ax.set_xlabel(r"$\lambda_\mathrm{q}$ [nm]")
    ax.set_ylabel("Idler power [dBm]")
    ax.set_title(
        rf"Telecom wl: {measured_data['telecom_wl']:.1f} nm, at max $\lambda_\mathrm{{s}}$"
    )
    tisa_min, tisa_max = ax.get_xlim()
    ax2 = ax.secondary_xaxis("top", functions=(tisa_to_idler, idler_to_tisa))
    ax2.set_xlim(idler_to_tisa(tisa_min), idler_to_tisa(tisa_max))
    ax2.set_xlabel(r"$\lambda_\mathrm{i}$ (nm)")
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))

    if fig_dir is not None:
        fig.savefig(os.path.join(fig_dir, "idler_pow.pdf"), bbox_inches="tight")
    fig, ax = plt.subplots()
    ax = cast(Axes, ax)
    ax.plot(
        measured_data["tisa_wl_array"],
        measured_data["optimum_vals_dict"]["idler_wls"],
    )
    ax.set_xlabel(r"$\lambda_\mathrm{s}$ [nm]")
    ax.set_ylabel(r"$\lambda_\mathrm{i}$ [nm]")
    ax.set_title(
        rf"Telecom wl: {measured_data['telecom_wl']:.1f} nm, at max $\lambda_\mathrm{{s}}$"
    )
    if fig_dir is not None:
        fig.savefig(os.path.join(fig_dir, "s_i_phase-match.pdf"), bbox_inches="tight")


for i, data in enumerate(data_list):
    plot_raw_spectra_at_specific_idler(
        data, "max_idler_pow", fig_super_dir=fig_super_dir
    )
    plot_raw_spectra_at_specific_idler(data, "max_sep", fig_super_dir=fig_super_dir)
    plot_relevant_results(data, fig_super_dir=fig_super_dir)
