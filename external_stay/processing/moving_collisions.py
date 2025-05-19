import glob
import os
import pickle
import re
from collections import defaultdict
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from external_stay.processing.helper_funcs import dBm_to_mW, mW_to_dBm
from external_stay.run_exp_funcs.moving_collisions import (
    CollisionMeasurementData,
)


def add_attributes_and_save(filename, **new_attrs):
    with open(filename, "rb") as f:
        obj = pickle.load(f)

    for attr, value in new_attrs.items():
        setattr(obj, attr, value)

    with open(filename, "wb") as f:
        pickle.dump(obj, f)

    print(f"Updated {filename} with: {new_attrs}")


# add_attributes_and_save(
#     filename,
#     yenista_filter_loss_p=6.3,
#     yenista_filter_loss_fwm=5.9,
# )


def extract_wl_p(filename: str) -> str | None:
    match = re.search(r"wl-p=(\d+)", filename)
    return match.group(1) if match else None


def group_files_by_wl_p(filenames: list[str]) -> dict[str, list[str]]:
    grouped = defaultdict(list)
    for fname in filenames:
        wl_p = extract_wl_p(fname)
        if wl_p:
            grouped[wl_p].append(fname)
    return grouped


def glob_files_in_directory(directory: str, suffix: str = ".pkl") -> list[str]:
    return sorted(glob.glob(os.path.join(directory, f"**/*{suffix}"), recursive=True))


def filter_filenames_by_keyword(filenames: list[str], **filters: str) -> list[str]:
    filtered = []
    for fname in filenames:
        if all(
            f"{key.replace('_', '-')}" + "=" + val in fname
            for key, val in filters.items()
        ):
            filtered.append(fname)
    return filtered


def extract_plotting_params(data: CollisionMeasurementData) -> tuple:
    non_zero_idxs = np.where(data.fwm_power != 0)[0]
    non_zero_idx = (
        non_zero_idxs[-1] if len(non_zero_idxs) > 0 else len(data.fwm_power) - 1
    )

    x_ax = (data.fut_Ltot - np.abs(data.coords[: non_zero_idx + 1]))[::-1]
    opt_brill_freqs = data.opt_brill_freqs[: non_zero_idx + 1] * 1e-9
    brill_gain = (
        data.brill_powers[: non_zero_idx + 1]
        - data.ref_brill_powers[: non_zero_idx + 1]
    )
    fwm_pow = data.fwm_power[: non_zero_idx + 1]

    return brill_gain, x_ax, opt_brill_freqs, fwm_pow, data.pm_avg_time


def plot_gain_single(x_ax, brill_gain, opt_brill_freqs):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(x_ax, brill_gain)
    ax2.plot(x_ax, opt_brill_freqs, color="k")
    ax2.grid(False)
    ax.set_xlabel("Collision point [m]")
    ax.set_ylabel("Brillouin gain [dB]")
    ax2.set_ylabel("Brillouin frequency [GHz]")
    return fig, ax


def plot_fwm_power_single(x_ax, fwm_pow):
    fig, ax = plt.subplots()
    ax.plot(x_ax, fwm_pow)
    ax.set_xlabel("Collision point [m]")
    ax.set_ylabel("FWM power [dBm]")
    return fig, ax


def plot_all_gains(x_list, brill_gains, label_list=None):
    fig, ax = plt.subplots()
    for x, gain, label in zip(x_list, brill_gains, label_list or [None] * len(x_list)):
        ax.plot(x, gain, label=label)
    ax.set_xlabel("Collision point [m]")
    ax.set_ylabel("Brillouin gain [dB]")
    if label_list:
        ax.legend()
    return fig, ax


def plot_all_fwm_powers(x_list, fwm_list, label_list=None):
    fig, ax = plt.subplots()
    for x, fwm, label in zip(x_list, fwm_list, label_list or [None] * len(x_list)):
        ax.plot(x, fwm, label=label)
    ax.set_xlabel("Collision point [m]")
    ax.set_ylabel("FWM power [dBm]")
    if label_list:
        ax.legend()
    return fig, ax


def process_collision_files(filenames: Sequence[str]) -> tuple[list, ...]:
    brill_gains = []
    x_axes = []
    opt_brill_freqs = []
    fwm_powers = []

    for fname in filenames:
        with open(fname, "rb") as f:
            data: CollisionMeasurementData = pickle.load(f)
        brill_gain, x_ax, opt_freq, fwm_pow, pm_avg_time = extract_plotting_params(data)
        brill_gains.append(brill_gain)
        x_axes.append(x_ax)
        opt_brill_freqs.append(opt_freq)
        fwm_powers.append(fwm_pow)
    return brill_gains, x_axes, opt_brill_freqs, fwm_powers


def plot_collision_meas(
    x_axes: list,
    brill_gains: list,
    opt_freqs: list,
    fwm_powers: list,
    extra_filename: str | None = None,
    plot_individual: bool = False,
    save_figs: bool = False,
):
    if plot_individual:
        for x, g, f in zip(x_axes, brill_gains, opt_freqs):
            plot_gain_single(x, g, f)

        for x, f in zip(x_axes, fwm_powers):
            plot_fwm_power_single(x, f)

    fig_gain_all, ax_gain_all = plot_all_gains(x_axes, brill_gains)

    fig_fwm_all, ax_fwm_all = plot_all_fwm_powers(x_axes, fwm_powers)
    if save_figs:
        fig_gain_all.savefig(
            f"{fig_dir}/fwm_power_all_{extra_filename}.pdf", bbox_inches="tight"
        )
        fig_fwm_all.savefig(
            f"{fig_dir}/brill_gain_all_{extra_filename}.pdf", bbox_inches="tight"
        )


plt.style.use("custom")
fig_dir = "/home/thjalfe/Documents/PhD/logbook/2025/may/figs/exp"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

save_figs = False
plot_individual = False

filenames = glob_files_in_directory("../data/moving_collisions/HNDS1615AAA-6-3-2")
filenames = glob_files_in_directory("../data/moving_collisions/HNDS1599CA-6-5/")
filenames = filenames[:3]

grouped_files = group_files_by_wl_p(filenames)
brill_gains_dict = {}
x_axes_dict = {}
opt_freqs_dict = {}
fwm_powers_dict = {}

for wl_p, files in grouped_files.items():
    brill_gains, x_axes, opt_freqs, fwm_powers = process_collision_files(files)
    brill_gains_dict[wl_p] = brill_gains
    x_axes_dict[wl_p] = x_axes
    opt_freqs_dict[wl_p] = opt_freqs
    fwm_powers_dict[wl_p] = fwm_powers

    # Optional: plot and save
    fig_gain, ax_gain = plot_all_gains(x_axes, brill_gains)
    ax_gain.set_title(rf"$\lambda_s=1530$, $\lambda_p={wl_p}$")
    fig_fwm, ax_fwm = plot_all_fwm_powers(x_axes, fwm_powers)
    ax_fwm.set_title(rf"$\lambda_s=1530$, $\lambda_p={wl_p}$")

    if save_figs:
        fig_gain.savefig(f"{fig_dir}/brill_gain_{wl_p}.pdf", bbox_inches="tight")
        fig_fwm.savefig(f"{fig_dir}/fwm_power_{wl_p}.pdf", bbox_inches="tight")

plt.show()
# # |%%--%%| <RFxJv2oALb|QoIrXUxAtS>
#
# dir = "../data/brillioun_amp/Fiber=HNDS1615AAA-6-3-2_257meter/moving_collision/"
# name = "only_sweep_collision_point_w_pol-scrambling_brill-freq-opt_pump-duration=3.0ns_len-scanned=50.0m_stepsize-m=-0.2_no-osa_only-pm_wait-before-pm=1.0s.pkl"
# with open(f"{dir}/{name}", "rb") as handle:
#     data = pickle.load(handle)
# collision_coordinates_full_overlap = data["seq1_start_idxs"]
# brill_powers = data["brill_powers"]
# gain = brill_powers - data["ref_brill_power"]
# # brill_std = np.std(data["brill_powers"], axis=1)
# try:
#     opt_brill_freqs = data["opt_brill_freqs"] * 10**-9
# except KeyError:
#     opt_brill_freqs = data["opt_brill_wls"] * 10**-9
#     data["opt_brill_freqs"] = data.pop("opt_brill_wls")
#     with open(f"{filename}", "wb") as handle:
#         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# fig, ax = plt.subplots()
# ax2 = ax.twinx()
# ax.plot(
#     collision_coordinates_full_overlap,
#     gain,
# )
# ax2.plot(collision_coordinates_full_overlap[1:], opt_brill_freqs[1:], color="k")
# ax2.grid(False)
# ax2.set_ylim(np.min(opt_brill_freqs[1:]) - 0.1, np.max(opt_brill_freqs[1:]) + 0.1)
# ax2.set_ylabel("Brilloin frequency [GHz]")
# ax.set_xlabel("Collision point [m]")
# ax.set_ylabel("Brillouin gain [dB]")
# fig, ax = plt.subplots()
# ax.plot(collision_coordinates_full_overlap, data["fwm_power"])
# ax.set_xlabel("Collision point [m]")
# ax.set_ylabel("FWM power [dBm]")
# plt.show()
