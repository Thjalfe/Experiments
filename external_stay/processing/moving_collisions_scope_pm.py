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
    CollisionMeasurementDataScope,
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


def extract_plotting_params(data: CollisionMeasurementDataScope) -> tuple:
    non_zero_idxs = np.where(data.fwm_power != 0)[0]
    non_zero_idx = (
        non_zero_idxs[-1] if len(non_zero_idxs) > 0 else len(data.fwm_power) - 1
    )

    x_ax = (data.fut_Ltot - np.abs(data.coords[: non_zero_idx + 1]))[::-1]
    x_ax_remainder = np.abs(data.coords[: non_zero_idx + 1][-1]) - np.abs(
        data.coords[-1]
    )
    x_ax = x_ax + x_ax_remainder
    opt_brill_freqs = data.opt_brill_freqs[: non_zero_idx + 1] * 1e-9
    pump_power = 10 * np.log10(data.brill_powers[: non_zero_idx + 1])
    fwm_pow = 10 * np.log10(data.fwm_power[: non_zero_idx + 1])

    return pump_power, x_ax, opt_brill_freqs, fwm_pow


def plot_pump_power(x_ax, pump_power, opt_brill_freqs):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(x_ax, pump_power)
    ax2.plot(x_ax, opt_brill_freqs, color="k")
    ax2.grid(False)
    ax.set_xlabel("Collision point [m]")
    ax.set_ylabel("Pump power [dBm]")
    ax2.set_ylabel("Brillouin frequency [GHz]")
    return fig, ax


def plot_fwm_power_single(x_ax, fwm_pow):
    fig, ax = plt.subplots()
    ax.plot(x_ax, fwm_pow)
    ax.set_xlabel("Collision point [m]")
    ax.set_ylabel("FWM power [dBm]")
    return fig, ax


def plot_all_pump_powers(x_list, pump_powers, label_list=None):
    fig, ax = plt.subplots()
    for x, power, label in zip(x_list, pump_powers, label_list or [None] * len(x_list)):
        ax.plot(x, power, label=label)
    ax.set_xlabel("Collision point [m]")
    ax.set_ylabel("Pump power [dBm]")
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
    pump_powers = []
    x_axes = []
    opt_brill_freqs = []
    fwm_powers = []
    flip_list = []

    for fname in filenames:
        with open(fname, "rb") as f:
            data: CollisionMeasurementDataScope = pickle.load(f)
        pump_power, x_ax, opt_freq, fwm_pow = extract_plotting_params(data)
        pump_powers.append(pump_power)
        x_axes.append(x_ax)
        opt_brill_freqs.append(opt_freq)
        fwm_powers.append(fwm_pow)
        if "flipped" in fname:
            flip_list.append(True)
        else:
            flip_list.append(False)
    return pump_powers, x_axes, opt_brill_freqs, fwm_powers, flip_list


def plot_collision_meas(
    x_axes: list,
    pump_powers: list,
    opt_freqs: list,
    fwm_powers: list,
    extra_filename: str | None = None,
    plot_individual: bool = False,
    save_figs: bool = False,
):
    if plot_individual:
        for x, g, f in zip(x_axes, pump_powers, opt_freqs):
            plot_pump_power(x, g, f)

        for x, f in zip(x_axes, fwm_powers):
            plot_fwm_power_single(x, f)

    fig_pump_power_all, ax_pump_power_all = plot_all_pump_powers(x_axes, pump_powers)

    fig_fwm_all, ax_fwm_all = plot_all_fwm_powers(x_axes, fwm_powers)
    if save_figs:
        fig_pump_power_all.savefig(
            f"{fig_dir}/fwm_power_all_{extra_filename}.pdf", bbox_inches="tight"
        )
        fig_fwm_all.savefig(
            f"{fig_dir}/pump_power_all_{extra_filename}.pdf", bbox_inches="tight"
        )


plt.style.use("custom")
fig_dir = "/home/thjalfe/Documents/PhD/logbook/2025/may/figs/exp"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

save_figs = False
plot_individual = False

filenames = glob_files_in_directory(
    "../data/moving_collisions/scope_pm/fully-from-end-fiber/HNDS1599CA-6-5"
)
# filenames = filenames[-5:]

grouped_files = group_files_by_wl_p(filenames)
pump_powers_dict = {}
x_axes_dict = {}
opt_freqs_dict = {}
fwm_powers_dict = {}
flip_list_dict = {}

for wl_p, files in grouped_files.items():
    pump_powers, x_axes, opt_freqs, fwm_powers, flip_list = process_collision_files(
        files
    )
    pump_powers_dict[wl_p] = pump_powers
    x_axes_dict[wl_p] = x_axes
    opt_freqs_dict[wl_p] = opt_freqs
    fwm_powers_dict[wl_p] = fwm_powers
    flip_list_dict[wl_p] = flip_list

    # Optional: plot and save
    fig_pump_power, ax_pump_power = plot_all_pump_powers(x_axes, pump_powers)
    ax_pump_power.set_title(rf"$\lambda_s=1530$, $\lambda_p={wl_p}$")
    fig_fwm, ax_fwm = plot_all_fwm_powers(x_axes, fwm_powers)
    ax_fwm.set_title(rf"$\lambda_s=1530$, $\lambda_p={wl_p}$")

    if save_figs:
        fig_pump_power.savefig(f"{fig_dir}/pump_power{wl_p}.pdf", bbox_inches="tight")
        fig_fwm.savefig(f"{fig_dir}/fwm_power_{wl_p}.pdf", bbox_inches="tight")

plt.show()
# |%%--%%| <b83soLygEW|E0KFjdcJeZ>
wlp = "1560"
pump_powers = pump_powers_dict[wlp]
x_axes = x_axes_dict[wlp]
x_ax_offset = np.min([np.min(x_ax) for x_ax in x_axes])
fwm_powers = fwm_powers_dict[wlp]
flip_list = flip_list_dict[wlp]
fig, ax = plt.subplots()
for i in range(len(pump_powers)):
    x_ax = x_axes[i]
    x_ax = x_ax - x_ax_offset - 4
    fwm_pow = fwm_powers[i]
    label = "Normal"
    if flip_list[i]:
        x_ax = x_ax[::-1]
        # Force to start at 0 if flipped
        x_ax = x_ax - np.min(x_ax)
        label = "Flipped"

    ax.plot(x_ax, fwm_pow, label=label)
ax.set_xlabel("Collision point [m]")
ax.set_ylabel("FWM power [dBm]")
ax.set_ylim(-112, ax.get_ylim()[1])
ax.legend()
plt.show()


# |%%--%%| <E0KFjdcJeZ|08zk2RZT79>
def pulse_area(
    x: np.ndarray, y: np.ndarray, threshold: float = 0.1, tail_fraction: float = 0.01
) -> float:
    y = np.asarray(y)
    x = np.asarray(x)

    n = len(y)
    tail_len = int(n * tail_fraction)

    # Estimate and remove DC baseline
    baseline = np.mean(np.concatenate([y[:tail_len], y[-tail_len:]]))
    y_corr = y - baseline

    # Find pulse region
    max_val = np.max(y_corr)
    assert isinstance(max_val, float)
    threshold = threshold * max_val
    mask = y_corr > threshold
    if not np.any(mask):
        return 0.0
    dx = x[1] - x[0]
    return np.sum(y_corr[mask]) * dx


def voltage_to_optical_power(
    volt: float, responsivity: float, load_resistance: float = 50
):
    return volt / (load_resistance * responsivity)


raw_dataname = filenames[0]
with open(f"{raw_dataname}", "rb") as handle:
    data = pickle.load(handle)
p_waveforms = data.pump_waveforms
fwm_waveforms = data.fwm_waveforms
idx = -1
fwm_waveform = fwm_waveforms[idx]
p_waveform = p_waveforms[idx]
fig, ax = plt.subplots()
ax.plot(fwm_waveform[0], fwm_waveform[1])
fig, ax = plt.subplots()
for i in range(0, len(fwm_waveforms), 10):
    fwm_waveform = fwm_waveforms[i]
    ax.plot(fwm_waveform[0], fwm_waveform[1], label=i)
ax.legend()
pows = np.zeros(len(fwm_waveforms))
max_pows = np.zeros(len(fwm_waveforms))

for i in range(len(fwm_waveforms)):
    fwm_waveform = fwm_waveforms[i]
    pows[i] = pulse_area(fwm_waveform[0], fwm_waveform[1])
    max_pows[i] = np.max(fwm_waveform[1])
fig, ax = plt.subplots()
ax.plot(10 * np.log10(pows / np.max(pows)))
ax.plot(10 * np.log10(max_pows / np.max(max_pows)))
plt.show()
