from glob import glob
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plot_for_paper.time_trace_funcs import overlap_pulses_multiple_duty_cycles

plt.style.use("custom")
figsize = (11, 7)
plt.rcParams["figure.figsize"] = figsize
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{upgreek}\usepackage{amsmath}\usepackage{newtxtext}\usepackage{newtxmath}\usepackage{courier}\usepackage{helvet}"
)
# plt.rcParams["font.family"] = "serif"
plt.rcParams["legend.fontsize"] = 24
plt.rcParams["axes.labelsize"] = 48
plt.rcParams["xtick.labelsize"] = 38
plt.rcParams["ytick.labelsize"] = 38
plt.rcParams["legend.title_fontsize"] = 28

paper_dir = "/home/thjalfe/Documents/PhD/Projects/papers/FC_QD/figs"

data_dir = "../../data/"
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # type: ignore
remove_col_idx = 8
colors.pop(remove_col_idx)
save_figs = False

data_dir = "../../data/classical-opt/sig-wl_same-as-qd/pump_traces/"
files = glob(data_dir + "*.pkl")
# 0, 4 for dir out of edfa
file = files[3]
print(f"Loading data from {file}")
with open(file, "rb") as f:
    data_dict = cast(dict, np.load(f, allow_pickle=True))
# plt.plot(data_dict["time_s"][3], data_dict["voltage_V"][3])
# plt.show()


time = data_dict["time_s"]
voltage = data_dict["voltage_V"]
duty_cycles = data_dict["duty_cycles"]
ignore_dc_vals = [0.05, 0.1, 0.125, 0.2, 0.25, 1.0]
ignore_dc_vals = [1.0]
pulse_freq = 1e5
pulse_dict = overlap_pulses_multiple_duty_cycles(
    time, voltage, duty_cycles, ignore_dc_vals
)
duty_cycles_processed = list(pulse_dict.keys())
# make into numpy array
duty_cycles_processed = np.array(duty_cycles_processed)
title_str = file.split("/")[-1].split(".")[0]
fig, ax = plt.subplots()
ax = cast(Axes, ax)
for dc in duty_cycles_processed:
    ax.plot(
        pulse_dict[dc]["time_full_duration"] * 1e6,
        pulse_dict[dc]["mean_pulse_full_duration"],
        label=f"{dc:.3f}",
    )
    ax.fill_between(
        pulse_dict[dc]["time_full_duration"] * 1e6,
        pulse_dict[dc]["mean_pulse_full_duration"]
        - pulse_dict[dc]["std_pulse_full_duration"],
        pulse_dict[dc]["mean_pulse_full_duration"]
        + pulse_dict[dc]["std_pulse_full_duration"],
        alpha=0.5,
    )
# ax.set_title(title_str)
ax.set_xlabel(r"Time [$\mu$s]")
ax.set_ylabel("Voltage [V]")
ax.legend(title="Duty cycle")
plt.show()
# |%%--%%| <Jtdb0IR7wP|62urEvoFaX>
# Raw traces
idx = 0
fig, ax = plt.subplots()
ax = cast(Axes, ax)
ax.plot(time[idx] * 1e6, voltage[idx])
ax.set_xlabel(r"Time [$\mu$s]")
ax.set_ylabel("Voltage [V]")
ax.set_title(f"Raw trace for duty cycle {duty_cycles[idx]}")
plt.show()
# |%%--%%| <JPySzuHkv6|Jtdb0IR7wP>
add_cw = False
duty_cycles_loc = duty_cycles_processed
durations = np.array([dc / pulse_freq + 0.1e-6 for dc in duty_cycles_processed])
time_axs = [pulse_dict[dc]["time_full_duration"] for dc in duty_cycles_processed]

idxs_durations = np.array(
    [np.argmin(np.abs(time_ax - dur)) for time_ax, dur in zip(time_axs, durations)]
)
integral_mean_pulses = np.array(
    [
        np.trapezoid(pulse_dict[dc]["mean_pulse_full_duration"][:idx])
        for dc, idx in zip(duty_cycles_processed, idxs_durations)
    ]
)
if add_cw:
    idx_duration_cw = np.argmin(np.abs(time[-1] - 1 / pulse_freq))
    integral_cw = np.trapezoid(voltage[-1][:idx_duration_cw])
    idx_duration_cw = np.nan
    integral_mean_pulses = np.append(integral_mean_pulses, integral_cw)
    duty_cycles_loc = np.append(duty_cycles_processed, 1.0)

x = 10 * np.log10(1 / duty_cycles_loc)
# x = 1 / duty_cycles_loc
integral_mean_pulses_norm = integral_mean_pulses / integral_mean_pulses[-1]
y = 10 * np.log10((integral_mean_pulses_norm / duty_cycles_loc) ** 2)
# y = integral_mean_pulses / duty_cycles_loc

fig, ax = plt.subplots()
ax = cast(Axes, ax)
ax.plot(
    x,
    y,
    "-o",
)

plt.show()
