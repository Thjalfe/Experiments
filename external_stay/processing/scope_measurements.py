import os
from dataclasses import dataclass
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")
from run_exp_funcs.helper_funcs import find_square_region


@dataclass
class SquarePulseStatisticsMultipleBiases:
    voltage_array: np.ndarray
    mean_trace: np.ndarray
    mean_pulse: np.ndarray
    mean_pulse_each_timeloc: np.ndarray
    std_pulse_each_timeloc: np.ndarray
    mean_noise: np.ndarray
    mean_noise_each_timeloc: np.ndarray
    std_noise_each_timeloc: np.ndarray
    mean_divided: np.ndarray
    mean_normalized: np.ndarray
    mean_divided_normalized: np.ndarray
    left_idx: int
    right_idx: int
    noise_idxs: np.ndarray

    def __post_init__(self):
        self.bias_max_pulse = self.voltage_array[np.argmax(self.mean_pulse)]


def get_pulse_statistics(
    trace: np.ndarray,
    voltage_array: np.ndarray,
    percentage_to_cut_both_sides: float = 0.05,
) -> SquarePulseStatisticsMultipleBiases:
    left_idx, right_idx = find_square_region(trace[0, 0, 1, :], threshold_ratio=0.7)
    total_idx = right_idx - left_idx
    cut_idx = int(total_idx * percentage_to_cut_both_sides)
    left_idx += cut_idx
    right_idx -= cut_idx
    noise_idxs = np.arange(right_idx + cut_idx * 5, len(time_ax) - 1)

    mean_trace = np.mean(trace[:, :, 1, :], axis=1)
    mean_pulse = np.mean(mean_trace[:, left_idx:right_idx], axis=1)
    mean_pulse_each_timeloc = np.mean(trace[:, :, 1, left_idx:right_idx], axis=1)
    std_pulse_each_timeloc = np.std(trace[:, :, 1, left_idx:right_idx], axis=1)
    mean_noise = np.mean(mean_trace[:, noise_idxs], axis=1)
    mean_noise_each_timeloc = np.mean(trace[:, :, 1, noise_idxs], axis=1)
    std_noise_each_timeloc = np.std(trace[:, :, 1, noise_idxs], axis=1)
    mean_divided = mean_pulse / np.mean(std_pulse_each_timeloc, axis=1)
    mean_normalized = mean_pulse / np.max(mean_pulse)
    mean_divided_normalized = mean_divided / np.max(mean_divided)

    return SquarePulseStatisticsMultipleBiases(
        voltage_array,
        mean_trace,
        mean_pulse,
        mean_pulse_each_timeloc,
        std_pulse_each_timeloc,
        mean_noise,
        mean_noise_each_timeloc,
        std_noise_each_timeloc,
        mean_divided,
        mean_normalized,
        mean_divided_normalized,
        left_idx,
        right_idx,
        noise_idxs,
    )


def max_bias_all_mzms(data: dict):
    max_biases = []
    for i in range(3):
        trace = data["traces"][f"trace_{i+1}"]
        voltage_arr = data["voltage_arrays"][i]
        pulse_statistics = get_pulse_statistics(
            trace, voltage_arr, percentage_to_cut_both_sides
        )
        max_biases.append(pulse_statistics.bias_max_pulse)
    return np.round(max_biases, 4)


if __name__ == "__main__":
    save_figs = False
    fig_dir = "/home/thjalfe/Documents/PhD/logbook/2025/march/figs/week4"
    plt.style.use("custom")
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    filename = "../data/mzm_bias_sweep/rf_in/scope-traces_bias-varied-close-to-peak_all-3-mzms.pkl"
    filename = "../data/mzm_bias_sweep/rf_in/scope-traces_bias-varied-close-to-peak_all-3-mzms_10ns-pulses.pkl"
    ref_electric_filename = (
        "../data/ref_pulse_directly_from_pulse-generator_dc=1:500.csv"
    )
    ref_electric_after_amp_filename = (
        "../data/ref_pulse_after-electric-amp_from_pulse-generator_dc=1:500.csv"
    )
    ref_data = np.loadtxt(ref_electric_filename)
    ref_data_amped = np.loadtxt(ref_electric_after_amp_filename)
    with open(filename, "rb") as f:
        data = pickle.load(f)

    trace_idx = 2
    percentage_to_cut_both_sides = 0.05
    trace = data["traces"][f"trace_{trace_idx+1}"]
    time_ax = (
        data["traces"][f"trace_{trace_idx+1}"][0, 0, 0, :] * 10**9
    )  # all have same time ax
    voltage_arr = data["voltage_arrays"][trace_idx]
    pulse_statistics = get_pulse_statistics(
        trace, voltage_arr, percentage_to_cut_both_sides
    )
    print(pulse_statistics.bias_max_pulse)

    plot_every_nth = 20
    fig1, ax = plt.subplots()
    for i in range(0, len(voltage_arr), plot_every_nth):
        ax.plot(
            time_ax,
            pulse_statistics.mean_trace[i],
            label=f"{voltage_arr[i]:.2f} V",
        )
        ax.fill_between(
            time_ax[pulse_statistics.left_idx : pulse_statistics.right_idx],
            pulse_statistics.mean_pulse_each_timeloc[i]
            - pulse_statistics.std_pulse_each_timeloc[i],
            pulse_statistics.mean_pulse_each_timeloc[i]
            + pulse_statistics.std_pulse_each_timeloc[i],
            alpha=0.5,
        )
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Voltage [a.u.]")
    ax.axvline(time_ax[pulse_statistics.left_idx], color="black", linestyle="--")
    ax.axvline(time_ax[pulse_statistics.right_idx], color="black", linestyle="--")
    ax.axvline(time_ax[pulse_statistics.noise_idxs[0]], color="red", linestyle="--")
    ax.legend(title="Bias voltage")

    fig2, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(voltage_arr, pulse_statistics.mean_pulse, label="Mean pulse")
    ax2.plot(
        voltage_arr,
        pulse_statistics.mean_divided,
        label="Mean pulse/mean std",
        color=color[1],
    )
    ax2.grid(False)
    ax.set_xlabel("Bias voltage [V]")
    ax.set_ylabel("Peak voltage [a.u.]")
    ax2.set_ylabel(r"Mean pulse/$\sigma$", color=color[1])
    # ax.legend()
    fig3, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(ref_data[0] * 10**9, ref_data[1])
    ax.set_xlabel(r"Time [ns]")
    ax.set_ylabel(r"Voltage [a.u.]")
    fig4, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(ref_data_amped[0] * 10**9, -ref_data_amped[1])
    ax.set_xlabel(r"Time [ns]")
    ax.set_ylabel(r"Voltage [a.u.]")
    if save_figs:
        fig1.savefig(f"{fig_dir}/pulses_diff_biases.pdf", bbox_inches="tight")
        fig2.savefig(
            f"{fig_dir}/pulse-power-and-noise_vs_bias.pdf", bbox_inches="tight"
        )
        fig3.savefig(
            f"{fig_dir}/elec-pulse_dir-out-of-pulsegen.pdf", bbox_inches="tight"
        )
        fig4.savefig(f"{fig_dir}/elec-pulse_after-amp.pdf", bbox_inches="tight")
    # plt.show()
