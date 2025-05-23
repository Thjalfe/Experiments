import glob
import os
import pickle
import re
import sys
import matplotlib.pyplot as plt
import numpy as np

current_file_path = os.path.abspath(".py")
current_file_dir = os.path.dirname(current_file_path)

project_root_dir = os.path.join(current_file_dir, "..", "..")
sys.path.append(os.path.normpath(project_root_dir))
import scipy.stats as stats
from pump_separation.funcs.utils import (
    calculate_differences,
    return_filtered_peaks,
)  # noqa: E402, E501

plt.style.use("custom")
cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.ion()
# plt.rcParams["figure.figsize"] = (20, 11)


def extract_numbers_from_filename(f):
    numbers = re.findall(r"\d+\.\d+|\d+", f)
    if numbers:
        return [float(num) for num in numbers]
    else:
        return []


def get_first_number_from_filename(f):
    numbers = extract_numbers_from_filename(f)
    if numbers:
        return numbers[0]
    else:
        return float("inf")


def load_spectra_data(file_dir):
    file_names = glob.glob(file_dir + "*spectra*.pkl")
    file_names.sort(key=get_first_number_from_filename, reverse=True)
    pump_vals = np.array([extract_numbers_from_filename(f) for f in file_names])
    pump_seps = pump_vals[:, 1] - pump_vals[:, 0]
    data = []
    for i, file_name in enumerate(file_names):
        with open(file_name, "rb") as f:
            data.append(pickle.load(f))
    return data, pump_vals, pump_seps


def load_oscilloscope_data(file_dir):
    osci_file_names = glob.glob(file_dir + "*oscilloscope*.pkl")
    osci_file_names.sort(key=get_first_number_from_filename, reverse=True)
    osci_data = []
    for i, file_name in enumerate(osci_file_names):
        with open(file_name, "rb") as f:
            osci_data.append(pickle.load(f))
    return osci_data


def calculate_ce(
    data,
    pump_vals,
    duty_cycles_available,
    peak_threshold=0.1,
    peak_min=-52,
    peak_range=1,
    peak_max=-20,
    duty_cycles_wanted=[0.1, 0.2, 0.5, 1],
):
    num_meas_for_each_config = len(data[0]["wavelengths"][0, :, 0])
    duty_cycle_idx_map = [duty_cycles_available.index(d) for d in duty_cycles_wanted]
    ce_all = np.zeros((len(data), len(duty_cycles_wanted), num_meas_for_each_config))
    for i, d in enumerate(data):
        for j in range(len(duty_cycles_wanted)):
            duty_cycle_idx = duty_cycle_idx_map[j]
            for k in range(num_meas_for_each_config):
                cur_data = np.vstack(
                    (
                        np.squeeze(d["wavelengths"])[duty_cycle_idx, k, :],
                        np.squeeze(d["powers"])[duty_cycle_idx, k, :],
                    )
                ).T
                x = return_filtered_peaks(
                    cur_data,
                    peak_threshold,
                    peak_min,
                    tuple(pump_vals[i]),
                    peak_range,
                    peak_max,
                )
                diff = calculate_differences(x, cur_data)[-1]
                ce_all[i, j, k] = diff
    ce_mean = np.mean(ce_all, axis=2)
    ce_std = np.std(ce_all, axis=2)
    return ce_mean, ce_std


def plot_ce_vs_duty_cycle(
    ce_mean,
    duty_cycles,
    pump_seps,
    fig_path,
    save_figs=False,
    extra_title="",
    extra_fig_name="",
):
    fig, ax = plt.subplots()
    for i in range(len(ce_mean)):
        ax.plot(
            np.array(duty_cycles) * 100,
            -ce_mean[i, :],
            "-o",
            label=f"{pump_seps[i]:.2f} nm",
        )
    ax.set_xlabel(r"Duty cycle (\%)")
    ax.set_ylabel("CE (dB)")
    # ax.set_title(f"CE vs duty cycle, {extra_title}")
    leg = ax.legend(title="Pump separation", loc="upper right")
    if save_figs:
        fig.savefig(
            fig_path + f"ce_vs_duty_cycle_all_pump_seps_{extra_fig_name}.pdf",
            bbox_inches="tight",
        )


def plot_ce_vs_pump_sep(
    ce_mean,
    pump_seps,
    duty_cycles,
    num_pulses,
    fig_path,
    save_figs=False,
    extra_title="",
    extra_fig_name="",
):
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(11, 6))
    for i in range(num_pulses):
        if duty_cycles[i] == 1:
            label = "CW"
        else:
            label = rf"{duty_cycles[i] * 100} \%"
        ax.plot(
            pump_seps,
            -ce_mean[:, i] - 10 * np.log10(duty_cycles[i]),
            "-o",
            label=label,
        )
    ax.set_xlabel(r"$\lambda_p-\lambda_q$ (nm)")
    ax.set_ylabel(r"$\eta$ (dB)")
    # ax.set_title(f"CE vs pump sep, with duty cycle offset, {extra_title}")
    leg = ax.legend(title="Duty cycle", loc="upper right")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    fig.tight_layout()
    if save_figs:
        fig.savefig(
            fig_path
            + f"ce_vs_pump_sep_all_duty_cycles_dutycycle_db_offset_{extra_fig_name}.pdf",
            bbox_inches="tight",
        )


def plot_raw_data(
    data, duty_cycles, fig_path, save_figs=False, extra_title="", extra_fig_name=""
):
    pulse_num = [0, -1]
    fig, ax = plt.subplots()
    for i in pulse_num:
        wl_ax = np.squeeze(data[0]["wavelengths"])[i, 1, :]
        p_ax = np.squeeze(data[0]["powers"])[i, 1, :]
        ax.plot(wl_ax, p_ax, label=rf"{duty_cycles[i] * 100} \%")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Power (dBm)")
    # ax.set_title(rf"2 nm sep, {extra_title}")
    leg = ax.legend(title="Duty cycle", loc="upper right")
    if save_figs:
        fig.savefig(
            fig_path + f"2nm_dutycycle_comp_{extra_fig_name}.pdf", bbox_inches="tight"
        )


file_dir_max = (
    "../data/large_pump_sep_within_L_band/pol_check/before_isolator/maximized/"
)
file_dir_min = (
    "../data/large_pump_sep_within_L_band/pol_check/before_isolator/minimized/"
)
file_dir_max = "../data/large_pump_sep_within_L_band/pol_check/maximized/"
file_dir_min = "../data/large_pump_sep_within_L_band/pol_check//minimized/"
file_dir_osc = "../data/large_pump_sep_within_L_band/"
fig_path = "../figs/pol_diff_within_L/"
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

save_figs = False

# Load and process data from first directory
spectra_data_max, pump_vals_max, pump_seps_max = load_spectra_data(file_dir_max)
osci_data_max = load_oscilloscope_data(file_dir_max)
duty_cycles_wanted = [0.1, 0.2, 0.5, 1]
ce_mean_max, ce_std_max = calculate_ce(
    spectra_data_max,
    pump_vals_max,
    spectra_data_max[0]["duty_cycle"],
    duty_cycles_wanted=duty_cycles_wanted,
)
duty_cycles_max = duty_cycles_wanted
num_pulses_max = len(duty_cycles_max)
# |%%--%%| <qVI1KpRhc4|vH38TpxZ2c>
c = 2.998e8
pump_seps_hz = np.zeros(len(pump_vals_max))
for i, wl_pair in enumerate(pump_vals_max):
    pump_seps_hz[i] = c * np.abs(1 / wl_pair[0] - 1 / wl_pair[1]) * 1e9
ce_adj = -ce_mean_max - 10 * np.log10(duty_cycles_max)
ce_adj = ce_adj.T
data = {
    "CE [dB]": ce_adj,
    # "Spectral separation [Hz]": pump_seps_hz,
    "pump_wls": pump_vals_max,
    "Duty cycles": duty_cycles_max,
}
# |%%--%%| <vH38TpxZ2c|gki1lrRu3L>


# Plot data from first directory
plot_ce_vs_duty_cycle(
    ce_mean_max,
    duty_cycles_max,
    pump_seps_max,
    fig_path,
    save_figs,
    extra_title="pol max",
    extra_fig_name="pol_max",
)
plot_ce_vs_pump_sep(
    ce_mean_max,
    pump_seps_max,
    duty_cycles_max,
    num_pulses_max,
    fig_path,
    False,
    extra_title="pol max",
    extra_fig_name="pol_max",
)
plot_raw_data(
    spectra_data_max,
    duty_cycles_max,
    fig_path,
    save_figs,
    extra_title="pol max",
    extra_fig_name="pol_max",
)
# |%%--%%| <gki1lrRu3L|Uiy4rJefy5>

# Load and process data from second directory
spectra_data_min, pump_vals_min, pump_seps_min = load_spectra_data(file_dir_min)
osci_data_min = load_oscilloscope_data(file_dir_min)
ce_mean_min, ce_std_min = calculate_ce(spectra_data_min, pump_vals_min)
duty_cycles_min = spectra_data_min[0]["duty_cycle"]
num_pulses_min = len(spectra_data_min[0]["wavelengths"][:, 0, 0])

# Plot data from second directory
plot_ce_vs_duty_cycle(
    ce_mean_min,
    duty_cycles_min,
    pump_seps_min,
    fig_path,
    save_figs,
    extra_title="pol min",
    extra_fig_name="pol_min",
)
plot_ce_vs_pump_sep(
    ce_mean_min,
    pump_seps_min,
    duty_cycles_min,
    num_pulses_min,
    fig_path,
    save_figs,
    extra_title="pol min",
    extra_fig_name="pol_min",
)
plot_raw_data(
    spectra_data_min,
    duty_cycles_min,
    fig_path,
    save_figs,
    extra_title="pol min",
    extra_fig_name="pol_min",
)

# Get oscilloscope data
osci_data = load_oscilloscope_data(file_dir_osc)
# |%%--%%| <Uiy4rJefy5|AshiPwhVfw>
fig, ax = plt.subplots()
for i, duty_cycle in enumerate(duty_cycles_max):
    ce_diff = ce_mean_min[:, i] - ce_mean_max[:, i]
    # ce_diff[ce_diff < 0] = np.nan
    ax.plot(pump_seps_max, ce_diff, "-o", label=rf"{duty_cycle * 100} \%")
ax.set_xlabel(r"Pump separation (nm)")
ax.set_ylabel("CE difference (dB)")
ax.set_title("CE difference vs pump sep")
leg = ax.legend(title="Duty cycle", loc="upper right")
if save_figs:
    fig.savefig(fig_path + "ce_diff_vs_pump_sep.pdf", bbox_inches="tight")
# |%%--%%| <AshiPwhVfw|T5Q2zxtJ4A>

max_osci_vals = np.zeros((len(osci_data), len(duty_cycles_max)))
for i, sub_osci in enumerate(osci_data):
    for j in range(6):
        max_osci_vals[i, j] = np.max(sub_osci["voltage_sig"][j])
max_osci_vals = max_osci_vals.T
voltage_duty_cycle = np.max(max_osci_vals, axis=1)
voltage_duty_cycle = voltage_duty_cycle / np.min(voltage_duty_cycle)
idx_pump_sep = 5
ce_mean_w_offset = -ce_mean_max[idx_pump_sep, :] - 10 * np.log10(duty_cycles_max)
# linear fit to get slope
slope, intercept, r_value, p_value, std_err = stats.linregress(
    voltage_duty_cycle, ce_mean_w_offset
)
fig, ax = plt.subplots()
ax.plot(voltage_duty_cycle, ce_mean_w_offset, "-o")
ax.plot(
    voltage_duty_cycle,
    slope * voltage_duty_cycle + intercept,
    color="black",
    linestyle="--",
    label=rf"$y = {slope:.2f}x + {intercept:.2f}$",
)
ax.set_xlabel("Voltage relative to CW (a.u.)")
ax.set_ylabel("CE (dB)")
ax.set_title(f"CE vs voltage, pump sep = {pump_seps_max[idx_pump_sep]} nm")
ax.legend()

# |%%--%%| <T5Q2zxtJ4A|y7zG55BCh3>
# oscilloscope data
idx_osci = 1
duty_cycles = duty_cycles_max
# idx_duty_cycle = 0
for idx_duty_cycle in range(len(duty_cycles)):
    ref_data = np.vstack(
        (
            osci_data[idx_osci]["time_ref"][idx_duty_cycle],
            osci_data[idx_osci]["voltage_ref"][idx_duty_cycle],
        )
    ).T
    sig_data = np.vstack(
        (
            osci_data[idx_osci]["time_sig"][idx_duty_cycle],
            osci_data[idx_osci]["voltage_sig"][idx_duty_cycle],
        )
    ).T
    fig, ax = plt.subplots()
    ax.plot(ref_data[:, 0], ref_data[:, 1])
    ax1 = ax.twinx()
    ax1.plot(sig_data[:, 0], sig_data[:, 1], color="C1")
    ax1.grid()
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Voltage (V)")
    # ax.set_title(
    #     rf"Duty cycle = {duty_cycles[idx_duty_cycle] * 100:.2f} \%, Pulse freq = {pulse_freqs / 1e3:.2f} kHz"
    # )
    # if save_figs:
    #     fig.savefig(
    #         fig_path
    #         + f"osci_data_duty_cycle_{duty_cycles[idx_duty_cycle] * 100:.2f}.pdf",
    #         bbox_inches="tight",
    #     )
