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
from pump_separation.funcs.utils import calculate_differences, return_filtered_peaks  # noqa: E402, E501

# plt.style.use("custom")
cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.ion()
plt.rcParams["figure.figsize"] = (20, 11)


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


file_dir = "../data/large_pump_sep_within_L_band/pol_check/maximized/"
file_dir = "../data/large_pump_sep_within_L_band/pol_check/minimized/"
# file_dir = "../data/large_pump_sep_within_L_band/"
file_names = glob.glob(file_dir + "*spectra*.pkl")

file_names.sort(key=get_first_number_from_filename, reverse=True)
pump_vals = np.array([extract_numbers_from_filename(f) for f in file_names])

pump_seps = pump_vals[:, 1] - pump_vals[:, 0]
data = []
for i, file_name in enumerate(file_names):
    with open(file_name, "rb") as f:
        data.append(pickle.load(f))
num_meas_for_each_config = len(data[0]["wavelengths"][0, :, 0])
num_pulses = len(data[0]["wavelengths"][:, 0, 0])
duty_cycles = data[0]["duty_cycle"]
pulse_freqs = data[0]["pulse_freq"]
osci_file_names = glob.glob(file_dir + "*oscilloscope*.pkl")
osci_file_names.sort(key=get_first_number_from_filename, reverse=True)
osci_data = []
for i, file_name in enumerate(osci_file_names):
    with open(file_name, "rb") as f:
        osci_data.append(pickle.load(f))
fig_path = "../figs/large_sep_twomode_fiber/"
save_figs = False
# |%%--%%| <GFvYHFoVjg|oAeBSarYRn>
# ce vs pulse section
ce_all = np.zeros((len(data), num_pulses, num_meas_for_each_config))
for i, d in enumerate(data):
    for j in range(num_pulses):
        for k in range(num_meas_for_each_config):
            cur_data = np.vstack((d["wavelengths"][j, k, :], d["powers"][j, k, :])).T
            x = return_filtered_peaks(cur_data, 0.1, -52, tuple(pump_vals[i]), 1, -20)
            diff = calculate_differences(x, cur_data)[-1]
            ce_all[i, j, k] = diff
ce_mean = np.mean(ce_all, axis=2)
ce_std = np.std(ce_all, axis=2)
fig, ax = plt.subplots()
for i in range(len(data)):
    ax.plot(
        np.array(duty_cycles) * 100,
        -ce_mean[i, :],
        "-o",
        label=f"{pump_seps[i]:.2f} nm",
    )
ax.set_xlabel(r"Duty cycle (\%)")
ax.set_ylabel("CE (dB)")
ax.set_title("CE vs duty cycle")
# legend title
leg = ax.legend(title="Pump separation", loc="upper right")
if save_figs:
    fig.savefig(fig_path + "ce_vs_duty_cycle_all_pump_seps.pdf", bbox_inches="tight")
# |%%--%%| <WkBWh64SUa|h152GFiqzY>
fig, ax = plt.subplots()
for i in range(num_pulses):
    ax.plot(
        pump_seps,
        -ce_mean[:, i] - 10 * np.log10(duty_cycles[i]),
        "-o",
        label=rf"{duty_cycles[i] * 100} \%",
    )
ax.set_xlabel(r"Pump separation (nm)")
ax.set_ylabel("CE (dB)")
ax.set_title("CE vs pump sep, with duty cycle offset")
# legend title
leg = ax.legend(title="Duty cycle", loc="upper right")
if save_figs:
    fig.savefig(fig_path + "ce_vs_pump_sep_all_duty_cycles_dutycycle_db_offset.pdf", bbox_inches="tight")
# |%%--%%| <u5c3cfqgGK|fSBNuH9Tn5>
# plot raw
pulse_num = [1, -1]
fig, ax = plt.subplots()
for i in pulse_num:
    wl_ax = data[0]["wavelengths"][i, 2, :]
    p_ax = data[0]["powers"][i, 2, :]
    ax.plot(wl_ax, p_ax, label=rf"{duty_cycles[i] * 100} \%")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Power (dBm)")
ax.set_title(r"2 nm sep")
# legend title
leg = ax.legend(title="Duty cycle", loc="upper right")
if save_figs:
    fig.savefig(fig_path + "2nm_dutycycle_comp.pdf", bbox_inches="tight")
# |%%--%%| <Kl6mSc6m0K|59LW50I86a>
# oscilloscope data
idx_osci = 2
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
    ax.set_title(
        rf"Duty cycle = {duty_cycles[idx_duty_cycle] * 100:.2f} \%, Pulse freq = {pulse_freqs / 1e3:.2f} kHz"
    )
    if save_figs:
        fig.savefig(
            fig_path
            + f"osci_data_duty_cycle_{duty_cycles[idx_duty_cycle] * 100:.2f}.pdf",
            bbox_inches="tight",
        )
# |%%--%%| <uwSuG4phat|6fsI2cDiBi>
# find and fit peak locations vs pump separation
meas_num = 0
wl_ax = []
p_ax = []
peak_loc = []
peak_loc_wl = []
for i, d in enumerate(data):
    wl_ax.append(d["wavelengths"][pulse_num, meas_num, :])
    p_ax.append(d["powers"][pulse_num, meas_num, :])
    peak_loc.append(p_ax[i].argmax())
    peak_loc_wl.append(wl_ax[i][peak_loc[i]])
y = peak_loc_wl
x = pump_seps
fit = np.polyfit(x, y, 1)
fit_fn = np.poly1d(fit)
fig, ax = plt.subplots()
ax.plot(x, y, "o", x, fit_fn(x), "--k")
ax.set_xlabel("Pump separation (nm)")
ax.set_ylabel("Peak location (nm)")
ax.set_title("Peak location vs pump separation")
