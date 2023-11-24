import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

plt.ioff()
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.style.use("custom")
plt.rcParams["figure.figsize"] = (20, 11)

with open("../data/scope_traces/freq_cycle_vary_pumps.pkl", "rb") as f:
    pump_data = pickle.load(f)

with open("../data/scope_traces/sig_idler_traces/pump_sep_15_sig.pkl", "rb") as f:
    sig_data = pickle.load(f)

with open("../data/scope_traces/sig_idler_traces/pump_sep_15_idler.pkl", "rb") as f:
    idler_data = pickle.load(f)
fig_path = "../figs/pulse_shapes/"
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
save_figs = False
freqs = list(pump_data.keys())
#|%%--%%| <rKfmAIqGIa|Ioj6lFLhNs>
freq_idx = 3
for freq_idx in range(len(freqs)):
    freq = freqs[freq_idx]
    sub_data = pump_data[freq]
    duty_cycles = np.array(sub_data["duty_cycle"])
    time_ref = np.array(sub_data["time_ax_ref"])
    voltage_ref = np.array(sub_data["voltage_ref"])
    time = np.array(sub_data["time_ax"])
    voltage = np.array(sub_data["voltage"])

    fig, ax = plt.subplots()
    for i, duty_cycle in enumerate(duty_cycles):
        # ax1 = ax.twinx()
        # ax1.plot(time_ref[i], voltage_ref[i], label='Reference')
        ax.plot(
            time[i] * 10**6, voltage[i] * 10**3, label=f"{duty_cycle*100:.0f}\%"
        )
    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(f"Frequency={freq}")
    ax.legend()
    if save_figs:
        fig.savefig(
            f"{fig_path}rep_rate_{freq}_diff_duty_cycles.pdf", bbox_inches="tight"
        )

# |%%--%%| <Ioj6lFLhNs|qVuX5LFjNy>
fig, ax = plt.subplots()
ax.plot(sig_data[0] * 10**6, sig_data[1] * 10**3, label="Signal")
ax.plot(idler_data[0] * 10**6, idler_data[1] * 10**3, label="Idler")
ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel("Voltage (mV)")
# |%%--%%| <qVuX5LFjNy|a0eHedtdST>
# Find regions where voltage surpasses the threshold


def detect_pulses(voltage_values, threshold, min_distance, pulse_slope="pos"):
    if pulse_slope == "neg":
        voltage_values = -voltage_values
        threshold = -threshold
    pulse_starts = []
    pulse_ends = []

    in_pulse = False
    for i in range(1, len(voltage_values)):
        if voltage_values[i] > threshold and voltage_values[i - 1] <= threshold:
            if not in_pulse and (
                not pulse_ends or (i - pulse_ends[-1] >= min_distance)
            ):
                pulse_starts.append(i)
                in_pulse = True
        elif np.mean(voltage_values[i : i + 2000]) <= threshold and in_pulse:
            pulse_ends.append(i)
            in_pulse = False
    if len(pulse_starts) > len(pulse_ends):
        pulse_ends.append(len(voltage_values) - 1)

    return pulse_starts, pulse_ends


def rolling_average(data, window_size):
    """Compute the rolling average of data using a given window size."""
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd to have a symmetric window

    # Compute the number of points on either side of the center point
    half_window = window_size // 2

    smoothed_data = np.copy(data)
    for i in range(half_window, len(data) - half_window):
        smoothed_data[i] = np.mean(data[i - half_window : i + half_window + 1])

    return smoothed_data


threshold_idler = 0.003
time_values = idler_data[0]
voltage_values = idler_data[1]
sample_rate = 1 / (time_values[1] - time_values[0])
min_distance_points = int(1 * 10**-6 * sample_rate)
pulse_length = 1 * 10**-6


pulse_starts_idler, pulse_ends_idler = detect_pulses(
    voltage_values, threshold_idler, min_distance_points, pulse_slope="pos"
)
pulse_starts_pump, pulse_ends_pump = detect_pulses(
    pump_data["100KHz"]["voltage"][0],
    0.5,
    min_distance_points,
    pulse_slope="pos",
)
pulse_starts_sig, pulse_ends_sig = detect_pulses(
    sig_data[1], 0.019, min_distance_points, pulse_slope="neg"
)
shift_idler_idx = pulse_starts_idler[0] - 480
pump_time = pump_data["100KHz"]["time_ax"][0]
pump_voltage = pump_data["100KHz"]["voltage"][0]
pump_voltage_normalized = pump_voltage / np.max(pump_voltage)
idler_time_shifted = (
    time_values[shift_idler_idx:] - time_values[shift_idler_idx]
) * 10**6
voltage_values_normalized = voltage_values / np.max(voltage_values)
rolling_avg = rolling_average(voltage_values, 5)
rolling_avg = rolling_avg / np.max(rolling_avg)
fig, ax = plt.subplots()
ax.plot(idler_time_shifted, voltage_values_normalized[shift_idler_idx:], label="Idler")
# ax.plot(idler_time_shifted, rolling_avg[shift_idler_idx:], label="Rolling Average")
ax.plot(pump_time * 10**6, pump_voltage_normalized**2, label="Pump")
ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel("Voltage (a.u.)")
ax.legend()
# |%%--%%| <a0eHedtdST|Fk7XujLiPM>
# Mean for idler
time_around_pulse_start = [-1 * 10**-6, 2 * 10**-6]


def get_pulse_windows(
    pulse_starts,
    time_ax,
    time_around_pulse_start,
    voltage_values,
    start_idx=0,
    end_idx=-1,
):
    time_ax = time_ax[start_idx:end_idx]
    voltage_values = voltage_values[start_idx:end_idx]
    sample_rate = 1 / (time_ax[1] - time_ax[0])
    idx_shift_start = int(time_around_pulse_start[0] * sample_rate)
    idx_shift_end = int(time_around_pulse_start[1] * sample_rate)
    idx_shifts = np.array([idx_shift_start, idx_shift_end])
    if pulse_starts[0] < start_idx:
        pulse_starts = pulse_starts[1:]
    if pulse_starts[-1] + idx_shift_end > end_idx:
        pulse_starts = pulse_starts[:-1]
    pulse_starts = [p - start_idx for p in pulse_starts]
    pulse_starts = np.array(pulse_starts)
    pulse_windows = pulse_starts[:, np.newaxis] + idx_shifts
    time_ax = np.linspace(
        time_around_pulse_start[0],
        time_around_pulse_start[1],
        idx_shift_end - idx_shift_start,
    )
    voltage_pulses_only = []
    for i, pulse_window in enumerate(pulse_windows):
        if pulse_window[0] < 0:
            pulse_window[0] = 0
        voltage_tmp = voltage_values[pulse_window[0] : pulse_window[1]]
        voltage_pulses_only.append(voltage_tmp)
    mean_len = np.min([len(v) for v in voltage_pulses_only])
    voltage_pulses_only = np.array([v[:mean_len] for v in voltage_pulses_only])
    return voltage_pulses_only, time_ax, mean_len


voltage_pulses_only_idler, time_ax_idler, mean_len_idler = get_pulse_windows(
    pulse_starts_idler, time_values, time_around_pulse_start, voltage_values
)
mean_pulse_idler = np.mean(voltage_pulses_only_idler, axis=0)
std_pulse_idler = np.std(voltage_pulses_only_idler, axis=0)

voltages_pulses_only_pump, time_ax_pump, mean_len_pump = get_pulse_windows(
    pulse_starts_pump, pump_time, time_around_pulse_start, pump_voltage
)
mean_pulse_pump = np.mean(voltages_pulses_only_pump, axis=0)
std_pulse_pump = np.std(voltages_pulses_only_pump, axis=0)

voltages_pulses_only_sig, time_ax_sig, mean_len_sig = get_pulse_windows(
    pulse_starts_sig,
    sig_data[0],
    time_around_pulse_start,
    sig_data[1],
    start_idx=5000,
    end_idx=126000,
)
mean_pulse_sig = np.mean(voltages_pulses_only_sig, axis=0)
std_pulse_sig = np.std(voltages_pulses_only_sig, axis=0)
mean_pulse_idler = mean_pulse_idler * 1000
std_pulse_idler = std_pulse_idler * 1000
mean_pulse_sig = mean_pulse_sig * 1000
std_pulse_sig = std_pulse_sig * 1000
fig, ax = plt.subplots()
ax.plot(time_ax_idler * 10**6, mean_pulse_idler, label="Idler")
ax.fill_between(
    time_ax_idler * 10**6,
    mean_pulse_idler - std_pulse_idler,
    mean_pulse_idler + std_pulse_idler,
    alpha=0.5,
)
# ax.plot(time_ax_pump * 10**6, mean_pulse_pump, label="Pump")
ax.plot(time_ax_sig * 10**6, mean_pulse_sig, label="Signal")
ax.fill_between(
    time_ax_sig * 10**6,
    mean_pulse_sig - std_pulse_sig,
    mean_pulse_sig + std_pulse_sig,
    alpha=0.5,
)
ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel("Voltage (mV)")
ax.legend()
if save_figs:
    fig.savefig(f"{fig_path}mean_sig_idler_pulses.pdf", bbox_inches="tight")
#|%%--%%| <Fk7XujLiPM|gq11egzJkJ>
mean_pulse_idler_normalized = mean_pulse_idler / np.max(mean_pulse_idler)
std_pulse_idler_normalized = std_pulse_idler / np.max(mean_pulse_idler)
mean_pulse_idler_normalized_rolling = rolling_average(mean_pulse_idler_normalized, 3)
std_pulse_idler_normalized_rolling = rolling_average(std_pulse_idler_normalized, 3)
pump_voltage_normalized_idxs = pump_voltage_normalized[pulse_starts_pump[1] - 5000: pulse_starts_pump[1] + 10000]
pump_voltage_normalized_idxs_rolling = rolling_average(pump_voltage_normalized_idxs, 3)
plt.rcParams["figure.figsize"] = (16, 14)
plt.ioff()
fig, ax = plt.subplots()
ax.plot(time_ax_idler * 10**6, mean_pulse_idler_normalized, label="Idler")
# ax.plot(time_ax_idler * 10**6, mean_pulse_idler_normalized_rolling, label="Idler")
ax.fill_between(
    time_ax_idler * 10**6,
    mean_pulse_idler_normalized - std_pulse_idler_normalized,
    mean_pulse_idler_normalized + std_pulse_idler_normalized,
    alpha=0.5,
)
ax.plot(time_ax_pump * 10**6, pump_voltage_normalized_idxs**2, label=r"Pump$^2$")
# ax.plot(time_ax_pump * 10**6, pump_voltage_normalized_idxs_rolling**2, label=r"Pump$^2$ Rolling")
ax.set_xlim(-0.1, 1.1)
ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel("Voltage (a.u.)")
ax.legend()
if save_figs:
    fig.savefig(f"{fig_path}mean_pump_idler_pulses_normalized_w_std.pdf", bbox_inches="tight")
    fig.savefig("../../../../papers/cleo_us_2023/figs/mean_pump_idler_pulses_normalized_w_std.pdf", bbox_inches="tight")
freq_idx = 3
freq = freqs[freq_idx]
sub_data = pump_data[freq]
duty_cycles = np.array(sub_data["duty_cycle"])
time_ref = np.array(sub_data["time_ax_ref"])
voltage_ref = np.array(sub_data["voltage_ref"])
time = np.array(sub_data["time_ax"])
voltage = np.array(sub_data["voltage"])
new_voltage = voltage[0]
new_voltage[pulse_starts_pump[0]: pulse_ends_pump[0]] = voltage[0][pulse_starts_pump[1]: pulse_ends_pump[1] + 1]
norm_factor = np.max(new_voltage)
fig, ax = plt.subplots()
for i, duty_cycle in enumerate(duty_cycles):
    if duty_cycle == 0.75 or duty_cycle == 0.25:
        continue
    if i == 0:
        ax.plot(time[i] * 10**6, new_voltage / norm_factor, label=f"{duty_cycle}")
    else:
        ax.plot(time[i] * 10**6, voltage[i] / norm_factor, label=f"{duty_cycle}")
    ax.set_xlim(0, 5.1)
    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel("Voltage (a.u.)")
    # ax.set_title(f"Frequency={freq}, Duty Cycle: {duty_cycle}")
    ax.legend(title="Duty Cycle")
if save_figs:
    fig.savefig(
        f"{fig_path}rep_rate_{freq}_diff_duty_cycles.pdf", bbox_inches="tight"
    )
    fig.savefig("../../../../papers/cleo_us_2023/figs/rep_rate_100KHz_diff_duty_cycles.pdf", bbox_inches="tight")
#|%%--%%| <gq11egzJkJ|le08kchA0b>
#### 2x1 subplot of the plots above
mean_pulse_idler_normalized = mean_pulse_idler / np.max(mean_pulse_idler)
std_pulse_idler_normalized = std_pulse_idler / np.max(mean_pulse_idler)
mean_pulse_idler_normalized_rolling = rolling_average(mean_pulse_idler_normalized, 3)
std_pulse_idler_normalized_rolling = rolling_average(std_pulse_idler_normalized, 3)
pump_voltage_normalized_idxs = pump_voltage_normalized[pulse_starts_pump[1] - 5000: pulse_starts_pump[1] + 10000]
pump_voltage_normalized_idxs_rolling = rolling_average(pump_voltage_normalized_idxs, 3)
plt.rcParams["figure.figsize"] = (16, 6)
plt.rcParams["legend.fontsize"] = 30
plt.rcParams["axes.labelsize"] = 45
plt.rcParams["xtick.labelsize"] = 35
plt.rcParams["ytick.labelsize"] = 35
plt.rcParams["legend.title_fontsize"] = 35

plt.ioff()
fig, ax = plt.subplots(1, 2, sharey=True)
ax[1].plot(time_ax_idler * 10**6, mean_pulse_idler_normalized, label=r"Norm.(P$_{\mathrm{idler}}$)")
# ax[0].plot(time_ax_idler * 10**6, mean_pulse_idler_normalized_rolling, label="Idler")
# ax[1].fill_between(
#     time_ax_idler * 10**6,
#     mean_pulse_idler_normalized - std_pulse_idler_normalized,
#     mean_pulse_idler_normalized + std_pulse_idler_normalized,
#     alpha=0.5,
# )
ax[1].plot(time_ax_pump * 10**6, pump_voltage_normalized_idxs**2, label=r"Norm.(P$_{\mathrm{pump}})^2$")
# ax[0].plot(time_ax_pump * 10**6, pump_voltage_normalized_idxs_rolling**2, label=r"Pump$^2$ Rolling")
ax[1].set_xlim(-0.1, 1.1)
ax[1].set_xlabel(r"Time ($\mu$s)")
ax[0].set_ylabel("Norm. power", labelpad=20)
ax[1].legend()
ax[1].set_ylim(-0.1, 1.1)
freq_idx = 3
freq = freqs[freq_idx]
sub_data = pump_data[freq]
duty_cycles = np.array(sub_data["duty_cycle"])
time_ref = np.array(sub_data["time_ax_ref"])
voltage_ref = np.array(sub_data["voltage_ref"])
time = np.array(sub_data["time_ax"])
voltage = np.array(sub_data["voltage"])
new_voltage = voltage[0]
new_voltage[pulse_starts_pump[0]: pulse_ends_pump[0]] = voltage[0][pulse_starts_pump[1]: pulse_ends_pump[1] + 1]
norm_factor = np.max(new_voltage)
for i, duty_cycle in enumerate(duty_cycles):
    if duty_cycle == 0.75 or duty_cycle == 0.25:
        continue
    if i == 0:
        ax[0].plot(time[i] * 10**6, new_voltage / norm_factor, label=f"{duty_cycle}")
    else:
        ax[0].plot(time[i] * 10**6, voltage[i] / norm_factor, label=f"{duty_cycle}")
    ax[0].set_xlim(0, 5.1)
    ax[0].set_xlabel(r"Time ($\mu$s)")
    # ax[1].set_title(f"Frequency={freq}, Duty Cycle: {duty_cycle}")
    ax[0].set_xticks([0, 1, 2, 3, 4, 5])
    ax[0].legend(title="Duty Cycle")
ax[0].spines["right"].set_visible(True)
ax[1].spines["right"].set_visible(True)
ax[0].spines["top"].set_visible(True)
ax[1].spines["top"].set_visible(True)
fig.tight_layout()
if save_figs:
    fig.savefig(
        f"{fig_path}trace_subplot", bbox_inches="tight"
    )
    fig.savefig("../../../../papers/cleo_us_2023/figs/trace_subplot.pdf", bbox_inches="tight")
# |%%--%%| <le08kchA0b|7DvGxzPBgL>

fig, ax = plt.subplots()
for i, voltage_tmp in enumerate(voltage_pulses_only_idler):
    ax.plot(
        time_ax_idler[: len(voltage_tmp)] * 10**6,
        voltage_tmp,
        label=f"Pulse {i}",
    )
ax.plot(time_ax_idler[:mean_len_idler] * 10**6, mean_pulse_idler, label="Mean")
ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel("Voltage (a.u.)")
ax.legend()
# |%%--%%| <b72tWx3ixr|7z9jgFWlvJ>
idler_test_data_loc = (
    "../data/scope_traces/sig_idler_traces/idler_traces/100.0kHz_pump_sep15nm.pkl"
)
sig_test_data_loc = (
    "../data/scope_traces/sig_idler_traces/sig_traces/100.0kHz_pump_sep15nm.pkl"
)
with open(idler_test_data_loc, "rb") as f:
    idler_test_data = pickle.load(f)
with open(sig_test_data_loc, "rb") as f:
    sig_test_data = pickle.load(f)
time_arrays_idler = idler_test_data["time"]
voltage_arrays_idler = idler_test_data["voltage"]
duty_cycle_array_idler = idler_test_data["duty_cycle"]
time_arrays_sig = sig_test_data["time"]
voltage_arrays_sig = sig_test_data["voltage"]
duty_cycle_array_sig = sig_test_data["duty_cycle"]
fig, ax = plt.subplots()
for i, voltage_tmp in enumerate(voltage_arrays_idler):
    ax.plot(
        time_arrays_idler[i] * 10**6,
        voltage_tmp * 10**3,
        label=f"Duty Cycle: {duty_cycle_array_idler[i]}",
    )
    ax.plot(
        time_arrays_sig[i] * 10**6,
        voltage_arrays_sig[i] * 10**3,
        label=f"Duty Cycle: {duty_cycle_array_sig[i]}",
    )
ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel("Voltage (a.u.)")
ax.legend()
# |%%--%%| <7z9jgFWlvJ|KAYookS8Sn>
sub_time_idler = time_arrays_idler[1]
sub_voltage_idler = voltage_arrays_idler[1]
fig, ax = plt.subplots()
ax.plot(sub_time_idler * 10**6, sub_voltage_idler * 10**3)
ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel("Voltage (mV)")
sub_time_sig = time_arrays_sig[1]
sub_voltage_sig = voltage_arrays_sig[1]
fig, ax = plt.subplots()
ax.plot(sub_time_sig * 10**6, sub_voltage_sig * 10**3)
ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel("Voltage (mV)")
# |%%--%%| <KAYookS8Sn|lzcGFu9ign>
voltage_pulses_only_idler, time_ax_idler, mean_len_idler = get_pulse_windows(
    pulse_starts_idler, idler_time_shifted, time_around_pulse_start
)
mean_pulse_idler = np.mean(voltage_pulses_only_idler, axis=0)
std_pulse_idler = np.std(voltage_pulses_only_idler, axis=0)
