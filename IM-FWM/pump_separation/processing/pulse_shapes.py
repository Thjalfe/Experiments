import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

plt.ion()
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
# |%%--%%| <kmVhiYHnNW|t2yhxHMzWr>
freqs = list(pump_data.keys())
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
# |%%--%%| <t2yhxHMzWr|O1JUwo5q2G>
freq_idx = 3
freq = freqs[freq_idx]
sub_data = pump_data[freq]
duty_cycles = np.array(sub_data["duty_cycle"])
time_ref = np.array(sub_data["time_ax_ref"])
voltage_ref = np.array(sub_data["voltage_ref"])
time = np.array(sub_data["time_ax"])
voltage = np.array(sub_data["voltage"])

for i, duty_cycle in enumerate(duty_cycles):
    fig, ax = plt.subplots()
    # ax1 = ax.twinx()
    # ax1.plot(time_ref[i], voltage_ref[i], label='Reference')
    ax.plot(time[i] * 10**6, voltage[i] * 10**3)
    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(f"Frequency={freq}, Duty Cycle: {duty_cycle}")
    ax.legend()

# |%%--%%| <O1JUwo5q2G|qVuX5LFjNy>
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
shift_idler_idx = pulse_starts_idler[0] - 550
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
ax.plot(pump_time * 10**6, pump_voltage_normalized, label="Pump")
ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel("Voltage (a.u.)")
ax.legend()
# |%%--%%| <a0eHedtdST|ztO2XPQNTL>
# Mean for idler
time_around_pulse_start = [-1 * 10**-6, 2 * 10**-6]


def get_pulse_windows(pulse_starts, time_ax, time_around_pulse_start):
    idx_shift_start = int(time_around_pulse_start[0] * sample_rate)
    idx_shift_end = int(time_around_pulse_start[1] * sample_rate)
    idx_shifts = np.array([idx_shift_start, idx_shift_end])
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
    pulse_starts_idler, idler_time_shifted, time_around_pulse_start
)
mean_pulse_idler = np.mean(voltage_pulses_only_idler, axis=0)
std_pulse_idler = np.std(voltage_pulses_only_idler, axis=0)

voltages_pulses_only_pump, time_ax_pump, mean_len_pump = get_pulse_windows(
    pulse_starts_pump, pump_time, time_around_pulse_start
)
mean_pulse_pump = np.mean(voltages_pulses_only_pump, axis=0)
std_pulse_pump = np.std(voltages_pulses_only_pump, axis=0)

voltages_pulses_only_sig, time_ax_sig, mean_len_sig = get_pulse_windows(
    pulse_starts_sig, sig_data[0], time_around_pulse_start
)
mean_pulse_sig = np.mean(voltages_pulses_only_sig, axis=0)
std_pulse_sig = np.std(voltages_pulses_only_sig, axis=0)

fig, ax = plt.subplots()
# ax.plot(time_ax_idler * 10**6, mean_pulse_idler, label="Idler")
ax.plot(time_ax_pump * 10**6, mean_pulse_pump, label="Pump")
# ax.plot(time_ax_sig * 10**6, mean_pulse_sig, label="Signal")
# |%%--%%| <ztO2XPQNTL|tGLw1YBky4>

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
