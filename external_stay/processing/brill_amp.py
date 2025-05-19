import glob
import matplotlib.pyplot as plt
import numpy as np
from external_stay.run_exp_funcs.brill_amp import TracesSpectraMultipleColls
from external_stay.processing.processing_funcs.brill_amp import (
    load_dataset_multiple_powers,
    process_single_dataset_entry,
    SpectraProcessed,
)

plt.style.use("custom")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


fig_dir = "/home/thjalfe/Documents/PhD/logbook/2025/april/figs/week1"
save_figs = True
data_loc = "../data/brillioun_amp/Fiber=HNDS1615AAA-6-3-2_257meter/weird_pulse_things/rf=9.36GHz_traces-and-spectra_full-seq-len=2km_moving_pump_const-pol.pkl"
data = TracesSpectraMultipleColls.load(data_loc)
data.time_ax = data.pump_ref_time_traces[0, 0, :]
data.wl_ax = data.pump_ref_spectrum[0, :]
data.mean_std_traces()
data.calc_gain()
fig, ax = plt.subplots(2, 1)
for i in range(data.num_segment_steps):
    ax[0].plot(data.time_ax * 10**9 - 10 * i - 13, data.mean_pump_norm[i, :])
    ax[1].plot(data.time_ax * 10**9 - 10 * i, data.mean_signal_norm[i, :])
ax[0].set_ylabel("Voltage [a.u.]")
ax[1].set_xlabel("Time [ns]")
ax[1].set_ylabel("Voltage [a.u.]")
ax[0].set_xlim(-75, 100)
ax[1].set_xlim(-75, 100)
if save_figs:
    fig.savefig(
        f"{fig_dir}/sig_pump_traces_moving-in-from-no-collision_single-overlap.pdf",
        bbox_inches="tight",
    )
fig, ax = plt.subplots()
for i in range(data.num_segment_steps):
    ax.plot(
        -10 * i,
        data.gain_vs_segment_step[i],
        marker="x",
        color=colors[i],
        markersize=20,
        markeredgewidth=5,
    )
ax.set_xlabel("Pump pulse delay relative to start [ns]")
ax.set_ylabel("Brillouin gain from OSA [dB]")
if save_figs:
    fig.savefig(
        f"{fig_dir}/brill-gain_moving-in-from-no-collision_single-overlap.pdf",
        bbox_inches="tight",
    )
plt.show()
# |%%--%%| <HYAIeZ06bL|Csm08JSsvf>
data_dir = "../data/brillioun_amp/Fiber=HNDS1615AAA-6-3-2_257meter/pulses_overlap_in_center_probably-double-collisions/power_sweep/"
pulse_duration = 10e-9

fig_dir = "/home/thjalfe/Documents/PhD/logbook/2025/april/figs/week1"
save_figs = True
dataset, powers = load_dataset_multiple_powers(data_dir)
first_entry = dataset[0]
time_axis = first_entry["pump_ref"]["time_traces"][0, 0, :]
processed_time_data = []
processed_spec_data = []
gain = []
amplified_power = []
sig_power_idx = 0
for data in dataset:
    processed_time_data_tmp = process_single_dataset_entry(
        data, pulse_duration=10e-9, time_axis=time_axis
    )
    pump_ref = data["pump_ref"]["spectrum"]
    wl_ax = pump_ref[0]
    sig_ref = data["signal_ref"]["spectrum"]
    amplified_spec = data["signal_w_amplification"]["spectrum"]
    spectra_processed = SpectraProcessed(
        wl_ax=wl_ax,
        signal_ref=sig_ref,
        pump_ref=pump_ref,
        amplified_spectrum=amplified_spec,
    )
    processed_time_data.append(processed_time_data_tmp)
    processed_spec_data.append(spectra_processed)
    gain.append(spectra_processed.gain)
    amplified_power.append(spectra_processed.amplified_power)
    carrier_wl = wl_ax[np.argmax(pump_ref[1])]
    fig, ax = plt.subplots()
    ax.plot(wl_ax, sig_ref[1])
    ax.plot(wl_ax, amplified_spec[1])
    ax.set_xlim(1549.5, 1550.5)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Power [dBm]")
    ax.set_xticks(np.arange(1549.5, 1550.51, 0.25))
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_title(f"Signal power in: {powers[sig_power_idx]} dBm")
    # ax.set_xticks(1550, 155
    if save_figs:
        fig.savefig(
            f"{fig_dir}/spectra_Pin={powers[sig_power_idx]}_double_overlap.pdf",
            bbox_inches="tight",
        )
    sig_power_idx += 1
plt.show()
# |%%--%%| <Csm08JSsvf|LCzkXxJFUH>
#
fig, ax = plt.subplots()
# ax2 = ax.twinx()
ax.plot(powers, gain, "-o")
# ax2.plot(powers, amplified_power, "-o", color=colors[1])
# ax2.grid(False)
ax.set_xlabel(r"Power meter reading (P$_\mathrm{s}$) [dBm]")
ax.set_ylabel("Brillouin gain [dB]")
# ax2.set_ylabel("Output power [dBm]")
if save_figs:
    fig.savefig(f"{fig_dir}/brill_gain_double_overlap.pdf", bbox_inches="tight")
plt.show()

# |%%--%%| <LCzkXxJFUH|gl61CRE41t>
sig_power_idx = 0
for sig_power_idx in range(len(processed_time_data)):
    timedata_tmp = processed_time_data[sig_power_idx]
    processed_timedata = timedata_tmp.processed_traces
    legend = ["Pump ref", "Signal ref", "Pump with signal on", "Signal with pump on"]
    fig, ax = plt.subplots()
    for idx, processed_subdata_key in enumerate(processed_timedata):
        processed_traces = processed_timedata[processed_subdata_key]
        time_ax_ns = timedata_tmp.time_axis * 10**9
        ax.plot(time_ax_ns, processed_traces.mean_normalized, label=legend[idx])
        ax.fill_between(
            time_ax_ns,
            (processed_traces.mean_normalized - processed_traces.std_normalized),
            (processed_traces.mean_normalized + processed_traces.std_normalized),
            alpha=0.3,
        )
        ax.axvline(
            processed_time_data[sig_power_idx].idler_loc_time[0] * 10**9,
            linestyle="--",
            color="gray",
            alpha=0.5,
        )
        ax.axvline(
            processed_time_data[sig_power_idx].idler_loc_time[1] * 10**9,
            linestyle="--",
            color="gray",
        )
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Voltage [a.u]")
    ax.legend()
    ax.set_title(f"Signal power in: {powers[sig_power_idx]} dBm")
    if save_figs:
        fig.savefig(
            f"{fig_dir}/traces_Pin={powers[sig_power_idx]}_double_overlap.pdf",
            bbox_inches="tight",
        )
plt.show()
