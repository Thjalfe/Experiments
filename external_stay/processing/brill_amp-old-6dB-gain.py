import numpy as np
import os
from matplotlib.ticker import MaxNLocator
import glob
import matplotlib.pyplot as plt
import pickle
from helper_funcs import dBm_to_mW, mW_to_dBm

plt.style.use("custom")

save_figs = True
fig_dir = "/home/thjalfe/Documents/PhD/logbook/2025/march/figs/week4"
ref_filename = "../data/brillouin_amp/spectra_const-bias-optimized-for-9.5GHz_rf-freq=9.0GHz-10.0GHz_stepsize=10.0MHz.pkl"
data_dir = "../data/brillouin_amp/rf_sweep-manual-bias/"
data_files = glob.glob(f"{data_dir}/*.pkl")
with open(f"{ref_filename}", "rb") as handle:
    ref = pickle.load(handle)
idx = 50
extra_sub_dir = "SSB_specs_const_bias"
if not os.path.exists(f"{fig_dir}/{extra_sub_dir}"):
    os.makedirs(f"{fig_dir}/{extra_sub_dir}")
rf_freqs = ref["rf_freqs"]
# for idx in range(len(rf_freqs)):
#     fig, ax = plt.subplots()
#     spectrum = ref["spectra"][idx]
#     spectrum[1, spectrum[1] < -80] = np.nan
#     ax.plot(spectrum[0], spectrum[1])
#     ax.set_xlabel("Wavelength [nm]")
#     ax.set_ylabel("Power [dBm]")
#     ax.axvline(1550.08, color="k")
#     ax.set_xlim(1549.5, 1550.5)
#     ax.set_xticks(np.arange(1549.5, 1550.51, 0.25))
#     ax.ticklabel_format(useOffset=False, style="plain")
#     ax.set_title(f"Clock freq: {rf_freqs[idx]*10**-9:.2f} GHz")
#     if save_figs:
#         fig.savefig(
#             f"{fig_dir}/{extra_sub_dir}/freq={rf_freqs[idx]*10**-9:.2f}.pdf",
#             bbox_inches="tight",
#         )
#
# plt.close("all")
# |%%--%%| <6IMBmWnDv0|KbtaV4IhSK>
save_figs = True
extra_sub_dir = "SSB_amp"
if not os.path.exists(f"{fig_dir}/{extra_sub_dir}"):
    os.makedirs(f"{fig_dir}/{extra_sub_dir}")
with open(f"{data_files[0]}", "rb") as handle:
    data = pickle.load(handle)
num_reps = np.shape(data["amplified_spectra"])[1]
wl_ax = data["amplified_spectra"][0, 0, 0, :]
rf_freqs_ghz = data["rf_freqs"] * 10**-9
best_idx = 1
fig1, ax = plt.subplots()
ax.plot(wl_ax, data["pump_ref"][1])
ax.set_xlabel(r"Wavelength [nm]")
ax.set_ylabel(r"Power [dBm]")
ax.set_xlim(1549.5, 1550.5)
ax.set_xticks(np.arange(1549.5, 1550.51, 0.25))
ax.ticklabel_format(useOffset=False, style="plain")
ax.set_title("Pump only")
fig2, ax = plt.subplots(nrows=1, ncols=1)
for i in range(num_reps):
    ax.plot(wl_ax, data["amplified_spectra"][best_idx, i, 1])
ax.set_xlabel(r"Wavelength [nm]")
ax.set_ylabel(r"Power [dBm]")
ax.set_xlim(1549.5, 1550.5)
ax.set_xticks(np.arange(1549.5, 1550.51, 0.25))
ax.ticklabel_format(useOffset=False, style="plain")
ax.set_title(
    f"Amplified spectra, rf freq: {rf_freqs_ghz[best_idx]:.2f} GHz, pol sweeps w scrambler"
)
mean_spectra = np.mean(data["amplified_spectra"][:, :, 1, :], axis=1)
std_spectra = np.std(data["amplified_spectra"][:, :, 1, :], axis=1)
fig3, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(wl_ax, mean_spectra[best_idx])
ax.fill_between(
    wl_ax,
    mean_spectra[best_idx] - std_spectra[best_idx],
    mean_spectra[best_idx] + std_spectra[best_idx],
    alpha=0.3,
)
ax.set_xlabel(r"Wavelength [nm]")
ax.set_ylabel(r"Power [dBm]")
ax.set_xlim(1549.5, 1550.5)
ax.set_xticks(np.arange(1549.5, 1550.51, 0.25))
ax.ticklabel_format(useOffset=False, style="plain")
ax.set_title(
    f"Mean+std of amp. spectra, rf freq: {rf_freqs_ghz[best_idx]:.2f} GHz, pol sweeps w scrambler"
)
fig4, ax = plt.subplots()
fig5, ax2 = plt.subplots()
for i in range(len(mean_spectra)):
    ax.plot(wl_ax, mean_spectra[i], label=f"{data['rf_freqs'][i]*10**-9:.2f}")
    ax2.plot(
        wl_ax, data["ref_spectra"][i, 1], label=f"{data['rf_freqs'][i]*10**-9:.2f}"
    )
ax.legend()
ax.set_xlim(1549.5, 1550.5)
ax.set_xticks(np.arange(1549.5, 1550.51, 0.25))
ax.ticklabel_format(useOffset=False, style="plain")
ax.set_xlabel(r"Wavelength [nm]")
ax.set_ylabel(r"Power [dBm]")
ax.set_title("Mean of amp. spectra, tried to optimize SSB for each freq")
ax2.set_xlim(1549.5, 1550.5)
ax2.set_xticks(np.arange(1549.5, 1550.51, 0.25))
ax2.ticklabel_format(useOffset=False, style="plain")
ax2.set_xlabel(r"Wavelength [nm]")
ax2.set_ylabel(r"Power [dBm]")
ax2.set_title("Ref spectra, tried to optimize SSB for each freq")
ax2.legend()
if save_figs:
    fig1.savefig(f"{fig_dir}/{extra_sub_dir}/pump_only.pdf", bbox_inches="tight")
    fig2.savefig(
        f"{fig_dir}/{extra_sub_dir}/amp_spectra_rf-freq={rf_freqs_ghz[best_idx]:.2f}.pdf",
        bbox_inches="tight",
    )
    fig3.savefig(
        f"{fig_dir}/{extra_sub_dir}/mean_std_amp_spectra_rf-freq={rf_freqs_ghz[best_idx]:.2f}.pdf",
        bbox_inches="tight",
    )
    fig4.savefig(
        f"{fig_dir}/{extra_sub_dir}/amp_spectra_multiple-freqs.pdf",
        bbox_inches="tight",
    )
    fig5.savefig(
        f"{fig_dir}/{extra_sub_dir}/ref_spectra-multiple-spectra.pdf",
        bbox_inches="tight",
    )
plt.show()
# |%%--%%| <KbtaV4IhSK|h2iTeJeHVf>


def calc_gain(
    wl_ax: np.ndarray,
    ref_spectra: np.ndarray,
    amplified_spectra: np.ndarray,
    pump_spectrum: np.ndarray,
    osa_res_tol_factor: int = 5,
) -> tuple:
    ref_spectra = ref_spectra[1]
    amplified_spectra = amplified_spectra[:, 1]
    pump_spectrum = pump_spectrum[1]
    osa_res = np.round(wl_ax[1] - wl_ax[0], 3)
    wl_tol = osa_res_tol_factor * osa_res
    ref_powers = ref_spectra
    amplified_spec = amplified_spectra
    amplified_powers = np.zeros(num_reps)

    sig_wl = wl_ax[np.argmax(ref_powers)]
    sig_power = np.max(ref_powers)
    wl_idxs_within_tol = np.where(
        (wl_ax > sig_wl - wl_tol) & (wl_ax < sig_wl + wl_tol)
    )[0]
    pump_power = np.max(pump_spectrum[wl_idxs_within_tol])
    for i in range(len(amplified_spectra)):
        amplified_power = np.max(amplified_spec[i, wl_idxs_within_tol])
        amplified_powers[i] = amplified_power
    amplified_powers = mW_to_dBm(dBm_to_mW(amplified_powers) + dBm_to_mW(pump_power))
    mean_power = np.mean(amplified_powers)
    std_power = np.std(amplified_powers)
    gain = mean_power - sig_power
    return gain, mean_power, std_power


gain, mean_power, std_power = [
    np.zeros(len(data["amplified_spectra"])) for _ in range(3)
]
for i, rf_freq in enumerate(data["rf_freqs"]):
    wl_ax = data["ref_spectra"][i, 0]
    gain[i], mean_power[i], std_power[i] = calc_gain(
        wl_ax,
        data["ref_spectra"][i],
        data["amplified_spectra"][i],
        data["pump_ref"],
    )

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.errorbar(rf_freqs_ghz, gain, std_power, fmt="o", capsize=20)
ax.set_xlabel(r"Rf freq [GHz]")
ax.set_ylabel(r"Brillouin gain [dB]")
fig.savefig(f"{fig_dir}/{extra_sub_dir}/gain_vs_rf-freq.pdf", bbox_inches="tight")
plt.show()
# |%%--%%| <h2iTeJeHVf|xqtfU1WmIU>
# Time traces of Brill amp
save_figs = True
filename = "../data/brillioun_amp/time_traces/traces_with-and-without-amplification.pkl"
with open(f"{filename}", "rb") as handle:
    trace_data = pickle.load(handle)
time_arr = trace_data["ref_traces"][0, 0, :] * 10**9
ref_traces = trace_data["ref_traces"][:, 1, :]
amplified_traces = trace_data["amplified_traces"][:, 1, :]
ref_traces_mean = np.mean(ref_traces, axis=0)
ref_traces_std = np.std(ref_traces, axis=0)
amplified_traces_mean = np.mean(amplified_traces, axis=0)
amplified_traces_std = np.std(amplified_traces, axis=0)

fig1, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(time_arr, ref_traces_mean)
ax.fill_between(
    time_arr,
    ref_traces_mean - ref_traces_std,
    ref_traces_mean + ref_traces_std,
    alpha=0.3,
)
ax.set_xlabel(r"Time [ns]")
ax.set_ylabel(r"Voltage [a.u.]")
fig2, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(time_arr, amplified_traces_mean)
ax.fill_between(
    time_arr,
    amplified_traces_mean - amplified_traces_std,
    amplified_traces_mean + amplified_traces_std,
    alpha=0.3,
)
ax.set_xlabel(r"Time [ns]")
ax.set_ylabel(r"Voltage [a.u.]")
if save_figs:
    fig1.savefig(f"{fig_dir}/{extra_sub_dir}/time-trace_ref.pdf", bbox_inches="tight")
    fig2.savefig(
        f"{fig_dir}/{extra_sub_dir}/time-trace_amplified.pdf", bbox_inches="tight"
    )
plt.show()
