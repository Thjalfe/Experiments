import numpy as np
import matplotlib.pyplot as plt

plt.style.use("large_fonts")
fig_path_cleous_pres = (
    "/home/thjalfe/Documents/PhD/Projects/papers/cleo_us_2024/presentation/figs/results"
)


data_files = [
    "../data/tisa_pump/1585.0nm_1595.0nm_pumps/distant_bs_for_comp_pump1wl=1585.0_pump2wl=1595.0.csv",
    "../data/tisa_pump/1582.5nm_1597.5nm_pumps/distant_bs_for_comp_pump1wl=1582.5_pump2wl=1597.5.csv",
    "../data/tisa_pump/1580_1600nm_pumps/10dBcoupler/distant_bs_for_comp_pump1wl=1600_pump2wl=1580_reoptimized_10dB_coupler.csv",
]
data = []
c = 299792458
telecom_seps = np.array([10, 15, 20])
for file in data_files:
    data.append(np.loadtxt(file, delimiter=","))
data = np.array(data)
sig_powers = np.max(data[:, 1, :], axis=1)
idler_powers = np.max(data[:, 1, 200:], axis=1)
idler_max_locs = np.argmax(data[:, 1, 200:], axis=1)
ce_vs_sep = -(sig_powers - idler_powers)
distant_bs_power_ratios = np.array([-28.77, -30.36, -35])
distant_bs_ce_vals = distant_bs_power_ratios
idler_wavelengths = data[:, 0, 200:][np.arange(3), idler_max_locs] * 10**-9
signal_wavelengths = np.array([1595, 1597.5, 1600]) * 10**-9
idler_frequencies = c / idler_wavelengths
signal_frequencies = c / signal_wavelengths
energy_ratio_log = 10 * np.log10(signal_frequencies / idler_frequencies)
distant_bs_ce_vals = energy_ratio_log + distant_bs_power_ratios
fig, ax = plt.subplots()
ax.plot(telecom_seps, ce_vs_sep, "-o", label="Nearby BS")
ax.plot(telecom_seps, distant_bs_ce_vals, "-o", label="Distant BS")
ax.set_xlabel(r"$\Delta\lambda$ (nm)")
ax.set_ylabel("Conversion efficiency (dB)")
# Conversion efficiency diff
ce_diff = ce_vs_sep - distant_bs_ce_vals
fig, ax = plt.subplots(figsize=(10, 5))
# bold markers
ax.plot(telecom_seps, ce_diff, "-x", markersize=20, markeredgewidth=3)
ax.set_xlabel(r"$\lambda_q-\lambda_p$ (nm)")
ax.set_ylabel(r"$\Delta$CE (dB)")
ax.set_xlim([np.min(telecom_seps) - 0.2, np.max(telecom_seps) + 0.2])
ax.set_ylim([np.min(ce_diff) - 0.1, np.max(ce_diff) + 0.1])
fig.tight_layout()
# fig.savefig(
#     f"{fig_path_cleous_pres}/nearby_vs_distant_ce_comp.pdf", bbox_inches="tight"
# )
