import numpy as np
import matplotlib.pyplot as plt
from external_stay.processing.helper_funcs import dBm_to_mW

plt.style.use("custom")
filename = "../data/mzm_bias_sweep/no_rf_in/coarse_sweep_find_p2p_MXER-LN-20.csv"
fig_dir = "/home/thjalfe/Documents/PhD/logbook/2025/april/figs/week1"
save_figs = True
data = np.loadtxt(filename)
fig1, ax = plt.subplots()
ax.plot(data[0], data[1])
ax.set_xlabel("DC voltage [V]")
ax.set_ylabel("Power [dBm]")
fig2, ax = plt.subplots()
ax.plot(data[0], dBm_to_mW(data[1]))
ax.set_xlabel("DC voltage [V]")
ax.set_ylabel("Power [mW]")

if save_figs:
    fig1.savefig(f"{fig_dir}/MXER-LN-20_modulator_char-log.pdf", bbox_inches="tight")
    fig2.savefig(f"{fig_dir}/MXER-LN-20_modulator_char-lin.pdf", bbox_inches="tight")
