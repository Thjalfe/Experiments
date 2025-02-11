import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("custom")

datafile = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\LPG_T_3_1600\TMSI_tisa_wl=965.0_pump_L_band_sweep\sweep_both_pumps\p1_wl=1607-1612_p2_wl=1570-1610_p1_stepsize=1_p2_stepsize=0.2\data.pkl"
datafile = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\LPG_T_3_1600\TMSI_tisa_wl=965.0_pump_L_band_sweep\sweep_both_pumps_TEST\p1_wl=1607-1607_p2_wl=1570-1610_p1_stepsize=1_p2_stepsize=0.2\data.pkl"
datafile = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\LPG_T_3_1600\TMSI_tisa_wl=965.1_pump_L_band_sweep\sweep_both_pumps_\p1_wl=1607-1611.0_p2_wl=1567.4000000000033-1610_p1_stepsize=0.5_p2_stepsize=0.2\data.pkl"
datafile = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\LPG_T_3_1600\TMSI_tisa_wl=965.1_pump_L_band_sweep\sweep_both_pumps__\p1_wl=1607-1611.200000000001_p2_wl=1573.0000000000043-1610_p1_stepsize=0.2_p2_stepsize=0.2\data.pkl"
datafile = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\LPG_T_3_1600\TMSI_tisa_wl=965.1_pump_L_band_sweep\sweep_both_pumps______\p1_wl=1607.0-1611.0_p2_wl=1568.4-1610.0_p1_stepsize=0.50_p2_stepsize=0.20\data.pkl"
datafile2 = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\LPG_T_3_1600\TMSI_tisa_wl=965.1_pump_L_band_sweep\sweep_both_pumps______\p1_wl=1608.5-1610.0_p2_wl=1577.0-1610.0_p1_stepsize=0.50_p2_stepsize=0.20\data.pkl"
datafile = r"../data/LPG_T_3_1600/TMSI_tisa_wl=965.1_pump_L_band_sweep/sweep_both_pumps______/p1_wl=1607.0-1611.0_p2_wl=1568.4-1610.0_p1_stepsize=0.50_p2_stepsize=0.20/data.pkl"
datafile2 = r"../data/LPG_T_3_1600/TMSI_tisa_wl=965.1_pump_L_band_sweep/sweep_both_pumps______/p1_wl=1608.5-1610.0_p2_wl=1577.0-1610.0_p1_stepsize=0.50_p2_stepsize=0.20/data.pkl"
with open(datafile, "rb") as f:
    data = pickle.load(f)
with open(datafile2, "rb") as f:
    data2 = pickle.load(f)


def get_ce_peak_pol_opt(data):
    p1_wls = list(data.keys())
    ce_peak_pol_opt = []
    p2_wls = []
    for p1_wl in p1_wls:
        if type(p1_wl) == str:
            continue
        ce_peak_pol_opt.append(data[p1_wl]["ce_peak_pol_opt"])
        p2_wls.append(data[p1_wl]["fine_step_p2_max_ce"])
    p1_wls = [num for num in p1_wls if isinstance(num, (int, float))]

    return ce_peak_pol_opt, p2_wls, p1_wls


def calc_idler_wl(p1_wl, p2_wl, sig_wl=965.1):
    return 1 / (1 / sig_wl + 1 / p1_wl - 1 / p2_wl)


save_fig = True
fig_dir = "../figs/LPG_T_3_1600_both_pumps_swept/"
if os.path.exists(fig_dir) is False:
    os.makedirs(fig_dir)
ce_peak_pol_opt, p2_wls, p1_wls = get_ce_peak_pol_opt(data)
wl_sep = np.array(p1_wls) - np.array(p2_wls)
idler_wls = [calc_idler_wl(p1_wl, p2_wl) for p1_wl, p2_wl in zip(p1_wls, p2_wls)]
ce_peak_pol_opt2, p2_wls2, p1_wls2 = get_ce_peak_pol_opt(data2)
wl_sep2 = np.array(p1_wls2) - np.array(p2_wls2)
idler_wls2 = [calc_idler_wl(p1_wl, p2_wl) for p1_wl, p2_wl in zip(p1_wls2, p2_wls2)]
fig, ax = plt.subplots()
ax.plot(wl_sep, ce_peak_pol_opt, "o-")
ax.plot(wl_sep2, ce_peak_pol_opt2, "o-")
ax2 = ax.twiny()
ax2.plot(idler_wls, ce_peak_pol_opt, "o-")
for line in ax2.get_lines():
    line.set_visible(False)
ax2.set_xlabel("Idler wavelength (nm)")
ax2.grid(False)
# ax.plot(wl_sep, np.mean(ce_peak_pol_opt, 1), "-o")
ax.set_xlabel("Pump separation (nm)")
ax.set_ylabel("CE peak (dB)")
if save_fig:
    fig.savefig(
        os.path.join(fig_dir, "ce-peak_vs_pump-sep_sig-wl=965_dc=0.05.pdf"),
        bbox_inches="tight",
    )
plt.show()
