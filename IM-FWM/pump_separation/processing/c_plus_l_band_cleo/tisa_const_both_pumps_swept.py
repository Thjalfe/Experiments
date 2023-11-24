from pump_separation.processing.c_plus_l_band_cleo.cleo_help_funcs import (
    process_ce_data_for_pump_sweep_around_opt,
)

import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use("custom")
plt.rcParams["figure.figsize"] = (16, 11)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.ion()
data_loc_both_sweep = (
    "../../data/C_plus_L_band/cleo/pol_opt_auto/pump_wl=(1607.0, 1533.0)_both_pumps_sweep_around_opt.pkl"
)
fig_folder = "../../figs/C_plus_L_band/cleo_us_2023/pol_opt_auto"
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

save_figs = False
save_spectra = False


(
    exp_params,
    ce_dict,
    sig_wl_dict,
    idler_wl_dict,
    ce_dict_processed,
    sig_wl_dict_processed,
    idler_wl_dict_processed,
    ce_dict_std,
    ce_dict_best_for_each_ando_sweep,
    ce_dict_best_loc_for_each_ando_sweep,
) = process_ce_data_for_pump_sweep_around_opt(data_loc_both_sweep, np.mean)

