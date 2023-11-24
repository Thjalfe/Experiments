from funcs.utils import extract_sig_wl_and_ce_multiple_spectra
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

plt.style.use("custom")
plt.rcParams["figure.figsize"] = (20, 11)
plt.ion()

data_loc = "../../data/C_plus_L_band/cleo/pol_opt_not_working/data_tisa_set_to_lin_fit.pkl"
# data_loc = "../data/C_plus_L_band/cleo/old_linear_fit_from_v_old_c_plus_l/data.pkl"
data_loc = "../../data/C_plus_L_band/cleo/manual/merged_data.pkl"
with open(data_loc, "rb") as f:
    data = pickle.load(f)

fig_folder = "../../figs/C_plus_L_band/cleo_us_2023/pol_opt_manually"
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

save_figs = False
# |%%--%%| <UuOjEq2SDJ|Z8tD1UP5ps>

pump_wl_pairs = list(data.keys())
try:
    duty_cycles = data["params"]["duty_cycles"]
except KeyError:
    duty_cycles = data[pump_wl_pairs[0]]["params"]["duty_cycles"]
pump_sep_ax = np.array([np.abs(pair[1] - pair[0]) for pair in pump_wl_pairs])
try:
    num_reps = data["params"]["num_sweep_reps"]
except KeyError:
    num_reps = data[pump_wl_pairs[0]]["params"]["num_sweep_reps"]
ce_dict = {
    pump_wl_pair: {dc: [] for dc in duty_cycles} for pump_wl_pair in pump_wl_pairs
}
sig_wl_dict = {
    pump_wl_pair: {dc: [] for dc in duty_cycles} for pump_wl_pair in pump_wl_pairs
}
idler_wl_dict = {
    pump_wl_pair: {dc: [] for dc in duty_cycles} for pump_wl_pair in pump_wl_pairs
}
for pump_wl_pair in pump_wl_pairs:
    for dc in duty_cycles:
        spectra = np.array(data[pump_wl_pair]["spectra"][dc])
        cur_shape = np.shape(spectra)
        if len(cur_shape) == 3:
            spectra = np.expand_dims(spectra, axis=0)
        if num_reps > 1:
            spectra = np.transpose(spectra, (1, 0, 2, 3))
        for rep in range(num_reps):
            if num_reps > 1:
                spectra_sgl_rep = np.transpose(spectra[rep], (0, 2, 1))
            else:
                spectra_sgl_rep = np.transpose(spectra, (0, 2, 1))
            sig_wl_tmp, ce_tmp, idler_wl_tmp = extract_sig_wl_and_ce_multiple_spectra(
                spectra_sgl_rep, list(pump_wl_pair), np.shape(spectra_sgl_rep)[0]
            )
            ce_dict[pump_wl_pair][dc].append(-ce_tmp)
            sig_wl_dict[pump_wl_pair][dc].append(sig_wl_tmp)
            idler_wl_dict[pump_wl_pair][dc].append(idler_wl_tmp)
        ce_dict[pump_wl_pair][dc] = np.median(ce_dict[pump_wl_pair][dc], axis=0)
        sig_wl_dict[pump_wl_pair][dc] = np.median(sig_wl_dict[pump_wl_pair][dc], axis=0)
        idler_wl_dict[pump_wl_pair][dc] = np.median(
            idler_wl_dict[pump_wl_pair][dc], axis=0
        )

#|%%--%%| <Z8tD1UP5ps|AfZeElvidc>
max_ce_vs_pumpsep = np.zeros((len(duty_cycles), len(pump_wl_pairs)))
for dc_idx, dc in enumerate(duty_cycles):
    for pump_wl_pair_idx, pump_wl_pair in enumerate(pump_wl_pairs):
        max_ce_vs_pumpsep[dc_idx, pump_wl_pair_idx] = np.max(ce_dict[pump_wl_pair][dc])
ce_offset = 10 * np.log10(np.array(duty_cycles))
fig, ax = plt.subplots()
for dc_idx, dc in enumerate(duty_cycles):
    ax.plot(
        pump_sep_ax,
        max_ce_vs_pumpsep[dc_idx, :] - ce_offset[dc_idx],
        "o-",
        label=dc
        # pump_sep_ax, max_ce_vs_pumpsep[dc_idx, :], "o-", label=dc
    )
ax_ticks = ax.get_xticks()
# ax2 = ax.twiny()
# ax2.set_xlim(mean_sig_wl_at_max_ce[0], mean_sig_wl_at_max_ce[-1])
# ax2.set_xlabel(r"$\lambda_s$ (nm)")
# ax2.grid(False)
ax.set_xlabel("Pump Separation (nm)")
ax.set_ylabel(r"CE (dB)")
# ax.set_title("Max CE vs Pump Separation")
ax.legend(title="Duty Cycle")
if save_figs:
    fig.savefig(os.path.join(fig_folder, "max_ce_vs_pumpsep.pdf"), bbox_inches="tight")
