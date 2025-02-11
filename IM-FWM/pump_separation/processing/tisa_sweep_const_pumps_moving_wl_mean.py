# %%
from pump_separation.funcs.utils import extract_sig_wl_and_ce_multiple_spectra
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

plt.style.use("custom")
plt.rcParams["figure.figsize"] = (16, 11)
plt.ioff()

data_loc = "../data/sweep_multiple_separations_w_polopt/cleo/manual/tisa_set_to_lin_fit/merged_data.pkl"
data_loc = "../data/tisa_sweep_to_find_opt/pump_wl_mean_1590/merged_data.pkl"
data_loc = "../data/sweep_multiple_separations_w_polopt/pol_opt_auto/tisa_sweep_around_opt/mean_p_wl=1590.0/merged_data.pkl"
data_loc = (
    "../data/tisa_sweep_to_find_opt/moving_pump_mean/pump_wl_dist=10nm/merged_data.pkl"
)
data_loc = "../data/sweep_multiple_separations_w_polopt/pol_opt_auto/tisa_sweep_around_opt/moving_pumpwl_mean/pump_wl_dist=10.0nm/merged_data.pkl"
# data_loc = "../data/sweep_multiple_separations_w_polopt/cleo/old_linear_fit_from_v_old_c_plus_l/data.pkl"
with open(data_loc, "rb") as f:
    data = pickle.load(f)

fig_folder = "../../figs/sweep_pumpwl_mean_const_pump_sep/10_nm"
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

save_figs = False
save_spectra = False


pump_wl_pairs = list(data.keys())
mean_p_wl = np.mean(pump_wl_pairs[0])

duty_cycles = data[pump_wl_pairs[0]]["params"]["duty_cycles"]
short_pump_wl_ax = np.array([pump_wl_pair[1] for pump_wl_pair in pump_wl_pairs])
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
process_method = np.mean
for pump_wl_pair in pump_wl_pairs:
    for dc in duty_cycles:
        spectra = np.array(data[pump_wl_pair]["spectra"][dc])
        for rep in range(num_reps):
            if len(np.shape(spectra)) == 3:
                spectra = np.expand_dims(spectra, axis=1)
            if (
                np.shape(spectra)[0] == num_reps
            ):  # Fix because some old data was a bit messed up
                spectra = np.transpose(spectra, (1, 0, 2, 3))
            spectra_sgl_rep = np.transpose(
                spectra[:, rep], (0, 2, 1)
            )  # Old convention for data processing methods, easier to do this fix
            sig_wl_tmp, ce_tmp, idler_wl_tmp = extract_sig_wl_and_ce_multiple_spectra(
                spectra_sgl_rep, list(pump_wl_pair), np.shape(spectra_sgl_rep)[0]
            )  # The dimension that is being returned over is the tisa sweep across opt
            ce_dict[pump_wl_pair][dc].append(-ce_tmp)
            sig_wl_dict[pump_wl_pair][dc].append(sig_wl_tmp)
            idler_wl_dict[pump_wl_pair][dc].append(idler_wl_tmp)
        ce_dict[pump_wl_pair][dc] = process_method(ce_dict[pump_wl_pair][dc], axis=0)
        sig_wl_dict[pump_wl_pair][dc] = process_method(
            sig_wl_dict[pump_wl_pair][dc], axis=0
        )
        idler_wl_dict[pump_wl_pair][dc] = process_method(
            idler_wl_dict[pump_wl_pair][dc], axis=0
        )
sig_wl_at_max_ce = np.zeros((len(duty_cycles), len(pump_wl_pairs)))
for dc_idx, dc in enumerate(duty_cycles):
    for pump_wl_pair_idx, pump_wl_pair in enumerate(pump_wl_pairs):
        sig_wl_at_max_ce[dc_idx, pump_wl_pair_idx] = sig_wl_dict[pump_wl_pair][dc][
            np.argmax(ce_dict[pump_wl_pair][dc])
        ]
mean_sig_wl_at_max_ce = np.mean(sig_wl_at_max_ce, axis=0)
# |%%--%%| <zj1zyrM71V|HYw7Wi7LKs>
if save_spectra:
    plt.ioff()
    for pump_wl_pair in pump_wl_pairs:
        short_pump_wl_ax = np.array([pump_wl_pair[1] for pump_wl_pair in pump_wl_pairs])
        for dc in duty_cycles:
            # pump_wl_pair_idx = -1
            # dc_idx = 0
            spectra = np.array(data[pump_wl_pair]["spectra"][dc])
            idx = np.argmax(ce_dict[pump_wl_pair][dc])
            for i in range(num_reps):
                fig, ax = plt.subplots()
                ax.plot(spectra[idx, i, 0, :], spectra[idx, i, 1, :])
                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("Power (dBm)")
                dir = os.path.join(fig_folder, f"spectra/pump_wl_pair={pump_wl_pair}")
                if not os.path.exists(dir):
                    os.makedirs(dir)
                fig.savefig(
                    os.path.join(
                        dir,
                        f"pump_wl_pair={pump_wl_pair}_dc={dc}_meas_num={i + 1}.pdf",
                    ),
                    bbox_inches="tight",
                )
    plt.ion()
# |%%--%%| <HYw7Wi7LKs|VgSaNZXPa6>
# Specific spectrum to be used
pump_wl_idx = -1
pump_wl_pair = pump_wl_pairs[pump_wl_idx]
dc_idx = -1
dc = duty_cycles[dc_idx]
rep_num = 0
spectra = np.array(data[pump_wl_pair]["spectra"][dc])
idx = np.argmax(ce_dict[pump_wl_pair][dc])
spectrum = spectra[idx, rep_num, :, :]
fig, ax = plt.subplots()
ax.plot(spectrum[0, :], spectrum[1, :])
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Power (dBm)")
plt.show()
# |%%--%%| <VgSaNZXPa6|4BT7pF8CVp>
for dc_idx, dc in enumerate(duty_cycles):
    fig, ax = plt.subplots()
    for pump_wl_pair_idx, pump_wl_pair in enumerate(pump_wl_pairs):
        ax.plot(
            sig_wl_dict[pump_wl_pair][dc],
            ce_dict[pump_wl_pair][dc],
            "o-",
            label=short_pump_wl_ax[pump_wl_pair_idx],
        )
    ax.set_xlabel("Signal Wavelength (nm)")
    ax.set_ylabel(r"CE (dB)")
    ax.set_title(f"Duty Cycle = {dc}")
    ax.legend(title="Pump Sep (nm)")
    if save_figs:
        fig.savefig(
            os.path.join(fig_folder, f"ce_vs_sig_wl_dc_{dc}.pdf"), bbox_inches="tight"
        )

# |%%--%%| <4BT7pF8CVp|hjzGaspXAv>
max_ce_vs_pumpsep = np.zeros((len(duty_cycles), len(pump_wl_pairs)))
plt.rcParams["figure.figsize"] = (16, 11)
plt.ioff()
for dc_idx, dc in enumerate(duty_cycles):
    for pump_wl_pair_idx, pump_wl_pair in enumerate(pump_wl_pairs):
        max_ce_vs_pumpsep[dc_idx, pump_wl_pair_idx] = np.max(ce_dict[pump_wl_pair][dc])
ce_offset = 10 * np.log10(np.array(duty_cycles))
fig, ax = plt.subplots()
ax2 = ax.twiny()
for dc_idx, dc in enumerate(duty_cycles):
    ax.plot(
        short_pump_wl_ax,
        max_ce_vs_pumpsep[dc_idx, :] - ce_offset[dc_idx],
        "o-",
        label=dc,
        # pump_sep_ax, max_ce_vs_pumpsep[dc_idx, :], "o-", label=dc
    )
ax_ticks = ax.get_xticks()
ax2.set_xlim(mean_sig_wl_at_max_ce[0], mean_sig_wl_at_max_ce[-1])
ax2.set_xlabel(r"$\lambda_s$ (nm)")
ax2.grid(False)
ax.set_xlabel("Short Pump Wavelength (nm)")
ax.set_ylabel(r"$\eta$ (dB)")
# ax.set_title("Max CE vs Pump Separation")
ax.legend(title="Duty Cycle")
if save_figs:
    fig.savefig(os.path.join(fig_folder, "max_ce_vs_pumpsep.pdf"), bbox_inches="tight")
# |%%--%%| <hjzGaspXAv|o0ZNR7Dx1u>
fig, ax = plt.subplots()
for dc_idx, dc in enumerate(duty_cycles):
    ax.plot(short_pump_wl_ax, sig_wl_at_max_ce[dc_idx, :], "o-", label=dc)
ax.set_xlabel("Pump Separation (nm)")
ax.set_ylabel(r"$\lambda_s$ at Max CE (nm)")
ax.set_title(r"$\lambda_s$ vs Pump Separation")
ax.legend(title="Duty Cycle")
# linear fit
linear_fit = np.polyfit(short_pump_wl_ax, mean_sig_wl_at_max_ce, 1)
linear_fit_fn = np.poly1d(linear_fit)
fig, ax = plt.subplots()
ax.plot(short_pump_wl_ax, mean_sig_wl_at_max_ce, "o-", label="Data")
ax.plot(short_pump_wl_ax, linear_fit_fn(short_pump_wl_ax), label="Linear Fit")
ax.set_xlabel("Pump Separation (nm)")
ax.set_ylabel(r"$\lambda_s$ at Max CE (nm)")
ax.set_title(r"$\lambda_s$ vs Pump Separation, linear fit on mean of all DC")
ax.legend()
if save_figs:
    fig.savefig(
        os.path.join(fig_folder, "mean_sig_wl_at_max_ce_vs_pumpsep.pdf"),
        bbox_inches="tight",
    )
# np.savetxt(f"./fits/mean_pumpwl_{mean_p_wl}nm.txt", linear_fit)
# np.savetxt(f"./fits/moving_pump_wl_dist_10nm.txt", linear_fit)
