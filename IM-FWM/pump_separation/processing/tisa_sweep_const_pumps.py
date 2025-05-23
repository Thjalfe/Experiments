# %%
from pump_separation.funcs.utils import extract_sig_wl_and_ce_multiple_spectra
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# plt.style.use("custom")
plt.rcParams["figure.figsize"] = (16, 11)
plt.ioff()

data_loc = "../data/sweep_multiple_separations_w_polopt/cleo/manual/tisa_set_to_lin_fit/merged_data.pkl"
data_loc = "../data/tisa_sweep_to_find_opt/pump_wl_mean_1590/merged_data.pkl"
data_loc = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\4MSI\tisa_sweep_to_find_opt\pump_modes=02_equal_input_pump_p\merged_data.pkl"
# data_loc = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\tisa_sweep_to_find_opt\pump_wl_mean_1606.5\pump_wl=(1609.0, 1604.0).pkl"
# data_loc = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\Experiments\IM-FWM\pump_separation\data\4MSI\tisa_sweep_to_find_opt\pump_wl_mean_1606.5\pump_wl=(1609.0, 1604.0).pkl"
# data_loc = "../data/sweep_multiple_separations_w_polopt/cleo/old_linear_fit_from_v_old_c_plus_l/data.pkl"
with open(data_loc, "rb") as f:
    data = pickle.load(f)

fig_folder = (
    "../../figs/sweep_multiple_separations_w_polopt/cleo_us_2023/pol_con_not_used"
)
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

save_figs = False
save_spectra = False


# |%%--%%| <PuEp92P5VV|zj1zyrM71V>
# data = {(1609.0, 1604.0): data}
pump_wl_pairs = list(data.keys())
mean_p_wl = np.mean(pump_wl_pairs[0])

duty_cycles = data[pump_wl_pairs[0]]["params"]["duty_cycles"]
pump_sep_ax = np.array([np.abs(pair[1] - pair[0]) for pair in pump_wl_pairs])
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
# ce = ce_dict[(1609.0, 1604.0)][0.2]
# sig_wl = sig_wl_dict[(1609.0, 1604.0)][0.2]
# %%
pumpwl_keys = list(ce_dict.keys())
plt.plot(sig_wl_dict[pumpwl_keys[-3]][0.1], ce_dict[pumpwl_keys[-3]][0.1])
# %%
mean_sig_wl_at_max_ce = np.squeeze(mean_sig_wl_at_max_ce)
longpumpwl_list = [pumpwl[0] for pumpwl in pump_wl_pairs]
plt.plot(longpumpwl_list[:-3], mean_sig_wl_at_max_ce[:-3])
# linear fit up to :-3
linear_fit = np.polyfit(longpumpwl_list[:-3], mean_sig_wl_at_max_ce[:-3], 1)
linear_fit_fn = np.poly1d(linear_fit)
plt.plot(longpumpwl_list[:-3], linear_fit_fn(longpumpwl_list[:-3]))
# %%
if save_spectra:
    plt.ioff()
    for pump_wl_pair in pump_wl_pairs:
        pump_sep = np.abs(pump_wl_pair[1] - pump_wl_pair[0])
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
pump_wl_idx = 3
pump_wl_pair = pump_wl_pairs[pump_wl_idx]
dc_idx = 0
dc = duty_cycles[dc_idx]
rep_num = 0
spectra = np.array(data[pump_wl_pair]["spectra"][dc])
idx = np.argmax(ce_dict[pump_wl_pair][dc])
spectrum = spectra[idx, rep_num, :, :]
spectrum[1, spectrum[1, :] < -70] = np.nan
fig, ax = plt.subplots()
ax.plot(spectrum[0, :], spectrum[1, :])
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Power (dBm)")
ax.set_title(f"Pump WL Pair = {pump_wl_pair}, DC = {dc}, Rep Num = {rep_num}")
plt.show()
# |%%--%%| <ZHwRWFGHeg|A8exSCotRy>
# plot all spectra for a specific separation and rep
pump_wl_idx = 2
pump_wl_pair = pump_wl_pairs[pump_wl_idx]
dc_idx = 0
dc = duty_cycles[dc_idx]
rep_num = 0
spectra = np.array(data[pump_wl_pair]["spectra"][dc])
fig, ax = plt.subplots()
for i in range(np.shape(spectra)[0]):
    spectrum = spectra[i, rep_num, :, :]
    spectrum[1, spectrum[1, :] < -70] = np.nan
    ax.plot(spectrum[0, :], spectrum[1, :])
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Power (dBm)")
ax.set_title(f"Pump WL Pair = {pump_wl_pair}, DC = {dc}, Rep Num = {rep_num}")
plt.show()

# |%%--%%| <VgSaNZXPa6|ZHwRWFGHeg>
for dc_idx, dc in enumerate(duty_cycles):
    fig, ax = plt.subplots()
    for pump_wl_pair_idx, pump_wl_pair in enumerate(pump_wl_pairs):
        ax.plot(
            sig_wl_dict[pump_wl_pair][dc],
            ce_dict[pump_wl_pair][dc],
            "o-",
            label=pump_sep_ax[pump_wl_pair_idx],
        )
    ax.set_xlabel("Signal Wavelength (nm)")
    ax.set_ylabel(r"CE (dB)")
    ax.set_title(f"Duty Cycle = {dc}")
    ax.legend(title="Pump Sep (nm)")
    if save_figs:
        fig.savefig(
            os.path.join(fig_folder, f"ce_vs_sig_wl_dc_{dc}.pdf"), bbox_inches="tight"
        )

# |%%--%%| <A8exSCotRy|hjzGaspXAv>
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
        pump_sep_ax,
        max_ce_vs_pumpsep[dc_idx, :] - ce_offset[dc_idx],
        "o-",
        label=dc,
        # pump_sep_ax, max_ce_vs_pumpsep[dc_idx, :], "o-", label=dc
    )
ax_ticks = ax.get_xticks()
ax2.set_xlim(mean_sig_wl_at_max_ce[0], mean_sig_wl_at_max_ce[-1])
ax2.set_xlabel(r"$\lambda_s$ (nm)")
ax2.grid(False)
ax.set_xlabel("Pump Separation (nm)")
ax.set_ylabel(r"$\eta$ (dB)")
# ax.set_title("Max CE vs Pump Separation")
ax.legend(title="Duty Cycle")
if save_figs:
    fig.savefig(os.path.join(fig_folder, "max_ce_vs_pumpsep.pdf"), bbox_inches="tight")
    fig.savefig(
        "../../../../../papers/cleo_us_2023/figs/max_ce_vs_pumpsep.pdf",
        bbox_inches="tight",
    )
# |%%--%%| <hjzGaspXAv|oj2EO72Fal>
fig, ax = plt.subplots()
for dc_idx, dc in enumerate(duty_cycles):
    ax.plot(pump_sep_ax, sig_wl_at_max_ce[dc_idx, :], "o-", label=dc)
ax.set_xlabel("Pump Separation (nm)")
ax.set_ylabel(r"$\lambda_s$ at Max CE (nm)")
ax.set_title(r"$\lambda_s$ vs Pump Separation")
ax.legend(title="Duty Cycle")
# linear fit
linear_fit = np.polyfit(pump_sep_ax, mean_sig_wl_at_max_ce, 1)
linear_fit_fn = np.poly1d(linear_fit)
fig, ax = plt.subplots()
ax.plot(pump_sep_ax, mean_sig_wl_at_max_ce, "o-", label="Data")
ax.plot(pump_sep_ax, linear_fit_fn(pump_sep_ax), label="Linear Fit")
ax.set_xlabel("Pump Separation (nm)")
ax.set_ylabel(r"$\lambda_s$ at Max CE (nm)")
ax.set_title(r"$\lambda_s$ vs Pump Separation, linear fit on mean of all DC")
ax.legend()
if save_figs:
    fig.savefig(
        os.path.join(fig_folder, "mean_sig_wl_at_max_ce_vs_pumpsep.pdf"),
        bbox_inches="tight",
    )
np.savetxt(f"./fits/modelocked_1571_pump_fit.txt", linear_fit)
