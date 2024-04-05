import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle

plt.style.use("custom")
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.rcParams["figure.figsize"] = (16, 11)
plt.style.use("large_fonts")
plt.ion()


def dBm_to_mW(dBm):
    return 10 ** (dBm / 10)


def lin_to_dB(lin):
    return 10 * np.log10(lin)


pump_data_loc = "../../data/sweep_multiple_separations_w_polopt/cleo/pump_powers.pkl"
pump_data_loc = (
    "../../data/sweep_multiple_separations_w_polopt/cleo/pump_powers_after_FMF.pkl"
)
with open(pump_data_loc, "rb") as f:
    pump_data = pickle.load(f)

lpg_data_loc = "../../data/lpg/LPGM21.xlsx"
sheet_2_df = pd.read_excel(lpg_data_loc, sheet_name="LPGM21port_wavres")
stats_df = sheet_2_df.groupby("# Wavelength [nm]")["conversion eff"].agg(
    ["mean", "std"]
)
stats_df_dB = sheet_2_df.groupby("# Wavelength [nm]")[" Cross talk [dB]"].agg(
    ["mean", "std"]
)
fig_dir = "../../figs/sweep_multiple_separations_w_polopt/cleo_us_2023/pump_powers/"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
save_figs = False

merged_data = {}

for duty_cycle in pump_data["L_band"]:
    merged_data[duty_cycle] = np.concatenate(
        (pump_data["L_band"][duty_cycle], pump_data["C_band"][duty_cycle])
    )
for duty_cycle, array in merged_data.items():
    # Assume the wavelength is in the first column (index 0)
    sorted_indices = np.argsort(array[:, 0])
    merged_data[duty_cycle][merged_data[duty_cycle][:, 1] > 35] = np.nan
    pump_data["L_band"][duty_cycle][pump_data["L_band"][duty_cycle][:, 1] > 35] = np.nan
    pump_data["C_band"][duty_cycle][pump_data["C_band"][duty_cycle][:, 1] > 35] = np.nan
    merged_data[duty_cycle] = array[sorted_indices]

fig, ax = plt.subplots()
for i_dc, duty_cycle in enumerate(merged_data):
    # ax.plot(merged_data[duty_cycle][:, 0], dBm_to_mW(merged_data[duty_cycle][:, 1]), '-x', label=duty_cycle)
    ax.plot(
        pump_data["L_band"][duty_cycle][:, 0],
        dBm_to_mW(pump_data["L_band"][duty_cycle][:, 1]),
        "-x",
        label=duty_cycle,
        color=color_cycle[i_dc],
    )
    ax.plot(
        pump_data["C_band"][duty_cycle][:, 0],
        dBm_to_mW(pump_data["C_band"][duty_cycle][:, 1]),
        "-x",
        color=color_cycle[i_dc],
    )
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Pump Power (dBm)")
ax.legend()
if save_figs:
    fig.savefig(f"{fig_dir}pump_powers.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(11, 8))
ax.errorbar(
    stats_df.index[1:],
    stats_df["mean"][1:] * 100,
    yerr=stats_df["std"][1:] * 100,
    fmt="-x",
    markersize=10,
)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel(r"LP$_{01}\rightarrow$LP$_{11}$ (\%)")
if save_figs:
    fig.savefig(f"{fig_dir}lpg_conversion_eff.pdf", bbox_inches="tight")
fig.savefig(
    "../../../../../papers/cleo_us_2024/presentation/figs/setup_method/lpg_conversion_eff2.pdf",
    bbox_inches="tight",
)

fig, ax = plt.subplots()
ax.errorbar(stats_df_dB.index, stats_df_dB["mean"], yerr=stats_df_dB["std"], fmt="-x")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Cross talk (dB)")
if save_figs:
    fig.savefig(f"{fig_dir}lpg_cross_talk.pdf", bbox_inches="tight")
# |%%--%%| <9jdEEKTbjH|2swx2uFoNK>
## Pump power correction in correct mode etc
fmf_loss = (
    0.19 * 0.9
)  # data from http://www.fp7-cosign.eu/wp-content/uploads/2015/09/COSIGN-deliverable-D2-3-submitted_final_v4.pdf with 900 m fiber
fit_lpg = np.polyfit(stats_df.index[1:], stats_df["mean"][1:], 6)
eval_lpg = np.poly1d(fit_lpg)
fig, ax = plt.subplots()
ax.errorbar(
    stats_df.index[1:],
    stats_df["mean"][1:] * 100,
    yerr=stats_df["std"][1:] * 100,
    fmt="-x",
    markersize=10,
)
ax.plot(stats_df.index[1:], eval_lpg(stats_df.index[1:]) * 100, "--")

p_l_band_corrected = {}
p_c_band_corrected = {}
p_merged_corrected = {}
p_total_corrected = {}

for duty_cycle in merged_data.keys():
    p_merged_corrected[duty_cycle] = np.copy(merged_data[duty_cycle])
    p_merged_corrected[duty_cycle][:, 1] = dBm_to_mW(
        p_merged_corrected[duty_cycle][:, 1] + fmf_loss
    )
    p_merged_corrected[duty_cycle][:, 1] *= eval_lpg(
        p_merged_corrected[duty_cycle][:, 0]
    )
    p_merged_corrected[duty_cycle][:, 1] = lin_to_dB(
        p_merged_corrected[duty_cycle][:, 1]
    )
    p_l_band_corrected[duty_cycle] = np.copy(pump_data["L_band"][duty_cycle])
    p_l_band_corrected[duty_cycle][:, 1] = dBm_to_mW(
        p_l_band_corrected[duty_cycle][:, 1] + fmf_loss
    )
    p_l_band_corrected[duty_cycle][:, 1] *= eval_lpg(
        p_l_band_corrected[duty_cycle][:, 0]
    )
    p_c_band_corrected[duty_cycle] = np.copy(pump_data["C_band"][duty_cycle])
    p_c_band_corrected[duty_cycle][:, 1] = dBm_to_mW(
        p_c_band_corrected[duty_cycle][:, 1] + fmf_loss
    )
    p_c_band_corrected[duty_cycle][:, 1] *= eval_lpg(
        p_c_band_corrected[duty_cycle][:, 0]
    )
    p_total_corrected[duty_cycle] = (
        p_l_band_corrected[duty_cycle][6:-1, 1] + p_c_band_corrected[duty_cycle][:, 1]
    )
    p_c_band_corrected[duty_cycle][:, 1] = lin_to_dB(
        p_c_band_corrected[duty_cycle][:, 1]
    )
    p_l_band_corrected[duty_cycle][:, 1] = lin_to_dB(
        p_l_band_corrected[duty_cycle][:, 1]
    )
    p_total_corrected[duty_cycle] = lin_to_dB(p_total_corrected[duty_cycle])

# |%%--%%| <2swx2uFoNK|KljMpJat7P>
fig, ax = plt.subplots()
for duty_cycle in p_merged_corrected.keys():
    ax.plot(
        dBm_to_mW(p_total_corrected[duty_cycle]),
        label=duty_cycle,
    )
ax.legend()
