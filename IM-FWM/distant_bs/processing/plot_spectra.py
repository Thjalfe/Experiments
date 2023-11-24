import numpy as np
import brokenaxes
import os
import pickle
import matplotlib.pyplot as plt
import glob

plt.ion()

plt.style.use("custom")
home = os.path.expanduser("~")
fig_path = f"{home}/Documents/PhD/logbook/2023/october/figs/exp/"
save_figs = True
file_list = ["../data/sig_ase_distant_bs_CW.pkl"]
data = []
for file in file_list:
    data.append(pickle.load(open(file, "rb")))
data_dict = data[0]
p_wl = list(data_dict.keys())

# |%%--%%| <RAi8cR9tVd|uRgXdrk71p>
from scipy.signal import savgol_filter, find_peaks

p_wl_idx = 0
ti_sa_idx = 10
labels = ["q", "p", "s", "i"]
for p_wl_idx in range(len(p_wl)):
# for p_wl_idx in range(3, 4):
    p_wl_tmp = p_wl[p_wl_idx]
    fig = plt.figure()
    data_tmp = data_dict[p_wl_tmp][ti_sa_idx]
    break_idx = np.where(np.diff(data_tmp[0, :]) > 10)[0][0] + 1
    data_tmp[data_tmp < -80] = np.nan
    wl_short = data_tmp[0, :break_idx]
    wl_long = data_tmp[0, break_idx:]
    power_short = data_tmp[1, :break_idx]
    power_long = data_tmp[1, break_idx:]
    power_short_smooth = savgol_filter(power_short, window_length=15, polyorder=3)
    tisa_power = np.nanargmax(power_short)
    tisa_wl = wl_short[np.nanargmax(power_short)]
    secondary_peaks, _ = find_peaks(
        power_short_smooth, prominence=2, height=(-80, tisa_power - 10)
    )
    idler_wl = wl_short[secondary_peaks[np.argmax(power_short_smooth[secondary_peaks])]]
    if idler_wl < tisa_wl:
        s_wl = 1 / (1 / p_wl_tmp - 1 / idler_wl + 1 / tisa_wl)
    else:
        s_wl = 1 / (1 / p_wl_tmp - 1 / tisa_wl + 1 / idler_wl)
    label_locs = [tisa_wl, p_wl_tmp, s_wl, idler_wl]

    bax = brokenaxes.brokenaxes(
        xlims=((wl_short[0], wl_short[-1]), (wl_long[0], wl_long[-1])), hspace=0.001
    )
    bax.plot(data_tmp[0, :], data_tmp[1, :])
    bax.set_xlabel("Wavelength (nm)", labelpad=45)
    bax.set_ylabel("Power (dBm)", labelpad=80)
    bax.vlines(label_locs[2], -80, 0, color="k", linestyle="dashed")
    # bax.set_title(f"Ti:Sa Wavelength: {tisa_wl:.2f} nm, Pump Wavelength: {p_wl[p_wl_idx]} nm")


# |%%--%%| <d0i4IBPZNg|Yfimr0huk6>

data = "../data/ok_distant_bs.pkl"
with open(data, "rb") as f:
    data = pickle.load(f)
duty_cycles = set(data["short_wl"].keys()).union(data["long_wl"].keys())
predicted_i_wl = 1 / (1 / 1608 - 1 / 1603.025 + 1 / 964.64)
peak_wavelengths = [964.666, 1603, 1608, predicted_i_wl]
labels = ["q", "p", "s", "i"]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


for i, duty_cycle in enumerate(sorted(duty_cycles)):
    fig, ax = plt.subplots(1, 2, figsize=(22, 11), sharey=True)
    for j, (data_key, subplot_title) in enumerate(
        zip(["short_wl", "long_wl"], ["Short wl", "Long wl"])
    ):
        if duty_cycle in data[data_key]:
            wl, spec = data[data_key][duty_cycle]
            spec[spec < -80] = np.nan
            ax[j].plot(wl, spec, label=f"Duty Cycle: {duty_cycle}")
        ax[j].set_title(f"{subplot_title}, Duty Cycle: {duty_cycle}")
        ax[j].set_xlabel("Wavelength")
        if duty_cycle in data[data_key]:
            ax[j].legend()
            for peak_wl, label in zip(peak_wavelengths, labels):
                if (data_key == "short_wl" and peak_wl < 1000) or (
                    data_key == "long_wl" and peak_wl >= 1000
                ):
                    idx = find_nearest(wl, peak_wl)
                    y_value = spec[idx]
                    ax[j].text(
                        peak_wl, y_value + 5, label, va="top", ha="center", fontsize=22
                    )

    ax[0].spines["right"].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    ax[1].legend().remove()
    ax[0].set_ylabel("Power (dBm)")
    if save_figs:
        plt.savefig(f"{fig_path}/965_pump/dc_{duty_cycle}.pdf", bbox_inches="tight")
plt.tight_layout()
plt.show()
