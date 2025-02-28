import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

save_figs = True

plt.style.use("custom")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
data_dir = "../data/tisa_pump_superk_spectra/"
filenames = glob.glob(data_dir + "*.csv")
pumps_on_filenames = []
pumps_on = []
pumps_off_filenames = []
pumps_off = []
for filename in filenames:
    if "pumps-on" in filename:
        pumps_on_filenames.append(filename)
        pumps_on.append(np.loadtxt(filename, delimiter=","))
    elif "pumps-off" in filename:
        pumps_off_filenames.append(filename)
        pumps_off.append(np.loadtxt(filename, delimiter=","))


def remove_data_below_thresh(
    data_list: list[np.ndarray], thresh: float = -80
) -> list[np.ndarray]:
    for i in range(len(data_list)):
        data_list[i][data_list[i] < thresh] = np.nan
    return data_list


pumps_on = remove_data_below_thresh(pumps_on)
pumps_off = remove_data_below_thresh(pumps_off)
short_span = (700, 1700)
long_span = (1200, 2400)
fig_dir = "../figs/"
fig, ax = plt.subplots(1, 2, figsize=(22, 8))
for i in range(len(pumps_on)):
    if "span=700" in pumps_on_filenames[i]:
        ax[0].plot(pumps_on[i][0], pumps_on[i][1], color=colors[0])
    if "span=700" in pumps_off_filenames[i]:
        ax[0].plot(
            pumps_off[i][0],
            pumps_off[i][1],
            color=colors[1],
            zorder=10,
        )
    if "span=1200" in pumps_on_filenames[i]:
        ax[1].plot(pumps_on[i][0], pumps_on[i][1], label="Pumps on", color=colors[0])
    if "span=1200" in pumps_off_filenames[i]:
        ax[1].plot(
            pumps_off[i][0],
            pumps_off[i][1],
            label="Pumps off",
            color=colors[1],
            zorder=10,
        )
ax[0].set_title("ANDO AQ6317")
ax[1].set_title("ANDO AQ6375")
ax[0].set_ylabel("Power (dBm)")
ax[0].set_xlabel("Wavelength (nm)")
ax[1].set_xlabel("Wavelength (nm)")
ax[0].set_xlim(*short_span)
ax[1].set_xlim(*long_span)
ax[0].xaxis.set_major_locator(MaxNLocator(nbins=5))
ax[1].xaxis.set_major_locator(MaxNLocator(nbins=6))
ax[0].set_ylim(-80, 12)
ax[1].set_ylim(-80, 12)
ax[1].legend()
if save_figs:
    fig.savefig(
        fig_dir + "pump-on-off_superK_spectra-700-2400.pdf", bbox_inches="tight"
    )
    fig.savefig(
        "../../../../../logbook/2025/feb/figs/pump-on-off_superK_spectra-700-2400.pdf",
        bbox_inches="tight",
    )
