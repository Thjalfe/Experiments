import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re

plt.style.use("custom")
plt.ion()

long_wl_ref_files = glob.glob("../data/long_wl_ref/*.csv")
long_wl_ref_name_list = ["long_wl_after_tap", "long_wl_after_freespace"]
long_wl_ref_dict = {}
for i in range(len(long_wl_ref_files)):
    long_wl_ref_dict[long_wl_ref_name_list[i]] = np.loadtxt(
        long_wl_ref_files[i], delimiter=",", skiprows=1
    ).T
pump_spec_idx = 1
pump_spec = long_wl_ref_dict[long_wl_ref_name_list[pump_spec_idx]]
plot_title_list = [
    "Spectra, long wavelength after tap",
    "Spectra, long wavelength after freespace",
]

data_dir = ["../data/1589_1591", "../data/1585_1595"]


def load_files_from_dir(dir: str) -> dict:
    nums_in_dir_name = re.findall(r"\d+", dir)
    sep_nm = np.abs(int(nums_in_dir_name[0]) - int(nums_in_dir_name[1]))
    files = glob.glob(os.path.join(dir, "*.csv"))
    data_dict = {"file_names": []}
    pol_opt_data_dict = {"file_names": []}
    pol_file_num_count = 0
    data_dir_num_count = 0
    for i, file in enumerate(files):
        if "pol" in file:
            file_name = os.path.basename(file)
            pol_opt_data_dict[pol_file_num_count] = np.loadtxt(
                file, delimiter=",", skiprows=1
            ).T
            pol_opt_data_dict["file_names"].append(file_name)
            pol_file_num_count += 1

        else:
            file_name = os.path.basename(file)
            data_dict[data_dir_num_count] = np.loadtxt(
                file, delimiter=",", skiprows=1
            ).T
            data_dict["file_names"].append(file_name)
            data_dir_num_count += 1
    return data_dict, pol_opt_data_dict, sep_nm


data_dict = {}
pol_opt_data_dict = {}
for dir in data_dir:
    data_dict_tmp, pol_opt_data_dict_tmp, sep_nm = load_files_from_dir(dir)
    data_dict[sep_nm] = data_dict_tmp
    pol_opt_data_dict[sep_nm] = pol_opt_data_dict_tmp


fig_path = "../figs"
if os.path.exists(fig_path) is False:
    os.mkdir(fig_path)
save_figs = False
# |%%--%%| <YDbu9K7t0m|c7YsmhKzbT>
# plotting raw spectra
labels = ["Distant", "Nearby"]
for sep_nm in data_dict.keys():
    fig, ax = plt.subplots(1, 2)
    for i in range(len(data_dict[sep_nm]["file_names"])):
        ax[i].plot(
            data_dict[sep_nm][i][0, :],
            data_dict[sep_nm][i][1, :],
        )
        ax[i].set_title(f"sep:{sep_nm} nm, {labels[i]}")
        ax[i].set_xlabel("Wavelength (nm)")
        ax[i].set_ylabel("Power (dBm)")
    fig.tight_layout()
    if save_figs:
        fig.savefig(
            os.path.join(fig_path, f"spectra_{sep_nm}.pdf"), bbox_inches="tight"
        )
# plotting pol opt spectra
tot_time = 50  # s
num_points = len(pol_opt_data_dict[sep_nm][0][0, :])
time_axis = np.linspace(0, tot_time, num_points)
labels_pol = ["Nearby", "Distant"]
for sep_nm in pol_opt_data_dict.keys():
    fig, ax = plt.subplots(1, 2)
    for i in range(len(pol_opt_data_dict[sep_nm]["file_names"])):
        mask = pol_opt_data_dict[sep_nm][i][1, :] > -80
        ax[i].plot(
            time_axis[mask],
            pol_opt_data_dict[sep_nm][i][1, mask],
        )
        ax[i].set_title(f"sep:{sep_nm} nm, {labels_pol[i]}")
        ax[i].set_xlabel("Time (s)")
        ax[i].set_ylabel("Power (dBm)")
    fig.tight_layout()
    if save_figs:
        fig.savefig(
            os.path.join(fig_path, f"pol_opt_{sep_nm}.pdf"), bbox_inches="tight"
        )
# |%%--%%| <c7YsmhKzbT|773fVW0Jw4>
# %% Broken axis plot
short_wl_spec = long_wl_ref_dict["short_wl"]
width_ratios = [1.5, 1]
fig, (ax, ax2) = plt.subplots(
    1,
    2,
    sharey=True,
    facecolor="w",
    gridspec_kw={"width_ratios": width_ratios, "wspace": 0.1},
)

# Determine the break point on the x-axis
break_start = np.max(
    short_wl_spec[0, :]
)  # This should be set to where you want the axis to break
break_end = np.min(
    pump_spec[0, :]
)  # This should be set to where you want the axis to resume

# Plot the two parts of the x-axis on the different subplots
ax.plot(short_wl_spec[0, :], short_wl_spec[1, :], label="Sub Data")
ax2.plot(pump_spec[0, :], pump_spec[1, :], label="Pump Spectrum")

# Set the limits for the broken axes
ax.set_xlim(np.min(short_wl_spec[0, :]), break_start)
ax2.set_xlim(break_end, np.max(pump_spec[0, :]))
ax.set_xlim(np.min(short_wl_spec[0]), np.max(short_wl_spec[0]))
ax2.set_xlim(np.min(pump_spec[0, :]), np.max(pump_spec[0, :]))

# Hide the spines between ax and ax2
ax.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax.yaxis.tick_left()
ax.spines["top"].set_visible(True)
ax2.spines["top"].set_visible(True)
ax2.spines["right"].set_visible(True)
ax.tick_params(labelright="off")
ax2.yaxis.tick_right()


# Add diagonal lines to indicate the break in the axis
def draw_diagonals(ax, direction="lr", d=0.015, ax_width=1):
    kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
    dx = d * ax_width
    dy = d
    if direction == "lr":  # left-to-right
        ax.plot((1 - dx, 1 + dx), (-dy, +dy), **kwargs)
        ax.plot((1 - dx, 1 + dx), (1 - dy, 1 + dy), **kwargs)
    elif direction == "rl":  # right-to-left
        ax.plot((-dx, +dx), (1 - dy, 1 + dy), **kwargs)
        ax.plot((-dx, +dx), (-dy, +dy), **kwargs)


d = 0.015  # Size of the diagonal lines
draw_diagonals(ax, "lr", d=d, ax_width=width_ratios[1])
draw_diagonals(ax2, "rl", d=d, ax_width=width_ratios[0])

# ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Power (dBm)")
# ax2.set_xlabel("Wavelength (nm)")
fig.text(
    0.5, 0.04, "Wavelength (nm)", ha="center", va="center", fontsize=28 * 1.44
)  # Common x-label
ax.tick_params(labelright=False, left=False)
fig.text(
    0.5,
    0.94,
    plot_title_list[pump_spec_idx],
    ha="center",
    va="center",
    fontsize=28 * 1.44,
)
if save_figs:
    fig.savefig(
        os.path.join(fig_path, f"spectra_{name_list[pump_spec_idx]}.pdf"),
        bbox_inches="tight",
    )
