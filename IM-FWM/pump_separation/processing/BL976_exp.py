import os
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pickle
import glob
import sys
import re
import matplotlib.pyplot as plt

current_file_path = os.path.abspath(".py")
current_file_dir = os.path.dirname(current_file_path)
project_root_dir = os.path.join(current_file_dir, "..", "..")
sys.path.append(os.path.normpath(project_root_dir))
import numpy as np
from pump_separation.funcs.utils import process_single_dataset

plt.style.use("custom")
plt.ion()
plt.rcParams["figure.figsize"] = (20, 11)


def get_ce(
    spectrum,
    pump_wl1,
    pump_wl2,
    prominence=0.1,
    height=-75,
    tolerance_nm=0.25,
    max_peak_min_height=-45,
):
    processed = process_single_dataset(
        spectrum,
        prominence,
        height,
        (pump_wl1, pump_wl2),
        tolerance_nm,
        max_peak_min_height,
    )
    if type(processed) != str:
        ce = processed["differences"][-1]
    else:
        ce = np.nan
    return ce


fig_path = "../figs/BL976_exp"
save_figs = False

# |%%--%%| <suYNaEK25Y|8kF6fIfBw5>
# Section for pump wavelength sweep
def process_pwl_sweep_dataset(
    dataset, prominence=0.1, height=-75, tolerance_nm=0.25, max_peak_min_height=-35
):
    if len(np.atleast_1d(dataset["Pump1:"])) > 1:
        wl_varied = dataset["Pump1:"]
        p2 = dataset["Pump2"]
    else:
        p1 = dataset["Pump1:"]
        wl_varied = dataset["Pump2"]
    wavelengths = dataset["Wavelengths"]
    powers = dataset["Powers"]
    ce = np.zeros(np.shape(wl_varied))
    for i in range(len(wl_varied)):
        if "p1" in locals():
            pump1_wl = p1
            pump2_wl = wl_varied[i]
        else:
            pump1_wl = wl_varied[i]
            pump2_wl = p2
        spectrum = np.vstack((wavelengths, powers[i])).T
        ce[i] = get_ce(
            spectrum,
            pump1_wl,
            pump2_wl,
            prominence,
            height,
            tolerance_nm,
            max_peak_min_height,
        )
    return wl_varied, ce


data_folder = "../data/BL976_signal"
file_name = "short_pwl_sweep_no_polopt.pkl"
data_path = os.path.join(data_folder, file_name)
with open(data_path, "rb") as f:
    dataset_short_pwl = pickle.load(f)
file_name = "long_pwl_sweep_no_polopt.pkl"
data_path = os.path.join(data_folder, file_name)
with open(data_path, "rb") as f:
    dataset_long_pwl = pickle.load(f)
pump_wavelengths_short, ce_short = process_pwl_sweep_dataset(dataset_short_pwl)
pump_wavelengths_long, ce_long = process_pwl_sweep_dataset(dataset_long_pwl)
pump_spec_file_name = "pump_specs.pkl"
pump_spec_path = os.path.join(data_folder, pump_spec_file_name)
with open(pump_spec_path, "rb") as f:
    pump_specs = pickle.load(f)
num_pump_specs = np.shape(pump_specs["Powers"])[0]
max_pump_locs = np.zeros(num_pump_specs)
max_pump_vals = np.zeros(num_pump_specs)
for i in range(num_pump_specs):
    max_pump = np.argmax(pump_specs["Powers"][i, :])
    max_pump_vals[i] = pump_specs["Powers"][i, max_pump]
    max_pump_locs[i] = pump_specs["Wavelengths"][max_pump, i]
fig, ax = plt.subplots()
ax.plot(pump_wavelengths_short, -ce_short, label="Short wavelength")
ax.plot(pump_wavelengths_long, -ce_long, label="Long wavelength")
ax.set_xlabel("Pump wavelength (nm)")
ax.set_ylabel("Conversion Efficiency (dB)")
ax.legend(title="Pump varied")
if save_figs:
    fig.savefig(os.path.join(fig_path, "pwl_sweep.pdf"), bbox_inches="tight")

pump_short_shift_center = np.argmax(-ce_short)
pump_long_shift_center = np.argmax(-ce_long)
fig, ax = plt.subplots()
ax.plot(
    pump_wavelengths_short - pump_wavelengths_short[pump_short_shift_center],
    -ce_short,
    label="Short wavelength",
)
ax.plot(
    pump_wavelengths_long - pump_wavelengths_long[pump_long_shift_center],
    -ce_long,
    label="Long wavelength",
)
ax.set_xlabel("Pump wavelength shift (nm)")
ax.set_ylabel("Conversion Efficiency (dB)")
ax.legend(title="Pump varied")
# ax.set_title("Comparison of CE for shifting the two different pumps")
if save_figs:
    fig.savefig(os.path.join(fig_path, "pwl_sweep_centered_at_max.pdf"), bbox_inches="tight")
# |%%--%%| <oembFF5EZT|8gTx21pXQl>
# Section for polarization check
def extract_numbers_from_filename(f):
    numbers = re.findall(r"\d+\.\d+", f)
    if numbers:
        return [float(num) for num in numbers]
    else:
        return []


def get_first_number_from_filename(f):
    numbers = extract_numbers_from_filename(f)
    if numbers:
        return numbers[0]
    else:
        return float("inf")


def sorting_key_by_min_max(filename):
    match = re.search(r"pol_(min|max)_pump_sep-([0-9.]+)=.pkl", filename)
    if match is not None:
        return (float(match.group(2)), 0 if match.group(1) == "max" else 1)
    else:
        return (0, 0)


file_dir = "../data/BL976_signal/pol_check_w_sig/"
file_dir = "../data/BL976_signal/pol_check_sig_opt_for_max_power/"
file_names = glob.glob(file_dir + "*.pkl")

file_names.sort(key=get_first_number_from_filename, reverse=True)
file_names.sort(key=sorting_key_by_min_max)
pump_separations = [
    get_first_number_from_filename(file_name) for file_name in file_names
]


def get_sig_idler_wl_idxs_for_plotting(wl_ax, pow, p1_wl, p2_wl):
    spectrum = np.vstack((wl_ax, pow)).T
    prominence = 0.1
    height = -75
    tolerance_nm = 0.25
    max_peak_min_height = -35
    processed = process_single_dataset(
        spectrum,
        prominence,
        height,
        (p1_wl, p2_wl),
        tolerance_nm,
        max_peak_min_height,
    )
    sig_wl = processed["peak_positions"][0]
    idler_wl = processed["peak_positions"][1]
    sig_wl_idx = np.where(
        np.logical_and(
            spectrum[:, 0] > sig_wl - 0.25,
            spectrum[:, 0] < sig_wl + 0.25,
        )
    )[0]
    # indices within 0.5 nm of the mean idler wavelength
    idler_wl_idx = np.where(
        np.logical_and(
            spectrum[:, 0] > idler_wl - 0.25,
            spectrum[:, 0] < idler_wl + 0.25,
        )
    )[0]
    return sig_wl_idx, idler_wl_idx


def process_single_dataset_bl976(data_path):
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)

    pump1_wl = dataset["Pump1:"]
    pump2_wl = dataset["Pump2"]
    wavelengths = dataset["Wavelengths"]
    powers = dataset["Powers"]
    num_specs = np.shape(powers)[0]
    mean_power = np.mean(powers, axis=0)
    std_power = np.std(powers, axis=0)
    spectrum = np.vstack((wavelengths, mean_power)).T
    ce_mean = get_ce(spectrum, pump1_wl, pump2_wl)
    sig_wl_idxs, idler_wl_idxs = get_sig_idler_wl_idxs_for_plotting(
        wavelengths, mean_power, pump1_wl, pump2_wl
    )
    return (
        mean_power,
        std_power,
        ce_mean,
        sig_wl_idxs,
        idler_wl_idxs,
        wavelengths,
        pump1_wl,
        pump2_wl,
    )


(
    mean_power,
    std_power,
    ce_mean,
    sig_wl_idxs_list,
    idler_wl_idxs_list,
    wl_axis,
    p1_wls,
    p2_wls,
) = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)
for file_name in file_names:
    (
        mean_pow,
        std_pow,
        ce,
        sig_wl_idx,
        idler_wl_idx,
        wl_ax,
        pump1_wl,
        pump2_wl,
    ) = process_single_dataset_bl976(file_name)
    mean_power.append(mean_pow)
    std_power.append(std_pow)
    ce_mean.append(ce)
    sig_wl_idxs_list.append(sig_wl_idx)
    idler_wl_idxs_list.append(idler_wl_idx)
    wl_axis.append(wl_ax)
    p1_wls.append(pump1_wl)
    p2_wls.append(pump2_wl)
# |%%--%%| <VmASHlGUWD|f6AO8IBsQs>
ce_diffs_vs_pump_sep = [
    ce_mean[i * 2] - ce_mean[i * 2 + 1] for i in range(len(ce_mean) // 2)
]
fig, ax = plt.subplots()
ax.plot(np.unique(pump_separations), np.abs(ce_diffs_vs_pump_sep))
ax.set_xlabel("Pump separation (nm)")
ax.set_ylabel(r"$\mathrm{CE}_{\mathrm{max}}-\mathrm{CE}_{\mathrm{min}}$(dB)")
ax.set_title("CE difference vs pump separation")
if save_figs:
    plt.savefig(os.path.join(fig_path, "ce_diff_vs_pump_sep.pdf"), bbox_inches="tight")
# |%%--%%| <v4YivyTsC1|uPQS3fKlA8>
def plot_polarization_data(wavelengths, pol_data, sig_wl_idxs, idler_wl_idxs, title_args):
    fig, ax = plt.subplots()

    for data in pol_data:
        mean_pol, std_pol, label = data
        ax.plot(wavelengths, mean_pol, label=f"{label}")
        ax.fill_between(
            wavelengths,
            mean_pol - std_pol,
            mean_pol + std_pol,
            alpha=0.5,
            label=f"{label} std",
        )

    sig_start, sig_end = sig_wl_idxs[0], sig_wl_idxs[-1]
    idler_start, idler_end = idler_wl_idxs[0], idler_wl_idxs[-1]

    axins1, axins2 = [ax.inset_axes([0.3, 0.6, 0.3, 0.3]), ax.inset_axes([0.7, 0.6, 0.3, 0.3])]
    insets = [axins1, axins2]
    idx_ranges = [(sig_start, sig_end), (idler_start, idler_end)]

    for axins, idx_range in zip(insets, idx_ranges):
        for data in pol_data:
            mean_pol, std_pol, label = data
            axins.plot(wavelengths[idx_range[0]:idx_range[1]], mean_pol[idx_range[0]:idx_range[1]])
            axins.fill_between(
                wavelengths[idx_range[0]:idx_range[1]],
                mean_pol[idx_range[0]:idx_range[1]] - std_pol[idx_range[0]:idx_range[1]],
                mean_pol[idx_range[0]:idx_range[1]] + std_pol[idx_range[0]:idx_range[1]],
                alpha=0.3,
            )
        axins.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # Set y-limits for axins1 based on y-limits of axins2
    lower_idler_lim, upper_idler_lim = axins2.get_ylim()
    idler_extinction = upper_idler_lim - lower_idler_lim
    upper_sig_lim = axins1.get_ylim()[1]
    axins1.set_ylim(upper_sig_lim - idler_extinction, upper_sig_lim)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Power (dBm)")
    ax.legend(loc="lower left")
    pump1_wl, pump2_wl, ce_diff = title_args
    ax.set_title(
        rf"50\% duty cycle, 10 spectra, $\lambda_p,\lambda_q$={pump1_wl:.2f}, {pump2_wl:.2f} nm, CE diff = {ce_diff:.2f} dB"
    )
    plt.show()

fig_path_extention = "pol_check"
if not os.path.exists(fig_path + "//" + fig_path_extention):
    os.makedirs(fig_path + "//" + fig_path_extention)
for pump_sep_num in range(0, 5):
    wavelengths = wl_axis[pump_sep_num * 2]
    mean_max_pol = mean_power[pump_sep_num * 2]
    mean_min_pol = mean_power[pump_sep_num * 2 + 1]
    std_max_pol = std_power[pump_sep_num * 2]
    std_min_pol = std_power[pump_sep_num * 2 + 1]
    max_min_data = [
        (mean_max_pol, std_max_pol, "Max"),
        (mean_min_pol, std_min_pol, "Min"),
    ]
    sig_wl_idxs = sig_wl_idxs_list[pump_sep_num * 2]
    idler_wl_idxs = idler_wl_idxs_list[pump_sep_num * 2]
    pump1_wl = p1_wls[pump_sep_num * 2]
    pump2_wl = p2_wls[pump_sep_num * 2]
    ce_mean_max_pol = ce_mean[pump_sep_num * 2]
    ce_mean_min_pol = ce_mean[pump_sep_num * 2 + 1]
    ce_diff = ce_mean_max_pol - ce_mean_min_pol
    title_args = [pump1_wl, pump2_wl, ce_diff]
    plot_polarization_data(wavelengths, max_min_data, sig_wl_idxs, idler_wl_idxs, title_args)
    if save_figs:
        plt.savefig(f"{fig_path}/{fig_path_extention}/{np.unique(pump_separations)[pump_sep_num]}nm.pdf", bbox_inches="tight")
#|%%--%%| <uPQS3fKlA8|AthEqp67rK>
# Pump spectrum sweeps
file_name = "pump_specs.pkl"
file_path = os.path.join(data_folder, file_name)
with open(file_path, "rb") as f:
    pump_specs = pickle.load(f)
powers = pump_specs["Powers"]
center_wls = pump_specs["Pump2"]
wavelengths = pump_specs["Wavelengths"]
max_powers = np.max(powers, axis=1)
max_power_idxs = np.argmax(powers, axis=1)
fig, ax = plt.subplots()
ax.plot(center_wls, max_powers - np.max(max_powers) + 10, label="Pump power + 10 dB")
ax.plot(center_wls, -ce_short - np.max(-ce_short), label="Idler power")
ax.set_xlabel("Pump wavelength (nm)")
ax.set_ylabel("Normalized power (dB)")
ax.legend()
if save_figs:
    plt.savefig(f"{fig_path}/pwl_sweep_w_pump_power.pdf", bbox_inches="tight")
