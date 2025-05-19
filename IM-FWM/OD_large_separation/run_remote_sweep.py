import pickle

import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys
import importlib


# sys.path.append(r"C:\Users\FTNK-FOD\Desktop\Thjalfe\InstrumentControl\LabInstruments")
# sys.path.append(r"C:\Users\FTNK-FOD\Desktop\Thjalfe\remote_control")
# unwanted_path = "U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl"
# if unwanted_path in sys.path:
#     sys.path.remove(unwanted_path)  # hack because i dont know why that path is there
# from remote_control_toptica_w_flask import DLCControlClient
# from osa_control import OSA
# from ipg_edfa import IPGEDFA
# from laser_control import AndoLaser, TiSapphire, PhotoneticsLaser, Laser
# from verdi_laser import VerdiLaser
# from picoscope2000 import PicoScope2000a
# from tektronix_oscilloscope import TektronixOscilloscope
# from misc import PM

# from pol_cons import ThorlabsMPC320, optimize_multiple_pol_cons
from remote_control_toptica_w_flask import DLCControlClient
from clients.osa_clients import OSAClient
from clients.laser_clients import AndoLaserClient, TiSapphireClient, VerdiLaserClient
from devices.osa_control import OSA

# importlib.reload(sys.modules["misc"])

plt.ioff()
# |%%--%%| <Ngf3b6rjw6|iHZwPozQu7>
GPIB_val = 0
pump1_start = 1581
base_url = "http://10.51.33.243:5000"
ando_pow_start = 8
pump1_laser = AndoLaserClient(
    base_url, "ando_laser_2", target_wavelength=pump1_start, power=ando_pow_start
)
pump1_laser.wavelength = 1580
pump1_laser.linewidth = 1
time.sleep(0.1)
pump1_laser.enable()
tisa = TiSapphireClient(base_url, "tisa")
verdi = VerdiLaserClient(base_url, "verdi")
verdi.power = 10
verdi.shutter = 1
remote_ip = "10.51.37.182"
cert_path = (
    r"/home/thjalfe/remote_test/ssl/server.crt"  # Path to the server certificate
)
toptica = DLCControlClient(remote_ip, cert_path)
toptica.open_connection()
osa2 = OSAClient(base_url, "osa_2", (1730, 1740))
# osa = OSA(
#     960,
#     990,
#     resolution=0.1,
#     GPIB_address=19,
#     GPIB_bus=1,
#     sweeptype="RPT",
# )
# osa2 = OSA(
#     1500,
#     1550,
#     resolution=0.1,
#     GPIB_address=17,
#     GPIB_bus=1,
#     sweeptype="RPT",
# )
# scope = TektronixOscilloscope()
time.sleep(0.5)


# |%%--%%| <UZh5poQs5w|oa12J9qdEe>
def increment_pump_wl(pump_laser, increment):
    cur_wl = pump_laser.wavelength
    pump_laser.wavelength = cur_wl + increment
    print(f"{pump_laser.wavelength:.3f}")


simulated_phase_matching_loc = "./simulation_res/d-dbeta-dr_w-phase-matching_large-sep_telecom-pump-wls=1530.0-1610.0nm.pkl"
with open(simulated_phase_matching_loc, "rb") as f:
    sim_data = pickle.load(f)


def calculate_idler_wl(
    p1: np.ndarray | float, s: np.ndarray | float, p2: np.ndarray | float
):
    return np.round(1 / (-1 / p1 + 1 / s + 1 / p2), 2)


def calculate_signal_wl(
    p1: np.ndarray | float, p2: np.ndarray | float, i: np.ndarray | float
):
    return np.round(1 / (1 / p1 - 1 / p2 + 1 / i), 2)


def rename_sim_data_dict(sim_data: dict):
    """
    Rename the keys of the dictionary since they were named with a different FWM config in mind
    """
    short_pump_wls = sim_data["short_input_wls"]
    short_input_wls = sim_data["short_pump_wls"]
    short_pump_wls_spline = sim_data["short_input_wls_spline"]
    sim_data["short_input_wls"] = short_input_wls
    sim_data["short_pump_wls"] = short_pump_wls
    sim_data["short_pump_wls_spline"] = short_pump_wls_spline
    return sim_data


def plot_spectra(
    spectra: np.ndarray, idxs_to_consider: np.ndarray | None = None, title=None
):
    fig, ax = plt.subplots()
    if idxs_to_consider is None:
        idxs_to_consider = np.arange(len(spectra))
    for idx, spectrum in enumerate(spectra):
        if idx in idxs_to_consider:
            spectrum[1][spectrum[1] < -80] = np.nan
            ax.plot(spectrum[0], spectrum[1])
    if title is not None:
        ax.set_title(title)
    return fig, ax


def plot_spectra_from_dict(data: dict):
    for tisa_wl in data["tisa_wl_array"]:
        plot_spectra(data["spectra"][tisa_wl], title=f"TISA WL: {tisa_wl} nm")


def rolling_average(a, n=4):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    rolling_avg = ret[n - 1 :] / n
    return np.concatenate(([np.nan] * (n - 1), rolling_avg))


def find_optimum_one_spectrum_series(spectra: np.ndarray, toptica_wl_array: np.ndarray):
    spectra[np.isnan(spectra)] = -200
    spectra_rolled = np.array(
        [rolling_average(spectrum[1], n=4) for spectrum in spectra]
    )
    max_toptica_wl_idx, max_idler_idx = np.unravel_index(
        np.nanargmax(spectra_rolled), np.shape(spectra_rolled)
    )
    max_toptica_wl = toptica_wl_array[max_toptica_wl_idx]
    max_idler_wl = spectra[max_toptica_wl_idx][0][max_idler_idx]
    max_idler_pow = spectra[max_toptica_wl_idx][1][max_idler_idx]
    return (
        max_toptica_wl,
        max_idler_wl,
        max_idler_pow,
        max_toptica_wl_idx,
        max_idler_idx,
    )


def calc_slope(max_toptica_wls: np.ndarray, tisa_wl_array: np.ndarray):
    slope = (max_toptica_wls[-1] - max_toptica_wls[0]) / (
        tisa_wl_array[-1] - tisa_wl_array[0]
    )
    return slope


def calculate_est_time_left(time_spent, cur_iter, total_iter):
    return time_spent / cur_iter * (total_iter - cur_iter)


sim_data = rename_sim_data_dict(sim_data)
sim_data_telecom_idx = -3
min_S_loc_from_sim = np.argmin(np.abs(sim_data["d_dbeta_dr"][sim_data_telecom_idx]))
tisa_wl_at_min_S = np.round(
    sim_data["short_pump_wls"][sim_data_telecom_idx][min_S_loc_from_sim] * 1000, 1
)
toptica_wl_at_min_S = np.round(
    sim_data["short_input_wls"][sim_data_telecom_idx][min_S_loc_from_sim] * 1000, 2
)
# |%%--%%| <oa12J9qdEe|m61N7iH1R6>
# %% First sweep with const tisa wl
tisa_wl = tisa_wl_at_min_S
toptica_wl = toptica_wl_at_min_S
telecom_wl = sim_data["telecom_pump_wls"][sim_data_telecom_idx] * 1000
pump1_laser.wavelength = telecom_wl
toptica_sweep_around_expected_phasematch = 5
toptica_stepsize = 0.2
toptica_wl_array = np.arange(
    toptica_wl - toptica_sweep_around_expected_phasematch,
    toptica_wl + toptica_sweep_around_expected_phasematch * 3 + toptica_stepsize,
    toptica_stepsize,
)
toptica_wl_array = np.round(toptica_wl_array, 2)
nm_around_idler = 2.5
spectra = []
tisa.wavelength = tisa_wl
osa2.resolution = 0.5
osa2.sensitivity = "SHI1"
osa2.stop_sweep()
osa2.sweeptype = "SGL"
for toptica_wl in toptica_wl_array:
    toptica.set_wavelength(toptica_wl)
    idler_wl = calculate_idler_wl(tisa_wl, toptica_wl, telecom_wl)
    osa2.span = idler_wl - nm_around_idler, idler_wl + nm_around_idler
    osa2.sweep()
    wavelengths = osa2.wavelengths
    powers = osa2.powers
    spectrum = np.vstack((wavelengths, powers))
    spectra.append(spectrum)
    print(f"Done with measurement number {len(spectra)} out of {len(toptica_wl_array)}")

# |%%--%%| <m61N7iH1R6|GBoAccbq1G>

# %% double loop sweeping both lasers
from scipy.stats import linregress

telecom_wl = sim_data["telecom_pump_wls"][sim_data_telecom_idx] * 1000
telecom_wl = 1580
pump1_laser.wavelength = telecom_wl
tisa_wl_sweep_around_min_S = 5
tisa_stepsize = 1
tisa_wl_array = np.arange(
    tisa_wl_at_min_S - tisa_wl_sweep_around_min_S,
    tisa_wl_at_min_S + tisa_wl_sweep_around_min_S + tisa_stepsize,
    tisa_stepsize,
)
slope = 0.22
start_tisa_wl = 930.9
toptica_wl_at_start_tisa = 966.06
# make linear relation from slope, start_tisa_wl, toptica_wl_at_start_tisa
toptica_wl_fit = np.poly1d([slope, toptica_wl_at_start_tisa - slope * start_tisa_wl])
tisa_wl_array = np.arange(930.9, 941.9, 1)
toptica_wls_at_sim_phase_matching = [
    sim_data["short_input_wls"][sim_data_telecom_idx][
        np.argmin(np.abs(sim_data["short_pump_wls"] - wl / 1000))
    ]
    for wl in tisa_wl_array
]

data_dir = f"./data/telecom-wl=1580.0nm/"
data_file = f"tisa-wl=915.9-929.9nm_toptica_fine_sweep_1700+nm-idler.pkl"
with open(os.path.join(data_dir, data_file), "rb") as f:
    old_data_for_fit = pickle.load(f)


fit = linregress(
    old_data_for_fit["tisa_wl_array"][:],
    old_data_for_fit["optimum_vals_dict"]["toptica_wls"][:],
)
toptica_wl_fit = np.poly1d([fit.slope, fit.intercept])
print(fit.slope, toptica_wl_fit(tisa_wl_array[0]))
# |%%--%%| <GBoAccbq1G|iGat5SdZ6L>
# %%
pump1_laser.wavelength = telecom_wl
print(telecom_wl)
nm_around_idler = 2
spectra = {tisa_wl: [] for tisa_wl in tisa_wl_array}
data = {}
data["spectra"] = spectra
data["tisa_wl_array"] = tisa_wl_array
data["toptica_wls"] = []
data["telecom_wl"] = telecom_wl
data["optimum_vals_dict"] = {
    "toptica_wls": [],
    "idler_wls": [],
    "idler_powers": [],
    "toptica_wl_idxs": [],
    "idler_idxs": [],
    "optimum_found": [],
}
if tisa_wl_array[0] < tisa_wl_array[-1]:
    reverse_data = True
else:
    reverse_data = False


def reverse_dataset(data: dict):
    import copy

    data_copy = copy.deepcopy(data)
    data_copy["tisa_wl_array"] = data_copy["tisa_wl_array"][::-1]
    spectra_keys = list(data_copy["spectra"].keys())
    spectra_keys_rev = spectra_keys[::-1]
    dummy_spectra = {}
    for key_rev in spectra_keys_rev:
        dummy_spectra[key_rev] = data_copy["spectra"][key_rev]
    data_copy["spectra"] = dummy_spectra
    data_copy["optimum_vals_dict"]["toptica_wls"] = data_copy["optimum_vals_dict"][
        "toptica_wls"
    ][::-1]
    data_copy["optimum_vals_dict"]["idler_wls"] = data_copy["optimum_vals_dict"][
        "idler_wls"
    ][::-1]
    data_copy["optimum_vals_dict"]["idler_powers"] = data_copy["optimum_vals_dict"][
        "idler_powers"
    ][::-1]
    data_copy["optimum_vals_dict"]["toptica_wl_idxs"] = data_copy["optimum_vals_dict"][
        "toptica_wl_idxs"
    ][::-1]
    data_copy["optimum_vals_dict"]["idler_idxs"] = data_copy["optimum_vals_dict"][
        "idler_idxs"
    ][::-1]
    data_copy["optimum_vals_dict"]["optimum_found"] = data_copy["optimum_vals_dict"][
        "optimum_found"
    ][::-1]
    return data_copy


osa2.resolution = 0.1
osa2.sensitivity = "SHI1"
osa2.samples = 251
osa2.level_scale = 10
osa2.level = -10
osa2.stop_sweep()
osa2.sweeptype = "SGL"
toptica_sweep_around_expected_phasematch = 0.1
toptica_stepsize = 0.01
toptica.enable_emission()
t0 = time.time()
data_dir = f"./data/telecom-wl={telecom_wl:.1f}nm"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
old_file = None
max_retries = 0
wl_array_for_slope = [[], []]
override_exp_toptica_wl = False


def sweep_toptica_wl(
    toptica: DLCControlClient,
    osa: OSA | OSAClient,
    idler_wl,
    nm_around_idler,
    toptica_wl_array,
    tisa_idx,
    tisa_wl_arr,
    tot_toptica_wls_swept,
    extra_sweeps,
):
    spectra = []
    for toptica_idx, toptica_wl in enumerate(toptica_wl_array):
        toptica.set_wavelength(toptica_wl)
        osa.span = idler_wl - nm_around_idler, idler_wl + nm_around_idler
        osa.sweep()
        wavelengths = osa.wavelengths
        powers = osa.powers
        spectrum = np.vstack((wavelengths, powers))
        spectra.append(spectrum)
        tot_toptica_wls_swept += 1
        print(
            f"Done with measurement number {toptica_idx+1} out of {len(toptica_wl_array)} for tisa measurment number {tisa_idx+1} out of {len(tisa_wl_arr)}"
        )
        print(f"Time elapsed: {time.time()-t0:.1f} s")
        print(
            f"Estimated time left: {calculate_est_time_left(time.time()-t0, tot_toptica_wls_swept, extra_sweeps + len(toptica_wl_array) * len(tisa_wl_arr)):.1f} s, which is at {time.ctime(time.time()+calculate_est_time_left(time.time()-t0, tot_toptica_wls_swept, len(toptica_wl_array) * len(tisa_wl_arr) + extra_sweeps)):s}"
        )
    return spectra, tot_toptica_wls_swept


tot_toptica_wls_swept = 0
extra_sweeps = 0
tmp_plot_loc = f"./tmp_plots/telecom-wl={telecom_wl:.1f}nm"
if not os.path.exists(tmp_plot_loc):
    os.makedirs(tmp_plot_loc)
for tisa_idx, tisa_wl in enumerate(tisa_wl_array):
    tisa.wavelength = tisa_wl
    toptica_exp_wl = toptica_wl_fit(tisa_wl)
    toptica_wl_array = np.arange(
        toptica_exp_wl - toptica_sweep_around_expected_phasematch,
        toptica_exp_wl + toptica_sweep_around_expected_phasematch + toptica_stepsize,
        toptica_stepsize,
    )
    toptica_wl_array = np.round(toptica_wl_array, 3)
    data["toptica_wls"].append(toptica_wl_array)
    spectra, tot_toptica_wls_swept = sweep_toptica_wl(
        toptica,
        osa2,
        calculate_idler_wl(tisa_wl, toptica_exp_wl, telecom_wl),
        nm_around_idler,
        toptica_wl_array,
        tisa_idx,
        tisa_wl_array,
        tot_toptica_wls_swept,
        extra_sweeps,
    )
    spectra = np.array(spectra)
    fig, ax = plot_spectra(spectra, title=f"TISA WL: {tisa_wl} nm")
    figname = f"tisa-wl={tisa_wl:.1f}nm"
    if figname in os.listdir(tmp_plot_loc):
        # append number to filename, increasing by 1 if file already exists
        counter = 1
        while figname + f"_{counter}" in os.listdir(tmp_plot_loc):
            counter += 1
        figname += f"_{counter}"
    fig.savefig(os.path.join(tmp_plot_loc, f"{figname}.pdf"), bbox_inches="tight")

    max_toptica_wl, max_idler_wl, max_idler_pow, max_toptica_wl_idx, max_idler_idx = (
        find_optimum_one_spectrum_series(spectra, toptica_wl_array)
    )
    if max_idler_pow < -70.5 and max_retries > 0:
        print("Retrying")
        # try again with more steps up to 3 times
        max_retry_counter = 0
        while max_idler_pow < -70.5 and max_retry_counter < max_retries:
            extra_sweeps += len(toptica_wl_array)
            spectra, tot_toptica_wls_swept = sweep_toptica_wl(
                toptica,
                osa2,
                calculate_idler_wl(tisa_wl, toptica_exp_wl, telecom_wl),
                nm_around_idler * 1.5,
                toptica_wl_array,
                tisa_idx,
                tisa_wl_array,
                tot_toptica_wls_swept,
                extra_sweeps,
            )
            spectra = np.array(spectra)
            (
                max_toptica_wl,
                max_idler_wl,
                max_idler_pow,
                max_toptica_wl_idx,
                max_idler_idx,
            ) = find_optimum_one_spectrum_series(spectra, toptica_wl_array)
            max_retry_counter += 1
        if max_idler_pow < -70.5:
            print("Failed to find idler peak")
            data["optimum_vals_dict"]["optimum_found"].append(False)
        else:
            data["optimum_vals_dict"]["optimum_found"].append(True)
    else:
        data["optimum_vals_dict"]["optimum_found"].append(True)
    data["spectra"][tisa_wl] = spectra
    data["optimum_vals_dict"]["toptica_wls"].append(max_toptica_wl)
    data["optimum_vals_dict"]["idler_wls"].append(max_idler_wl)
    data["optimum_vals_dict"]["idler_powers"].append(max_idler_pow)
    data["optimum_vals_dict"]["toptica_wl_idxs"].append(max_toptica_wl_idx)
    data["optimum_vals_dict"]["idler_idxs"].append(max_idler_idx)
    if data["optimum_vals_dict"]["optimum_found"][-1] is not False:
        wl_array_for_slope[0].append(tisa_wl)
        wl_array_for_slope[1].append(max_toptica_wl)

    if len(wl_array_for_slope[0]) > 3:
        slope = calc_slope(
            np.array(wl_array_for_slope[1]), np.array(wl_array_for_slope[0])
        )
        toptica_wl_fit = np.poly1d([slope, max_toptica_wl - slope * tisa_wl])
    if reverse_data:
        save_data = reverse_dataset(data)
    else:
        save_data = data.copy()
    print(
        f"Done with tisa meas number {tisa_idx+1} out of {len(tisa_wl_array)} in {time.time()-t0:.1f} s"
    )
    filename = f"tisa-wl={tisa_wl_array[0]:.1f}-{tisa_wl:.1f}nm_toptica_fine_sweep_1700+nm-idler.pkl"
    with open(os.path.join(data_dir, filename), "wb") as f:
        pickle.dump(save_data, f)
    # delete prev old file
    if old_file is not None:
        os.remove(old_file)
    old_file = os.path.join(data_dir, filename)

osa.samples = 0
toptica.disable_emission()
# |%%--%%| <iGat5SdZ6L|O8XG7YI09D>
# %%


def plot_relevant_results(measured_data: dict):
    fig, ax = plt.subplots()
    ax.plot(
        measured_data["tisa_wl_array"],
        measured_data["optimum_vals_dict"]["toptica_wls"],
    )
    ax.set_xlabel("Tisa wl [nm]")
    ax.set_ylabel("Toptica wl [nm]")
    ax.set_title(
        f"Telecom wl: {measured_data['telecom_wl']:.1f} nm, toptica wl maximizing idler"
    )
    fig, ax = plt.subplots()
    ax.plot(
        measured_data["tisa_wl_array"],
        measured_data["optimum_vals_dict"]["idler_powers"],
    )
    ax.set_xlabel("Tisa wl [nm]")
    ax.set_ylabel("Idler power [dBm]")
    ax.set_title(f"Telecom wl: {measured_data['telecom_wl']:.1f} nm, at max toptica wl")
    fig, ax = plt.subplots()
    ax.plot(
        measured_data["tisa_wl_array"],
        measured_data["optimum_vals_dict"]["idler_wls"],
    )
    ax.set_xlabel("Tisa wl [nm]")
    ax.set_ylabel("Idler wl [nm]")
    ax.set_title(f"Telecom wl: {measured_data['telecom_wl']:.1f} nm, at max toptica wl")


data_loc = "./data/telecom-wl=1580.0nm/tisa-wl=915.9-929.9nm_toptica_fine_sweep_1700+nm-idler.pkl"
with open(data_loc, "rb") as f:
    measured_data = pickle.load(f)
plot_relevant_results(measured_data)
# |%%--%%| <O8XG7YI09D|aj0j0uFRcR>
# %%
meas_data_loc1 = "./data/telecom-wl=1590.0nm/tisa-wl=945.9-959.9nm_toptica_fine_sweep_1700+nm-idler.pkl"
meas_data_loc2 = "./data/telecom-wl=1590.0nm/tisa-wl=944.9-873.9nm_toptica_fine_sweep_1700+nm-idler.pkl"
with open(meas_data_loc1, "rb") as f:
    measured_data1 = pickle.load(f)
with open(meas_data_loc2, "rb") as f:
    measured_data2 = pickle.load(f)


def merge_two_datasets(data1: dict, data2: dict):
    data_merged = {}
    if np.mean(data2["tisa_wl_array"]) > np.mean(data1["tisa_wl_array"]):
        # orders by tisa wl
        data1, data2 = data2, data1
    for key, val in data1.items():
        # check if val is dict
        if key == "spectra":
            # delete all empty entries for data2
            dummy_data1_specs = {}
            dummy_data2_specs = {}
            nonempty_specs_1_count = 0
            nonempty_specs_2_count = 0
            for key2, val2 in val.items():
                if len(val2) > 0:
                    dummy_data1_specs[key2] = val2
                    nonempty_specs_1_count += 1
            for key2, val2 in data2[key].items():
                if len(val2) > 0:
                    dummy_data2_specs[key2] = val2
                    nonempty_specs_2_count += 1
            data_merged[key] = {**dummy_data1_specs, **dummy_data2_specs}
        elif key == "optimum_vals_dict":
            data_merged[key] = {}
            for key2, val2 in val.items():
                data_merged[key][key2] = val2 + data2[key][key2]
        elif key == "toptica_wls":
            data_merged[key] = val + data2[key]
        elif key == "tisa_wl_array":
            data_merged[key] = np.concatenate(
                (val[:nonempty_specs_1_count], data2[key][:nonempty_specs_2_count])
            )
        elif key == "telecom_wl":
            data_merged[key] = val
    return data_merged


measured_data = merge_two_datasets(measured_data1, measured_data2)
plot_relevant_results(measured_data)
with open(
    f"./data/telecom-wl={measured_data['telecom_wl']:.1f}nm/tisa-wl={measured_data['tisa_wl_array'][0]:.1f}-{measured_data['tisa_wl_array'][-1]:.1f}nm_toptica_fine_sweep_1700+nm-idler.pkl",
    "wb",
) as f:
    pickle.dump(measured_data, f)
# |%%--%%| <aj0j0uFRcR|Ctor93s2DZ>
# %%
meas_data_loc1 = "./data/telecom-wl=1600.0nm/tisa-wl=934.9-890.9nm_toptica_fine_sweep_1700+nm-idler.pkl"
meas_data_loc2 = "./data/telecom-wl=1590.0nm/tisa-wl=959.9-873.9nm_toptica_fine_sweep_1700+nm-idler.pkl"
with open(meas_data_loc1, "rb") as f:
    measured_data1 = pickle.load(f)

with open(meas_data_loc2, "rb") as f:
    measured_data2 = pickle.load(f)
plot_relevant_results(measured_data1)
plot_relevant_results(measured_data2)
tisa_wls_dataset1 = measured_data1["tisa_wl_array"]
tisa_wls_dataset2 = measured_data2["tisa_wl_array"]
tisa_wls_dataset2_same_wl_idxs = [
    np.where(tisa_wls_dataset2 == tisa_wl)[0][0] for tisa_wl in tisa_wls_dataset1
]
diff_between_opt_toptica_wls = (
    np.array(measured_data1["optimum_vals_dict"]["toptica_wls"])
    - np.array(measured_data2["optimum_vals_dict"]["toptica_wls"])[
        tisa_wls_dataset2_same_wl_idxs
    ]
)
tisa_wl_for_next_opt = 915.9
tisa_idx_for_next_opt1 = np.where(tisa_wls_dataset1 == tisa_wl_for_next_opt)[0][0]
tisa_idx_for_next_opt2 = np.where(tisa_wls_dataset2 == tisa_wl_for_next_opt)[0][0]
toptica_wl_at_tisa_opt = measured_data2["optimum_vals_dict"]["toptica_wls"][
    tisa_idx_for_next_opt2
]
slope = np.mean(diff_between_opt_toptica_wls)

# |%%--%%| <Ctor93s2DZ|UfFMWLsl9T>
# %% Section for sweeping just the toptica laser more manually
idx = 4
interesting_tisa_wl = 914.9
tisa.wavelength = interesting_tisa_wl
telecom_wl = 1580
telecom_wl_diff_factor = (1590 - telecom_wl) / 10
interesting_toptica_wls = 966.11 - 0.2
pump1_laser.wavelength = telecom_wl
# 914.9, 965.889
# 915.9, 966.11
# 916.9, 966.255
# |%%--%%| <UfFMWLsl9T|tcIzxXk03h>
# %%
# interesting_toptica_wls = 948.97


def increment_toptica(toptica_laser: DLCControlClient, increment: float):
    cur_wl = np.round(toptica_laser.get_wavelength()["wavelength"], 4)
    toptica_laser.set_wavelength(cur_wl + increment)
    print(f"{toptica_laser.get_wavelength()['wavelength']:.4f}")


increment = 0.02
osa2.sweeptype = "SGL"
osa2.stop_sweep()
toptica_wl_array = np.arange(
    interesting_toptica_wls - 0.2,
    interesting_toptica_wls + 0.2 + increment,
    increment,
)
toptica.enable_emission()
old_res = osa2.resolution
osa2.level_scale = 10
osa2.level = -10
osa2.sensitivity = "SHI1"
osa2.samples = 251
osa2.resolution = 0.1
spectra = []
t0 = time.time()
for i in range(len(toptica_wl_array)):
    toptica_wl = toptica_wl_array[i]
    toptica.set_wavelength(toptica_wl)
    expected_idler_wl = calculate_idler_wl(interesting_tisa_wl, toptica_wl, telecom_wl)
    osa2.span = expected_idler_wl - 2, expected_idler_wl + 2
    osa2.sweep()
    wavelengths = osa2.wavelengths
    powers = osa2.powers
    spectrum = np.vstack((wavelengths, powers))
    spectra.append(spectrum)
    print(f"Done with measurement number {i+1} out of {len(toptica_wl_array)}")
    print(f"Time elapsed {time.time()-t0:.1f} s")
    print(
        f"Estimated time left: {calculate_est_time_left(time.time()-t0, i+1, len(toptica_wl_array)):.1f} s, which is at {time.ctime(time.time()+calculate_est_time_left(time.time()-t0, i+1, len(toptica_wl_array))):s}"
    )
plot_spectra(spectra, title=f"TISA WL: {interesting_tisa_wl} nm")
toptica.disable_emission()
osa2.resolution = old_res
osa2.samples = 0
spectra = np.array(spectra)
max_toptica_wl, max_idler_wl, max_idler_pow, max_toptica_wl_idx, max_idler_idx = (
    find_optimum_one_spectrum_series(spectra, toptica_wl_array)
)
