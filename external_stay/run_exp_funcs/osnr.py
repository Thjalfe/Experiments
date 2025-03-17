import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from clients.laser_clients import AgilentLaserClient
from clients.osa_clients import OSAClient
from processing.helper_funcs import dBm_to_mW, rolling_average


def window_sidebands(
    spectrum: np.ndarray,
    center_wl: float,
    dist_from_center: float,
    window_size_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.copy(spectrum)
    spectrum[1] = dBm_to_mW(spectrum[1])
    left_idx = np.nanargmin(np.abs(spectrum[0] - (center_wl - dist_from_center)))
    right_idx = np.nanargmin(np.abs(spectrum[0] - (center_wl + dist_from_center)))
    delta_wl = spectrum[0][1] - spectrum[0][0]
    window_size = int(round(window_size_nm / delta_wl))
    left_data = spectrum[1][:left_idx]
    left_avg = rolling_average(left_data, window_size)
    left_x = spectrum[0][window_size - 1 : left_idx]
    right_data = spectrum[1][right_idx:]
    right_avg = rolling_average(right_data, window_size)
    right_x = spectrum[0][right_idx + window_size - 1 :]
    sidebands_only = np.concatenate((left_avg, right_avg))
    center_x = spectrum[0][left_idx:right_idx]
    center_data = spectrum[1][left_idx:right_idx]
    full_window_x = np.concatenate((left_x, center_x, right_x))
    full_window_rolled_sidebands = np.concatenate((left_avg, center_data, right_avg))
    full_spectra_rolled_sidebands = np.vstack(
        (full_window_x, full_window_rolled_sidebands)
    )
    return full_spectra_rolled_sidebands, sidebands_only


def calc_OSNR(power_linear: np.ndarray, sidebands: np.ndarray) -> float:
    noise_power = np.nanmean(sidebands)
    signal_power = np.nanmax(power_linear)
    return 10 * np.log10(signal_power / noise_power)


def osnr_multiple_laser_powers(
    laser_powers: np.ndarray,
    tls: AgilentLaserClient,
    osa: OSAClient,
    spectra_dict: dict,
    dist_from_center: float,
    window_size_nm: float,
    save_data: bool,
    data_dir: str,
    loc_str: str,
) -> dict:
    for power in laser_powers:
        tls.power = power
        time.sleep(0.5)
        osa.sweep()
        wavelengths = osa.wavelengths
        powers = osa.powers
        powers[powers < -80] = -80
        spectrum = np.vstack((wavelengths, powers))

        spectra_dict["spectra"].append(spectrum)
        center_wl = spectrum[0][np.nanargmax(spectrum[1])]
        rolled_spec, sidebands = window_sidebands(
            spectrum, center_wl, dist_from_center, window_size_nm
        )
        osnr = calc_OSNR(dBm_to_mW(spectrum[1]), sidebands)
        spectra_dict["spectra_rolled"].append(rolled_spec)
        spectra_dict["OSNR"].append(osnr)

    if save_data:
        with open(f"{data_dir}/{loc_str}.pkl", "wb") as f:
            pickle.dump(spectra_dict, f)
        fig, ax = plt.subplots()
        for i, trace in enumerate(spectra_dict["spectra"]):
            ax.plot(trace[0], trace[1], label=f"{laser_powers[i]} dBm")
        ax.legend()
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Power [dBm]")
        fig.savefig(f"{data_dir}/{loc_str}_spectra.pdf", bbox_inches="tight")
    return spectra_dict
