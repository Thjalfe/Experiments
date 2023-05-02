from .util_funcs import analyze_data
import numpy as np
import time
import sys
import os

sys.path.append("U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl")
import instrument_class as misc
from OSA_control import OSA


def save_sweep(data_folder, filename, **data):
    import pickle

    # Create the folder if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    # Create a unique filename
    filename = f"{filename}.pkl"
    # Save the data as a .pkl file
    with open(os.path.join(data_folder, filename), "wb") as f:
        pickle.dump(data, f)


def make_pump_power_equal(
    ando1,
    ando2,
    wl1,
    wl2,
    OSA_GPIB_num=[0, 19],
):
    """
    Adjust the power of the two pumps (ando1 and ando2) to make their peak power equal.

    Args:
        ando1 (object): The first pump.
        ando2 (object): The second pump.
        wl1 (float): Wavelength of the first pump.
        wl2 (float): Wavelength of the second pump.
        OSA_GPIB_num (list, optional): GPIB number for OSA. Defaults to [GPIB_val, 19].
        min_height (float, optional): Minimum height for peak detection. Defaults to -80.
    """

    def get_indices(wavelengths, wavelength_range):
        start_index = np.where(wavelengths >= wavelength_range[0])[0][0]
        end_index = np.where(wavelengths <= wavelength_range[1])[0][-1] + 1
        return start_index, end_index

    i = 0
    while True:
        OSA_temp = OSA(
            wl2 - 1,
            wl1 + 1,
            resolution=0.05,
            sensitivity="SMID",
            GPIB_num=OSA_GPIB_num,
        )
        wavelength_range1 = get_indices(OSA_temp.wavelengths, [wl1 - 0.1, wl1 + 0.1])
        wavelength_range2 = get_indices(OSA_temp.wavelengths, [wl2 - 0.1, wl2 + 0.1])
        peak1 = np.max(OSA_temp.powers[wavelength_range1[0] : wavelength_range1[1]])
        peak2 = np.max(OSA_temp.powers[wavelength_range2[0] : wavelength_range2[1]])
        # peaks, properties = find_peaks(OSA_temp.powers, height=min_height)
        # sorted_peaks_indices = np.argsort(properties["peak_heights"])[::-1]
        # largest_peaks_indices = sorted_peaks_indices[:2]
        # largest_peaks = peaks[largest_peaks_indices]
        peak_diff = peak1 - peak2
        print(peak1, peak2)
        i += 1
        if np.abs(peak_diff) <= 0.3:
            break
        elif peak_diff > 0:
            ando1.set_power(ando1.power - np.abs(peak_diff) / 2)
            time.sleep(1)
        elif peak_diff < 0:
            ando2.set_power(ando2.power - np.abs(peak_diff) / 2)
            time.sleep(1)
        # if np.abs(OSA_temp.wavelengths[largest_peaks[0]] - wl2) > 0.05:
        #     ando2.adjust_wavelength(OSA_GPIB_num=OSA_GPIB_num)
        # if np.abs(OSA_temp.wavelengths[largest_peaks[1]] - wl1) > 0.05:
        #     ando1.adjust_wavelength(OSA_GPIB_num=OSA_GPIB_num)


def get_new_OSA_lims(OSA_local, p1wl, p2wl, c=2.998 * 10**8):
    """
    Calculate new OSA limits based on the energy estimation of the idler.

    Args:
        OSA_local (object): OSA object.
        p1wl (float): Wavelength of the first pump.
        p2wl (float): Wavelength of the second pump.
        c (float, optional): Speed of light constant in m/s. Defaults to 2.998 * 10**8.

    Returns:
        np.array: New OSA limits.
    """
    energy_diff = np.abs(c / (p1wl * 10**-9) - c / (p2wl * 10**-9))
    wl_sig = OSA_local.wavelengths[np.argmax(OSA_local.powers)]
    wl_idler = c / (c / (wl_sig * 10**-9) + energy_diff) * 10**9
    OSA_lims = np.array(
        [wl_sig - np.abs(wl_sig - wl_idler) - 1, wl_sig + np.abs(wl_sig - wl_idler) + 1]
    )
    return OSA_lims


def new_sig_start(data_folder, pumpwl_low, pumpwl_high, wl_tot, max_peak_min_height):
    """
    Determine the new signal start position based on the analyzed data.

    Args:
        data_folder (str): The path to the data folder.
        pumpwl_low (float): Lower wavelength limit of the pump.
        pumpwl_high (float): Higher wavelength limit of the pump.
        wl_tot (float): Total wavelength range for the signal.

    Returns:
        float: New signal start position.
    """
    peaks_sorted = analyze_data(
        data_folder,
        pump_wl_pairs=[(pumpwl_low, pumpwl_high)],
        max_peak_min_height=max_peak_min_height,
    )[(pumpwl_low, pumpwl_high)]
    max_peak_idx = list(peaks_sorted)[0]
    if type(max_peak_idx) is not int:
        raise ValueError(
            "No peak found in the data. Maybe the max_peak_min_height is too high?"
        )
    return peaks_sorted[max_peak_idx]["peak_positions"][0] - wl_tot / 2


def run_experiment(
    data_folder,
    ando1,
    ando2,
    TiSa,
    ando1_wl,
    ando2_wl,
    num_sweeps,
    del_wl,
    wl_tot,
    sig_start,
    adjust_laser_wavelengths=True,
    equal_pump_power=False,
    log_pm=False,
    OSA_sens="SHI1",
    OSA_res=0.05,
    over_sampling=2.5,
    OSA_GPIB_num=[0, 19],
    max_peak_min_height=-35,
):
    """
    Run the main loop of the experiment and send an email notification upon completion or error.

    Args:
        data_folder (str): The path to the data folder.
        ando1_wl (list): List of wavelengths for Ando1.
        ando2_wl (list): List of wavelengths for Ando2.
        equal_pump_power (bool): If True, make the pump power equal.
        log_pm (bool): If True, log the power meter data.
        num_sweeps (int): Number of sweeps to perform.
        del_wl (float): Delta wavelength for TiSa.
        wl_tot (float): Total wavelength range for the signal.
        sig_start(float): Start wavelength for TiSa.
        GPIB_val (int, optional): GPIB value for the OSA. Defaults to 19.
    """
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    for i in range(len(ando1_wl)):
        wl1_temp = ando1_wl[i]
        wl2_temp = ando2_wl[i]
        ando1.set_power(8)
        ando2.set_power(8)
        ando1.set_wavelength(wl1_temp)
        ando2.set_wavelength(wl2_temp)
        if adjust_laser_wavelengths:
            ando1.adjust_wavelength(sens="SMID", OSA_GPIB_num=OSA_GPIB_num)
            ando2.adjust_wavelength(OSA_GPIB_num=OSA_GPIB_num)
        if equal_pump_power:
            make_pump_power_equal(ando1, ando2, wl1_temp, wl2_temp)
        OSA_temp = OSA(
            wl2_temp - 1,
            wl1_temp + 1,
            resolution=OSA_res,
            sensitivity=OSA_sens,
            GPIB_num=OSA_GPIB_num,
        )
        OSA_temp.save(f"{data_folder}/pumps{wl2_temp}_{wl1_temp}")
        OSA_temp.set_span(
            sig_start - np.abs(wl1_temp - wl2_temp),
            sig_start + np.abs(wl1_temp - wl2_temp),
        )
        OSA_temp.sweep()
        lims = get_new_OSA_lims(OSA_temp, wl1_temp, wl2_temp)
        temp_sampling = over_sampling * 2 * (lims[1] - lims[0]) / OSA_res
        OSA_temp.set_sample(temp_sampling)
        for j in range(num_sweeps):
            TiSa.delta_wl_nm(del_wl)
            time.sleep(0.5)
            OSA_temp.set_span(lims[0], lims[1])
            OSA_temp.sweep()
            if log_pm:
                save_sweep(
                    data_folder,
                    f"{wl2_temp}_{wl1_temp}_{j}",
                    wavelengths=OSA_temp.wavelengths,
                    powers=OSA_temp.powers,
                    ando_powers=(ando1.power, ando2.power),
                    PM=misc.PM().read(),
                )
            else:
                save_sweep(
                    data_folder,
                    f"{wl2_temp}_{wl1_temp}_{j}",
                    wavelengths=OSA_temp.wavelengths,
                    powers=OSA_temp.powers,
                    ando_powers=(ando1.power, ando2.power),
                )

            lims = get_new_OSA_lims(OSA_temp, wl1_temp, wl2_temp)
        sig_start = new_sig_start(
            data_folder, wl2_temp, wl1_temp, wl_tot, max_peak_min_height
        )
        # Set the new wavelength for TiSa
        TiSa.set_wavelength(sig_start, OSA_GPIB_num=OSA_GPIB_num)
