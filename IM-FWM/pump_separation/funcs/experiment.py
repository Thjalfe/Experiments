from .utils import analyze_data
import numpy as np
import time
import sys
import os
import pickle

sys.path.append("U:/Elec/NONLINEAR-FOD/Thjalfe/InstrumentControl/InstrumentControl")
import instrument_class as misc
from OSA_control import OSA


def save_sweep(data_folder, filename, **data):
    # Create the base folder if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Create a unique filename
    filename = f"{filename}.pkl"
    # Save the data as a .pkl file
    with open(os.path.join(data_folder, filename), "wb") as f:
        pickle.dump(data, f)


# def even_ando_power(ando1, ando2, ando_pow_start):
#     ando1_pow = ando1.get_true_power()
#     ando2_pow = ando2.get_true_power()
#     if ando1_pow > ando2_pow:
#         ando1.set_power(ando_pow_start)
#         ando2.set_power(ando_pow_start + (ando1_pow - ando2_pow))
#     elif ando2_pow > ando1_pow:
#         ando2.set_power(ando_pow_start)
#         ando1.set_power(ando_pow_start + (ando2_pow - ando1_pow))


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
        ando1.set_power(1)
        ando2.set_power(1)
        peak_diff = peak1 - peak2
        print(peak1, peak2)
        i += 1
        if np.abs(peak_diff) <= 0.5:
            break
        elif peak_diff > 0:
            ando1.set_power(ando1.power - np.abs(peak_diff) / 2)
            ando2.set_power(ando2.power + np.abs(peak_diff) / 2)
            time.sleep(0.5)
        elif peak_diff < 0:
            ando2.set_power(ando2.power - np.abs(peak_diff) / 2)
            ando1.set_power(ando1.power + np.abs(peak_diff) / 2)
            time.sleep(0.5)
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


def new_sig_start(
    data_folder, pumpwl_low, pumpwl_high, wl_tot, max_peak_min_height, sortby
):
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
    all_peaks, blue, red, _ = analyze_data(
        data_folder,
        pump_wl_pairs=[(pumpwl_low, pumpwl_high)],
        max_peak_min_height=max_peak_min_height,
    )
    if sortby == "red":
        data = red[(pumpwl_low, pumpwl_high)]
        shift_factor = 0.6  # for this specific fiber, the optimal signal is redshifted when considering red idler, so it is shifted a bit further to the red
    elif sortby == "blue":
        data = blue[(pumpwl_low, pumpwl_high)]
        shift_factor = 0.4
    elif sortby == "all":
        data = all_peaks[(pumpwl_low, pumpwl_high)]
        shift_factor = 0.5
    else:
        raise ValueError("sortby must be 'red', 'blue' or 'all'")
    max_peak_idx = list(data)[0]
    if type(max_peak_idx) is not int:
        raise ValueError(
            "No peak found in the data. Maybe the max_peak_min_height is too high?"
        )
    sig_peak_idx = np.argmax(data[max_peak_idx]["peak_values"])
    peak_pos = data[max_peak_idx]["peak_positions"][sig_peak_idx]
    new_start = peak_pos - wl_tot * shift_factor
    print(
        f"Peak found at {peak_pos:.2f} for ({pumpwl_low}, {pumpwl_high}). New signal start position: {new_start:.2f} nm"
    )
    return new_start


def new_data_folder(data_folder):
    # Find existing directories in the data_folder
    existing_dirs = [
        d
        for d in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, d))
    ]
    # Filter out directories that are not integers (not numeric)
    existing_dirs = [d for d in existing_dirs if d.isdigit()]

    # Find the highest numbered directory
    if existing_dirs:
        max_dir = max(existing_dirs, key=int)
    else:
        max_dir = "0"

    # Create next directory
    next_dir = str(int(max_dir) + 1)
    new_folder_path = os.path.join(data_folder, next_dir)
    os.makedirs(new_folder_path)
    if not new_folder_path.endswith("/"):
        new_folder_path += "/"
    return new_folder_path


def init_pump_and_osa(
    data_folder,
    osa,
    ando1,
    ando2,
    wl1,
    wl2,
    ando1_wl_name,
    ando2_wl_name,
    sig_start,
    ando1_power,
    ando2_power,
    over_sampling,
    equal_ando_output_powers,
    adjust_laser_wavelengths,
    OSA_res,
    OSA_GPIB_num,
):
    ando1.set_power(ando1_power, adjust_power=equal_ando_output_powers)
    ando2.set_power(ando2_power, adjust_power=equal_ando_output_powers)
    ando1.set_wavelength(wl1)
    ando2.set_wavelength(wl2)
    if adjust_laser_wavelengths:
        ando1.adjust_wavelength(OSA_GPIB_num=OSA_GPIB_num)
        ando2.adjust_wavelength(OSA_GPIB_num=OSA_GPIB_num)
    osa.save(f"{data_folder}/pumps{ando2_wl_name}_{ando1_wl_name}")
    if type(sig_start) == list:
        osa.set_span(
            sig_start - np.abs(wl1 - wl2),
            sig_start + np.abs(wl1 - wl2),
        )
    else:
        osa.set_span(
            sig_start - np.abs(wl1 - wl2),
            sig_start + np.abs(wl1 - wl2),
        )
    osa.sweep()
    lims = get_new_OSA_lims(osa, ando1_wl_name, ando2_wl_name)
    temp_sampling = over_sampling * 2 * (lims[1] - lims[0]) / OSA_res
    osa.set_sample(temp_sampling)
    return lims


def sweep_tisa_and_save(
    num_sweeps,
    del_wl,
    wl_tot,
    TiSa,
    osa,
    ando1,
    ando2,
    data_folder,
    lims,
    log_pm,
    wl1,
    wl2,
    sig_start,
    OSA_GPIB_num,
    max_peak_min_height,
    sortpeaksby,
    iter_num,
    sig_start_external,
):
    for j in range(num_sweeps):
        TiSa.delta_wl_nm(del_wl)
        time.sleep(0.5)
        osa.set_span(lims[0], lims[1])
        osa.sweep()
        if log_pm:
            save_sweep(
                data_folder,
                f"{wl2}_{wl1}_{j}",
                wavelengths=osa.wavelengths,
                powers=osa.powers,
                ando_powers=(ando1.power, ando2.power),
                PM=misc.PM().read(),
            )
        else:
            save_sweep(
                data_folder,
                f"{wl2}_{wl1}_{j}",
                wavelengths=osa.wavelengths,
                powers=osa.powers,
                ando_powers=(ando1.power, ando2.power),
            )
        lims = get_new_OSA_lims(osa, wl1, wl2)
    if not sig_start_external:
        if type(sig_start) == list:
            TiSa.set_wavelength(sig_start[iter_num], OSA_GPIB_num=OSA_GPIB_num)
            return
        else:
            sig_start = new_sig_start(
                data_folder,
                wl2,
                wl1,
                wl_tot,
                max_peak_min_height,
                sortpeaksby,
            )
# Set the new wavelength for TiSa
        TiSa.set_wavelength(sig_start, OSA_GPIB_num=OSA_GPIB_num)


def run_tisa_sweep_all_pump_wls(
    data_folder,
    ando1,
    ando2,
    TiSa,
    ando1_wl,
    ando2_wl,
    ando1_wl_names,
    ando2_wl_names,
    num_sweeps,
    del_wl,
    wl_tot,
    sig_start,
    adjust_laser_wavelengths=True,
    equal_ando_output_powers=True,
    equal_pump_power=False,
    log_pm=False,
    OSA_sens="SHI1",
    OSA_res=0.05,
    over_sampling=2.5,
    OSA_GPIB_num=[0, 19],
    max_peak_min_height=-35,
    ando1_power=0,
    ando2_power=0,
    sortpeaksby="blue",
    make_new_folder_iter=True,
    sweep_pumps=True,
    sig_start_external=False,
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
        sig_start(float or list): Start wavelength for TiSa, can be list if  we want to set it manually. Mainly done when trying to redo measurements from earlier.
        GPIB_val (int, optional): GPIB value for the OSA. Defaults to 19.
    """
    if make_new_folder_iter:
        data_folder = new_data_folder(data_folder)
    osa = OSA(
        ando2_wl[0] - 1,
        ando1_wl[0] + 1,
        resolution=OSA_res,
        sensitivity=OSA_sens,
        GPIB_num=OSA_GPIB_num,
    )
    for i, (wl1_temp, wl2_temp) in enumerate(zip(ando1_wl, ando2_wl)):
        lims = init_pump_and_osa(
            data_folder,
            osa,
            ando1,
            ando2,
            wl1_temp,
            wl2_temp,
            ando1_wl_names[i],
            ando2_wl_names[i],
            sig_start,
            ando1_power,
            ando2_power,
            over_sampling,
            equal_ando_output_powers,
            adjust_laser_wavelengths,
            OSA_res,
            OSA_GPIB_num,
        )
        sweep_tisa_and_save(
            num_sweeps,
            del_wl,
            wl_tot,
            TiSa,
            osa,
            ando1,
            ando2,
            data_folder,
            lims,
            log_pm,
            ando1_wl_names,
            ando2_wl_names,
            sig_start,
            OSA_GPIB_num,
            max_peak_min_height,
            sortpeaksby,
            i,
            sig_start_external,
        )


def run_tisa_sweep_single_pump_wl(
    data_folder,
    ando1,
    ando2,
    TiSa,
    ando1_wl,
    ando2_wl,
    ando1_wl_name,
    ando2_wl_name,
    num_sweeps,
    del_wl,
    wl_tot,
    sig_start,
    adjust_laser_wavelengths=True,
    equal_ando_output_powers=True,
    equal_pump_power=False,
    log_pm=False,
    OSA_sens="SHI1",
    OSA_res=0.05,
    over_sampling=2.5,
    OSA_GPIB_num=[0, 19],
    max_peak_min_height=-35,
    ando1_power=0,
    ando2_power=0,
    sortpeaksby="blue",
    make_new_folder_iter=True,
    sig_start_external=False,
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
        sig_start(float or list): Start wavelength for TiSa, can be list if  we want to set it manually. Mainly done when trying to redo measurements from earlier.
        GPIB_val (int, optional): GPIB value for the OSA. Defaults to 19.
    """
    if make_new_folder_iter:
        data_folder = new_data_folder(data_folder)
    osa = OSA(
        ando2_wl - 1,
        ando1_wl + 1,
        resolution=OSA_res,
        sensitivity=OSA_sens,
        GPIB_num=OSA_GPIB_num,
    )
    lims = init_pump_and_osa(
        data_folder,
        osa,
        ando1,
        ando2,
        ando1_wl,
        ando2_wl,
        ando1_wl_name,
        ando2_wl_name,
        sig_start,
        ando1_power,
        ando2_power,
        over_sampling,
        equal_ando_output_powers,
        adjust_laser_wavelengths,
        OSA_res,
        OSA_GPIB_num,
    )
    if not data_folder.endswith("/"):
        data_folder += "/"
    sweep_tisa_and_save(
        num_sweeps,
        del_wl,
        wl_tot,
        TiSa,
        osa,
        ando1,
        ando2,
        data_folder,
        lims,
        log_pm,
        ando1_wl_name,
        ando2_wl_name,
        sig_start,
        OSA_GPIB_num,
        max_peak_min_height,
        sortpeaksby,
        0,
        sig_start_external,
    )
