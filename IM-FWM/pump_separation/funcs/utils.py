import numpy as np
import scipy.signal
import re
import os
import glob
import pickle
from natsort import natsorted


def load_files(file_paths):
    loaded_files = []
    extra_data_list = []
    if file_paths[0].endswith(".pkl"):
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                loaded_data = pickle.load(f)
                # Ensure 'wavelengths' and 'powers' are the keys in the dictionary
                assert "wavelengths" in loaded_data and "powers" in loaded_data
                # Combine the wavelengths and powers into a 2D numpy array
                combined_data = np.column_stack(
                    (loaded_data["wavelengths"], loaded_data["powers"])
                )
                loaded_files.append(combined_data)

                # Remove 'wavelengths' and 'powers' from the loaded_data dictionary
                del loaded_data["wavelengths"]
                del loaded_data["powers"]
                extra_data_list.append(loaded_data)
    else:
        for file_path in file_paths:
            loaded_data = np.loadtxt(file_path, delimiter=",")
            loaded_files.append(loaded_data)
        extra_data_list = None

    loaded_files = np.stack(np.atleast_1d(loaded_files))
    return loaded_files, extra_data_list


def filter_files_by_pump_wl_pair(file_list, chosen_pair, pattern):
    file_paths = [
        file_name
        for file_name in file_list
        if pattern.match(file_name)
        and (
            float(pattern.match(file_name).group(1)),
            float(pattern.match(file_name).group(2)),
        )
        == chosen_pair
    ]
    return file_paths


def possible_idler_locations(sig_wl, p1_wl, p2_wl):
    """
    Returns a list of possible idler wavelengths given pump wavelengths.

    Args:
        sig_wl (float): Signal wavelength.
        p1_wl (float): First pump wavelength.
        p2_wl (float): Second pump wavelength.

    Returns:
        list: A list of possible idler wavelengths.
    """
    return [
        1 / (1 / sig_wl + 1 / p1_wl - 1 / p2_wl),
        1 / (1 / sig_wl - 1 / p1_wl + 1 / p2_wl),
    ]


def return_filtered_peaks(
    data, prominence, height, pump_wavelengths, tolerance_nm, max_peak_height
):
    """
    Filters the peaks based on prominence, height, pump wavelengths and tolerance_nm.

    Args:
        data (numpy.ndarray): 2D array containing x and y values of the dataset.
        prominence (float): Minimum prominence of the peaks.
        height (float): Minimum height of the peaks.
        pump_wavelengths (tuple): Tuple containing two pump wavelengths.
        tolerance_nm (float): Tolerance for peak position deviation.

    Returns:
        list: A list of filtered peaks.
    """
    wl_res = data[1, 0] - data[0, 0]
    tolerance_idxs = np.ceil(tolerance_nm / wl_res)
    peaks, _ = scipy.signal.find_peaks(data[:, 1], prominence=prominence, height=height)
    # Find the maximum peak location
    max_peak_loc = peaks[np.nanargmax(data[peaks, 1])]
    if data[max_peak_loc, 1] < max_peak_height:
        return ["No peak found"]
    # Calculate peak_positions based on energy_requirements
    peak_positions = possible_idler_locations(
        data[max_peak_loc, 0], pump_wavelengths[0], pump_wavelengths[1]
    )
    peak_position_indices = [
        np.nanargmin(np.abs(data[:, 0] - peak_position))
        for peak_position in peak_positions
    ]
    filtered_peaks = [max_peak_loc]
    for pos in peak_position_indices:
        peak = min(peaks, key=lambda x: abs(x - pos))
        if abs(peak - pos) <= tolerance_idxs:
            filtered_peaks.append(peak)
    return list(np.unique(filtered_peaks))


def calculate_differences(peaks, data):
    """
    Calculate the differences between the highest peak and other peaks.

    Args:
        peaks (list): List of peak indices.
        data (numpy.ndarray): 2D array containing x and y values of the dataset.

    Returns:
        list: A list of differences between the highest peak and other peaks.
    """
    peak_values = [data[p, 1] for p in peaks]

    # Find the index of the highest peak
    highest_peak_index = np.argmax(peak_values)
    highest_peak_value = peak_values[highest_peak_index]

    peak_differences = [
        abs(highest_peak_value - peak_values[i])
        if i != highest_peak_index
        else float("inf")
        for i in range(len(peak_values))
    ]
    return peak_differences


def process_single_dataset(
    data, prominence, height, peak_positions, tolerance_nm, max_peak_height
):
    """
    Process a single dataset to find peaks, differences, peak values, and peak positions.

    Args:
        data (numpy.ndarray): 2D array containing data for a single dataset.
        prominence (float): Minimum prominence of the peaks.
        height (float): Minimum height of the peaks.
        peak_positions (tuple): Tuple containing two pump wavelengths.
        tolerance_nm (float): Tolerance_nm for peak position deviation.
        max_peak_height (float, optional): Maximum peak height. This is to avoid false good measurements, in case a measurement failed and did not detect any peak at all.

    Returns:
        dict/str: A dictionary containing peaks, differences, peak values, and peak positions for the dataset or a string indicating a missing or single peak.
    """
    peaks = return_filtered_peaks(
        data, prominence, height, peak_positions, tolerance_nm, max_peak_height
    )
    if peaks == ["No peak found"]:
        return "No peak found"
    if len(peaks) == 1:
        return "Only signal peak found"

    differences = calculate_differences(peaks, data)
    return {
        "peaks": peaks,
        "differences": differences,
        "peak_values": [data[p, 1] for p in peaks],
        "peak_positions": [data[p, 0] for p in peaks],
    }


def process_multiple_datasets(
    dataset, prominence, height, peak_positions, tolerance_nm, max_peak_height
):
    """
    Process all datasets to find peaks, differences, peak values, and peak positions for each dataset.

    Args:
        dataset (numpy.ndarray): 3D array containing data for each file.
        prominence (float): Minimum prominence of the peaks.
        height (float): Minimum height of the peaks.
        peak_positions (tuple): Tuple containing two pump wavelengths.
        tolerance_nm (float): Tolerance_nm for peak position deviation.
        max_peak_height (float, optional): Maximum peak height. This is to avoid false good measurements, in case a measurement failed and did not detect any peak at all.

    Returns:
        dict: A dictionary containing peaks, differences, peak values, and peak positions for each file.
    """
    peak_data = {}
    for m in range(dataset.shape[0]):
        peak_data[m] = process_single_dataset(
            dataset[m],
            prominence,
            height,
            peak_positions,
            tolerance_nm,
            max_peak_height,
        )
    return peak_data


def get_signal_wavelength(item):
    if type(item) == dict:
        max_index = item["peak_values"].index(max(item["peak_values"]))
        return item["peak_positions"][max_index]
    else:
        return None


def sort_peak_data(peak_data, sort_key):
    def get_blue_shift(item, signal_wavelength, sort_key):
        if type(item) == dict and sort_key in item:
            blue_shift = [
                item[sort_key][i]
                for i, pos in enumerate(item["peak_positions"])
                if pos < signal_wavelength
            ]
            return blue_shift[0] if len(blue_shift) > 0 else float("inf")
        else:
            return float("inf")

    def get_red_shift(item, signal_wavelength, sort_key):
        if type(item) == dict and sort_key in item:
            red_shift = [
                item[sort_key][i]
                for i, pos in enumerate(item["peak_positions"])
                if pos > signal_wavelength
            ]
            return min(red_shift) if len(red_shift) > 0 else float("inf")
        else:
            return float("inf")

    abs_sort = dict(
        sorted(
            peak_data.items(),
            key=lambda x: min(x[1][sort_key]) if type(x[1]) == dict else float("inf"),
        )
    )
    blue_shift_sort = dict(
        sorted(
            peak_data.items(),
            key=lambda x: get_blue_shift(x[1], get_signal_wavelength(x[1]), sort_key),
        )
    )

    red_shift_sort = dict(
        sorted(
            peak_data.items(),
            key=lambda x: get_red_shift(x[1], get_signal_wavelength(x[1]), sort_key),
        )
    )
    return abs_sort, blue_shift_sort, red_shift_sort


def load_raw_data(
    data_folder, nan_filter, pair_indices=None, pump_wl_pairs=None, file_type="pkl"
):
    data_folder = data_folder[:-1]
    file_list = natsorted(glob.glob(data_folder + f"/*[0-9].{file_type}"))
    pattern = re.compile(rf"{data_folder}[\\/]([\d.]+)_([\d.]+)_\d+.{file_type}")
    if pair_indices is not None:
        unique_pairs = set()
        for file_name in file_list:
            match = pattern.match(file_name)
            if match:
                pair = (float(match.group(1)), float(match.group(2)))
                unique_pairs.add(pair)
        unique_pairs_list = list(unique_pairs)[::-1]
        chosen_pairs = [unique_pairs_list[pair_index] for pair_index in pair_indices]
    elif pump_wl_pairs is not None:
        chosen_pairs = pump_wl_pairs
    else:
        raise TypeError(
            'You must choose to pick files either with "pair_indices" or "pump_wl_pairs"'
        )
    # If only one pair is chosen, convert it to a list such that it can be iterated over
    if type(chosen_pairs) == tuple:
        chosen_pairs = [chosen_pairs]
    filtered_file_lists = []
    for pair in chosen_pairs:
        filtered_file_list = filter_files_by_pump_wl_pair(file_list, pair, pattern)
        filtered_file_lists.append(filtered_file_list)

    data = []
    extra_data = []
    for filtered_file_list in filtered_file_lists:
        dataset, extra_dataset = load_files(filtered_file_list)
        data.append(dataset)
        extra_data.append(extra_dataset)

    for dataset in data:
        dataset[dataset[:, :, 1] < nan_filter, 1] = np.nan

    return data, chosen_pairs, extra_data


def analyze_data(
    data_folder,
    pair_indices=None,
    pump_wl_pairs=None,
    nan_filter=-85,
    prominence=0.1,
    height=-75,
    tolerance_nm=0.25,
    max_peak_min_height=-35,
    file_type="pkl",
):
    data, chosen_pairs, _ = load_raw_data(
        data_folder, nan_filter, pair_indices, pump_wl_pairs, file_type
    )

    all_sorted_peak_data = {}
    all_blueshift_sorted_peak_data = {}
    all_redshift_sorted_peak_data = {}
    unsorted_peak_data = {}
    for i, dataset in enumerate(data):
        peak_data = process_multiple_datasets(
            dataset,
            prominence,
            height,
            chosen_pairs[i],
            tolerance_nm,
            max_peak_min_height,
        )
        (
            sorted_peak_data,
            blueshift_sorted_peak_data,
            redshift_sorted_peak_data,
        ) = sort_peak_data(peak_data, "differences")
        all_sorted_peak_data[chosen_pairs[i]] = sorted_peak_data
        all_blueshift_sorted_peak_data[chosen_pairs[i]] = blueshift_sorted_peak_data
        all_redshift_sorted_peak_data[chosen_pairs[i]] = redshift_sorted_peak_data
        unsorted_peak_data[chosen_pairs[i]] = peak_data
    return (
        all_sorted_peak_data,
        all_blueshift_sorted_peak_data,
        all_redshift_sorted_peak_data,
        unsorted_peak_data,
    )


def get_all_unique_pairs_list(data_folder, file_type="pkl"):
    data_folder = data_folder[:-1]
    file_list = natsorted(glob.glob(data_folder + f"/*[0-9].{file_type}"))
    pattern = re.compile(rf"{data_folder}[\\/]([\d.]+)_([\d.]+)_\d+.{file_type}")
    unique_pairs = set()
    for file_name in file_list:
        match = pattern.match(file_name)
        if match:
            pair = (float(match.group(1)), float(match.group(2)))
            unique_pairs.add(pair)
    unique_pairs_list = list(unique_pairs)
    sorted_unique_pairs_list = sorted(unique_pairs_list, key=lambda x: x[0])
    return sorted_unique_pairs_list[::-1]


def extract_pump_numbers(file_path, pump_name):
    match = re.search(rf"{pump_name}_([\d\.]+)_([\d\.]+)\.csv", file_path)
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        return 0, 0


def load_pump_files(data_folder, pump_name="pump_name", nan_filter=-80):
    pattern = os.path.join(data_folder, f"{pump_name}*")
    matching_files = glob.glob(pattern)
    # Sort the matching_files list based on the first number in descending order
    sorted_files = sorted(
        matching_files, key=lambda x: extract_pump_numbers(x, pump_name), reverse=True
    )
    pumps = []
    number_tuples = []
    for file in sorted_files:
        pumps.append(np.loadtxt(file, delimiter=","))
        number_tuples.append(extract_pump_numbers(file, pump_name))
    pumps = np.array(pumps)
    pumps[:, :, 1][pumps[:, :, 1] < nan_filter] = np.nan
    return pumps, number_tuples


def sort_by_pump_nm_difference(paths):
    def nm_difference(path):
        numbers = re.findall(r"(\d+.\d+)nm", path)
        return abs(float(numbers[-1]) - float(numbers[-2]))

    return sorted(paths, key=nm_difference)


def extract_pump_wls(s):
    numbers = re.findall(r"\d+\.\d+", s)
    numbers = [float(num) for num in numbers]
    numbers = numbers[-2:]
    numbers.sort()
    return numbers


def get_subdirectories(directory):
    return [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]
