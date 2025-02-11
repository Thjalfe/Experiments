import numpy as np
import re
import os
from .utils import get_all_unique_pairs_list, load_raw_data, analyze_data


def sort_multiple_datasets(data_folders, file_type="pkl"):
    sorted_peak_data = []
    blue_sorted_peak_data = []
    red_sorted_peak_data = []
    unique_pairs = []
    raw_data = []
    for data_folder in data_folders:
        unique_pairs_temp = get_all_unique_pairs_list(data_folder, file_type=file_type)
        rawdatatemp, _, _ = load_raw_data(
            data_folder, -85, None, unique_pairs_temp, file_type=file_type
        )
        (
            sorted_peak_data_temp,
            blueshift_sorted_peak_data_temp,
            redshift_sorted_peak_data_temp,
            _,
        ) = analyze_data(
            data_folder,
            pump_wl_pairs=unique_pairs_temp,
            file_type=file_type,
            max_peak_min_height=-45,
        )
        unique_pairs.append(unique_pairs_temp)
        raw_data.append(rawdatatemp)
        sorted_peak_data.append(sorted_peak_data_temp)
        blue_sorted_peak_data.append(blueshift_sorted_peak_data_temp)
        red_sorted_peak_data.append(redshift_sorted_peak_data_temp)
    return (
        unique_pairs,
        raw_data,
        sorted_peak_data,
        blue_sorted_peak_data,
        red_sorted_peak_data,
    )


def pumpsep_opt_ce(dataset, pairs, stepsize=1):
    try:
        init_sep = pairs[0][1] - pairs[0][0]
    except IndexError:
        init_sep = pairs[1] - pairs[0]
    best_ce = []
    best_ce_loc = []
    best_idx = []
    for i, key in enumerate(np.atleast_1d(pairs)):
        try:
            best_idx.append(list(dataset[key].keys())[0])
        except TypeError:
            key = tuple(key)
            best_idx.append(list(dataset[key].keys())[0])
        best_ce.append(min(dataset[key][best_idx[i]]["differences"]))
        temp_best_loc = np.argmax(dataset[key][best_idx[i]]["peak_values"])
        best_ce_loc.append(dataset[key][best_idx[i]]["peak_positions"][temp_best_loc])
    x = np.arange(init_sep, len(best_ce) + 1, stepsize)
    res = np.vstack((x, -np.array(best_ce), best_ce_loc))
    return res


def multi_pumpsep_opt_ce(datasets, pairs, stepsize=1):
    res = []
    for dataset, pair in zip(datasets, pairs):
        res.append(pumpsep_opt_ce(dataset, pair, stepsize=stepsize))
    return res


def get_subfolders(folder_path):
    subfolders = []
    for dir in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, dir)
        if os.path.isdir(subfolder_path):
            if not subfolder_path.endswith("/"):
                subfolder_path += "/"
            if re.fullmatch(r"\d+", dir):
                subfolders.append(subfolder_path)
    return subfolders
