import glob
import os
import re

import numpy as np
import pandas as pd


def extract_pump_wls_from_file_name(file_name: str) -> tuple[float, float]:
    def convert_str_to_float(s):
        return float(s.replace("p", "."))

    match = re.search(r"(\d+p\d+)_(\d+p\d+)", file_name)
    if match:
        wl1 = convert_str_to_float(match.group(1))
        wl2 = convert_str_to_float(match.group(2))
    else:
        raise ValueError("No match found in file name")
    return wl1, wl2


def extract_duty_cycle_from_file_name(file_name: str) -> float:
    pattern = r"duty-cycle-([0-9]*\.[0-9]+)"
    match = re.search(pattern, file_name)
    if match:
        return float(match.group(1))
    else:
        raise ValueError("No match found in file name")


def get_data_filenames(data_dir: str) -> tuple[list[str], list[str]]:
    data_file_list = glob.glob(os.path.join(data_dir, "**/", "*.txt"), recursive=True)
    data_file_list = [
        string for string in data_file_list if "time-evolution" not in string
    ]
    data_file_list = [string for string in data_file_list if "duty-cycle" in string]

    ref_file_names = [string for string in data_file_list if "ref" in string]
    data_file_names = [string for string in data_file_list if "ref" not in string]
    return ref_file_names, data_file_names


def load_data_as_dict(file_name_arr: list[str]) -> tuple[dict, np.ndarray]:
    wls = [extract_pump_wls_from_file_name(file_name) for file_name in file_name_arr]
    file_names_sorted, wls_sorted = zip(*sorted(zip(file_name_arr, wls)))
    duty_cycles = np.array(
        [extract_duty_cycle_from_file_name(file_name) for file_name in file_name_arr]
    )
    data_dict = {}
    for file_name, wl in zip(file_names_sorted, wls_sorted):
        data_dict[wl] = pd.read_csv(file_name, delimiter="\t", header=None).to_numpy()
    return data_dict, duty_cycles
