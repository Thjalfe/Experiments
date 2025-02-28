import glob

import os
import pickle
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np

plt.style.use("custom")
data_super_dir = "../data/"
data_files = glob.glob(f"{data_super_dir}**/*.pkl", recursive=True)
sub_str_req = os.path.join(data_super_dir, "telecom-wl")
data_filenames = [
    pathname
    for pathname in data_files
    if sub_str_req in pathname and "old" not in pathname
]
data_list = []
for filename in data_filenames:
    with open(filename, "rb") as f:
        data_list.append(pickle.load(f))


def merge_two_datasets(data1: dict, data2: dict):
    data_merged = {}
    nonempty_specs_1_count = 0
    nonempty_specs_2_count = 0
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


def plot_relevant_results(measured_data: dict):
    fig, ax = plt.subplots()
    ax = cast(Axes, ax)
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
    ax = cast(Axes, ax)
    ax.plot(
        measured_data["tisa_wl_array"],
        measured_data["optimum_vals_dict"]["idler_powers"],
    )
    ax.set_xlabel("Tisa wl [nm]")
    ax.set_ylabel("Idler power [dBm]")
    ax.set_title(f"Telecom wl: {measured_data['telecom_wl']:.1f} nm, at max toptica wl")
    fig, ax = plt.subplots()
    ax = cast(Axes, ax)
    ax.plot(
        measured_data["tisa_wl_array"],
        measured_data["optimum_vals_dict"]["idler_wls"],
    )
    ax.set_xlabel("Tisa wl [nm]")
    ax.set_ylabel("Idler wl [nm]")
    ax.set_title(f"Telecom wl: {measured_data['telecom_wl']:.1f} nm, at max toptica wl")
