# This file contains functions to process data created by ../../../classical_check/run_sweep_funcs.py
import numpy as np


def add_data_process_dict_to_data_dict(data_dict: dict) -> dict:
    def find_idler_wl_approx(sig_wl: float, pump_wls: tuple[float, float]) -> float:
        return 1 / (1 / sig_wl + 1 / pump_wls[0] - 1 / pump_wls[1])

    def find_sig_idler_wl(
        data: np.ndarray,
        wl_arr: np.ndarray,
        pump_wls: tuple[float, float],
        idler_tolerance: float = 0.5,
        min_counts_idler: float = -50,
    ) -> tuple[float, float]:
        possible_idler_locs_tmp = np.where(data > min_counts_idler)[0]
        sig_wl = wl_arr[np.argmax(data)]
        idler_wl_approx = find_idler_wl_approx(sig_wl, pump_wls)
        possible_idler_locs = possible_idler_locs_tmp[
            np.abs(wl_arr[possible_idler_locs_tmp] - idler_wl_approx) < idler_tolerance
        ]
        idler_loc = possible_idler_locs[np.argmax(data[possible_idler_locs])]
        idler_wl = float(wl_arr[idler_loc])
        return sig_wl, idler_wl

    def pump_wls_to_thz_sep(p1wl, p2wl, c=299792458):
        p1wl_thz = c / (p1wl * 10**-9) * 10**-12
        p2wl_thz = c / (p2wl * 10**-9) * 10**-12
        return p1wl_thz - p2wl_thz

    def pad_to_full_order(dc_values, full_order, array):
        full_order = (
            list(full_order) if isinstance(full_order, np.ndarray) else full_order
        )
        padded_array = np.full(len(full_order), np.nan)  # Start with NaNs
        for idx, dc in enumerate(dc_values):
            if dc in full_order:
                full_order_idx = full_order.index(dc)
                if idx < len(
                    array
                ):  # Assign only if there's a corresponding value in array
                    padded_array[full_order_idx] = array[idx]
        return padded_array

    float_keys = sorted(
        [k for k in data_dict.keys() if isinstance(k, (np.floating, np.integer))]
    )
    data_process_dict = {
        "p1_wls": np.array(float_keys),
        "p2_wl_max_ce": [],
        "sig_wls": [],
        "idler_wls": [],
        "dc_per_dataset": [],
        "pump_sep_thz": [],
        "red_idxs": [],
        "blue_idxs": [],
        "ce_peak_mean": [],
        "ce_peak_std": [],
        "ce_peak_mean_lin": [],
        "ce_peak_std_lin": [],
    }
    for p1_wl in float_keys:
        sub_data = data_dict[p1_wl]
        if len(sub_data.keys()) == 0:
            # pop and continue if there's no data
            data_dict.pop(p1_wl)
            continue
        data_process_dict["p2_wl_max_ce"].append(sub_data["p2_max_ce"])
        wl_arr = np.array(sub_data["spectra_pol_opt"][0, 0, 0, :])
        pow_arr = np.array(sub_data["spectra_pol_opt"][0, 0, 1, :])
        sig_wl, idler_wl = find_sig_idler_wl(
            pow_arr,
            wl_arr,
            (float(p1_wl), sub_data["p2_max_ce"]),
        )
        data_process_dict["sig_wls"].append(sig_wl)
        data_process_dict["idler_wls"].append(idler_wl)
        try:
            data_process_dict["dc_per_dataset"].append(sub_data["duty_cycle"])
        except KeyError:
            # Some older datasets only have duty_cycle in the top level
            data_process_dict["dc_per_dataset"].append(data_dict["duty_cycle"])
        data_process_dict["pump_sep_thz"].append(
            pump_wls_to_thz_sep(p1_wl, sub_data["p2_max_ce"])
        )
        data_process_dict["ce_peak_mean"].append(
            np.mean(np.atleast_2d(sub_data["ce_peak_pol_opt"]), axis=1)
        )
        data_process_dict["ce_peak_std"].append(
            np.std(np.atleast_2d(sub_data["ce_peak_pol_opt"]), axis=1)
        )
        ce_lin = 10 ** (np.atleast_2d(sub_data["ce_peak_pol_opt"]) / 10)
        data_process_dict["ce_peak_mean_lin"].append(np.mean(ce_lin, axis=1))
        data_process_dict["ce_peak_std_lin"].append(np.std(ce_lin, axis=1))

    data_process_dict["p2_wl_max_ce"] = np.array(data_process_dict["p2_wl_max_ce"])
    data_process_dict["sig_wls"] = np.array(data_process_dict["sig_wls"])
    data_process_dict["idler_wls"] = np.array(data_process_dict["idler_wls"])
    data_process_dict["pump_sep_thz"] = np.array(data_process_dict["pump_sep_thz"])
    data_process_dict["red_idxs"] = np.where(
        np.array(data_process_dict["pump_sep_thz"]) < 0
    )[0]
    data_process_dict["blue_idxs"] = np.where(
        np.array(data_process_dict["pump_sep_thz"]) > 0
    )[0]

    dc_arr_lens = [len(dc) for dc in data_process_dict["dc_per_dataset"]]
    max_dc_array = data_process_dict["dc_per_dataset"][np.argmax(dc_arr_lens)]
    keys_to_pad = [
        "ce_peak_mean",
        "ce_peak_std",
        "ce_peak_mean_lin",
        "ce_peak_std_lin",
    ]
    dc_per_dataset = data_process_dict["dc_per_dataset"]
    # Convert each key's data in data_process_dict to a 2D numpy array with padding
    data_process_dict["dc_per_dataset"] = np.array(
        [
            pad_to_full_order(dc_values, max_dc_array, dc_values)
            for dc_values in dc_per_dataset
        ]
    )

    # Apply the padding template to other keys
    keys_to_pad = ["ce_peak_mean", "ce_peak_std", "ce_peak_mean_lin", "ce_peak_std_lin"]
    for key in keys_to_pad:
        data_process_dict[key] = np.array(
            [
                pad_to_full_order(dc_values, max_dc_array, array)
                for dc_values, array in zip(dc_per_dataset, data_process_dict[key])
            ]
        )
    data_dict["processed_data"] = data_process_dict
    return data_dict


# Process all datasets in dirs and save alongside the original data
if __name__ == "__main__":
    from glob import glob
    import pickle

    super_dir = "./sweep_both_pumps_auto_pol_opt_780-fiber-out/"
    super_dir = "./sweep_both_pumps_auto_pol_opt_1060-fiber-out_tisa-sig/"
    super_dir = "./sweep_both_pumps_w_processed_merged_proper_data/p1-wl=1593.5-1596.0nm_p2-wl=1591.55-1572.90_merged_old_equal_p_powers-on-red-side/"
    data_files = glob(super_dir + "**/*.pkl", recursive=True)
    processed_file_names = [f.replace(".pkl", "_processed.pkl") for f in data_files]
    for data_file, processed_file in zip(data_files, processed_file_names):
        with open(data_file, "rb") as f:
            data_dict = pickle.load(f)
        data_dict = add_data_process_dict_to_data_dict(data_dict)
        with open(processed_file, "wb") as f:
            pickle.dump(data_dict, f)
        print(f"Processed data saved to {processed_file}")
