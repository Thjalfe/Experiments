import numpy as np
import pickle
from pump_separation.funcs.utils import extract_sig_wl_and_ce_multiple_spectra


def process_ce_data_for_pump_sweep_around_opt(
    datafile_loc: str, data_process_method: callable
):
    with open(datafile_loc, "rb") as f:
        data = pickle.load(f)

    pump_wl_pairs = list(data.keys())
    duty_cycles = data[pump_wl_pairs[0]]["params"]["duty_cycles"]
    ando1_wls = []
    ando2_wls = []
    ando1_wls_rel = []
    ando2_wls_rel = []
    for pump_wl_pair in pump_wl_pairs:
        ando1_wls.append(data[pump_wl_pair]["ando1_wls"])
        ando2_wls.append(data[pump_wl_pair]["ando2_wls"])
        ando1_wls_rel.append(data[pump_wl_pair]["ando1_wls"] - pump_wl_pair[0])
        ando2_wls_rel.append(data[pump_wl_pair]["ando2_wls"] - pump_wl_pair[1])
    pump_sep_ax = np.array([np.abs(pair[1] - pair[0]) for pair in pump_wl_pairs])
    num_reps = data[pump_wl_pairs[0]]["params"]["num_sweep_reps"]
    exp_params = {
        "pump_wl_pairs": pump_wl_pairs,
        "duty_cycles": duty_cycles,
        "ando1_wls": ando1_wls,
        "ando2_wls": ando2_wls,
        "ando1_wls_rel": ando1_wls_rel,
        "ando2_wls_rel": ando2_wls_rel,
        "pump_sep_ax": pump_sep_ax,
        "num_reps": num_reps,
    }
    ce_dict = {
        pump_wl_pair: {
            dc: np.zeros((num_reps, len(ando1_wls[0]), len(ando2_wls[0])))
            for dc in duty_cycles
        }
        for pump_wl_pair in pump_wl_pairs
    }
    sig_wl_dict = {
        pump_wl_pair: {
            dc: np.zeros((num_reps, len(ando1_wls[0]), len(ando2_wls[0])))
            for dc in duty_cycles
        }
        for pump_wl_pair in pump_wl_pairs
    }
    idler_wl_dict = {
        pump_wl_pair: {
            dc: np.zeros((num_reps, len(ando1_wls[0]), len(ando2_wls[0])))
            for dc in duty_cycles
        }
        for pump_wl_pair in pump_wl_pairs
    }
    ce_dict_processed = {
        pump_wl_pair: {
            dc: np.zeros((len(ando1_wls[0]), len(ando2_wls[0]))) for dc in duty_cycles
        }
        for pump_wl_pair in pump_wl_pairs
    }
    sig_wl_dict_processed = {
        pump_wl_pair: {
            dc: np.zeros((len(ando1_wls[0]), len(ando2_wls[0]))) for dc in duty_cycles
        }
        for pump_wl_pair in pump_wl_pairs
    }
    idler_wl_dict_processed = {
        pump_wl_pair: {
            dc: np.zeros((len(ando1_wls[0]), len(ando2_wls[0]))) for dc in duty_cycles
        }
        for pump_wl_pair in pump_wl_pairs
    }
    ce_dict_std = {
        pump_wl_pair: {
            dc: np.zeros((len(ando1_wls[0]), len(ando2_wls[0]))) for dc in duty_cycles
        }
        for pump_wl_pair in pump_wl_pairs
    }
    ce_dict_best_for_each_ando_sweep = {
        pump_wl_pair: {dc: [] for dc in duty_cycles} for pump_wl_pair in pump_wl_pairs
    }
    ce_dict_best_loc_for_each_ando_sweep = {
        pump_wl_pair: {dc: [] for dc in duty_cycles} for pump_wl_pair in pump_wl_pairs
    }
    for pump_wl_pair_idx, pump_wl_pair in enumerate(pump_wl_pairs):
        for dc in duty_cycles:
            ando1_wls_tmp = ando1_wls[pump_wl_pair_idx]
            spectra = np.array(data[pump_wl_pair]["spectra"][dc])
            for rep in range(num_reps):
                for i, wl1 in enumerate(ando1_wls_tmp):
                    spectra_sgl_rep = spectra[i, :, rep]
                    spectra_sgl_rep = np.transpose(spectra_sgl_rep, (0, 2, 1))
                    (
                        sig_wl_tmp,
                        ce_tmp,
                        idler_wl_tmp,
                    ) = extract_sig_wl_and_ce_multiple_spectra(
                        spectra_sgl_rep,
                        list(pump_wl_pair),
                        np.shape(spectra_sgl_rep)[0],
                    )
                    ce_dict[pump_wl_pair][dc][rep, i, :] = -ce_tmp
                    sig_wl_dict[pump_wl_pair][dc][rep, i, :] = sig_wl_tmp
                    idler_wl_dict[pump_wl_pair][dc][rep, i, :] = idler_wl_tmp
                    ce_dict_processed[pump_wl_pair][dc][i, :] = data_process_method(
                        ce_dict[pump_wl_pair][dc][:, i, :], axis=0
                    )
                    ce_dict_std[pump_wl_pair][dc][i, :] = np.std(
                        ce_dict[pump_wl_pair][dc][:, i, :], axis=0
                    )
                    sig_wl_dict_processed[pump_wl_pair][dc][i, :] = data_process_method(
                        sig_wl_dict[pump_wl_pair][dc][:, i, :], axis=0
                    )
                    idler_wl_dict_processed[pump_wl_pair][dc][
                        i, :
                    ] = data_process_method(
                        idler_wl_dict[pump_wl_pair][dc][:, i, :], axis=0
                    )
            ce_dict_best_for_each_ando_sweep[pump_wl_pair][dc] = np.max(
                np.squeeze(ce_dict_processed[pump_wl_pair][dc]), axis=0
            )
            ce_dict_best_loc_for_each_ando_sweep[pump_wl_pair][dc] = np.argmax(
                np.squeeze(ce_dict_processed[pump_wl_pair][dc]), axis=0
            )

    return (
        exp_params,
        ce_dict,
        sig_wl_dict,
        idler_wl_dict,
        ce_dict_processed,
        sig_wl_dict_processed,
        idler_wl_dict_processed,
        ce_dict_std,
        ce_dict_best_for_each_ando_sweep,
        ce_dict_best_loc_for_each_ando_sweep,
    )
