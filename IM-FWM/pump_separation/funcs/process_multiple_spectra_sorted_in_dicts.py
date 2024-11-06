import numpy as np
from pump_separation.funcs.utils import (
    extract_sig_wl_and_ce_multiple_spectra,
    extract_sig_wl_and_ce_single_spectrum,
)


def process_ce_data_for_pump_sweep_around_opt(
    data: dict, data_process_method: callable
):
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
                for idx_wl1 in range(len(ando1_wls_tmp)):
                    spectra_sgl_rep = spectra[idx_wl1, :, rep]
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
                    ce_dict[pump_wl_pair][dc][rep, idx_wl1, :] = -ce_tmp
                    sig_wl_dict[pump_wl_pair][dc][rep, idx_wl1, :] = sig_wl_tmp
                    idler_wl_dict[pump_wl_pair][dc][rep, idx_wl1, :] = idler_wl_tmp
                    ce_dict_processed[pump_wl_pair][dc][idx_wl1, :] = (
                        data_process_method(
                            ce_dict[pump_wl_pair][dc][:, idx_wl1, :], axis=0
                        )
                    )
                    ce_dict_std[pump_wl_pair][dc][idx_wl1, :] = np.std(
                        ce_dict[pump_wl_pair][dc][:, idx_wl1, :], axis=0
                    )
                    sig_wl_dict_processed[pump_wl_pair][dc][idx_wl1, :] = (
                        data_process_method(
                            sig_wl_dict[pump_wl_pair][dc][:, idx_wl1, :], axis=0
                        )
                    )
                    idler_wl_dict_processed[pump_wl_pair][dc][idx_wl1, :] = (
                        data_process_method(
                            idler_wl_dict[pump_wl_pair][dc][:, idx_wl1, :], axis=0
                        )
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


def process_ce_data_for_pump_sweep_simple(
    data: dict, data_process_method: callable, duty_cycles: list = [0.1], num_reps=1
):
    """
    Very overengineered function to process data from a pump sweep experiment. It is made this way because it is
    copy pasted from the above function that can handle more complex data, and refactoring it is not worth the time.
    """
    pump1_wls = list(data.keys())
    try:
        duty_cycles = data[pump1_wls[0]]["params"]["duty_cycles"]
    except KeyError:
        duty_cycles = duty_cycles
    pump2_wls = data[pump1_wls[0]]["pump2_wls"]
    exp_params = {
        "pump1_wls": pump1_wls,
    }
    ce_dict = {
        pump1_wl: {dc: np.zeros(len(pump2_wls)) for dc in duty_cycles}
        for pump1_wl in pump1_wls
    }
    sig_wl_dict = {
        pump1_wl: {dc: np.zeros(len(pump2_wls)) for dc in duty_cycles}
        for pump1_wl in pump1_wls
    }
    idler_wl_dict = {
        pump1_wl: {dc: np.zeros(len(pump2_wls)) for dc in duty_cycles}
        for pump1_wl in pump1_wls
    }
    ce_dict_processed = {
        pump1_wl: {dc: np.zeros(len(pump2_wls)) for dc in duty_cycles}
        for pump1_wl in pump1_wls
    }
    sig_wl_dict_processed = {
        pump1_wl: {dc: np.zeros(len(pump2_wls)) for dc in duty_cycles}
        for pump1_wl in pump1_wls
    }
    idler_wl_dict_processed = {
        pump1_wl: {dc: np.zeros(len(pump2_wls)) for dc in duty_cycles}
        for pump1_wl in pump1_wls
    }
    ce_dict_std = {
        pump1_wl: {dc: np.zeros(len(pump2_wls)) for dc in duty_cycles}
        for pump1_wl in pump1_wls
    }
    ce_dict_best_for_each_ando_sweep = {
        pump1_wl: {dc: [] for dc in duty_cycles} for pump1_wl in pump1_wls
    }
    ce_dict_best_loc_for_each_ando_sweep = {
        pump1_wl: {dc: [] for dc in duty_cycles} for pump1_wl in pump1_wls
    }
    for pump_wl_idx, pump1_wl in enumerate(pump1_wls):
        for dc in duty_cycles:
            spectra = data[pump1_wl]["spectra"]
            for rep in range(num_reps):
                for idx_wl2 in range(len(pump2_wls)):
                    spectra_sgl_rep = np.squeeze(spectra[idx_wl2]).T
                    (
                        sig_wl_tmp,
                        ce_tmp,
                        idler_wl_tmp,
                    ) = extract_sig_wl_and_ce_single_spectrum(
                        spectra_sgl_rep,
                        [pump1_wl, pump2_wls[idx_wl2]],
                    )
                    ce_dict[pump1_wl][dc][idx_wl2] = -ce_tmp
                    sig_wl_dict[pump1_wl][dc][idx_wl2] = sig_wl_tmp
                    idler_wl_dict[pump1_wl][dc][idx_wl2] = idler_wl_tmp
                    ce_dict_processed[pump1_wl][dc][idx_wl2] = data_process_method(
                        ce_dict[pump1_wl][dc][idx_wl2]
                    )
                    ce_dict_std[pump1_wl][dc][idx_wl2] = np.std(
                        ce_dict[pump1_wl][dc][idx_wl2]
                    )
                    sig_wl_dict_processed[pump1_wl][dc][idx_wl2] = data_process_method(
                        sig_wl_dict[pump1_wl][dc][idx_wl2]
                    )
                    idler_wl_dict_processed[pump1_wl][dc][idx_wl2] = (
                        data_process_method(idler_wl_dict[pump1_wl][dc][idx_wl2])
                    )
            ce_dict_best_for_each_ando_sweep[pump1_wl][dc] = np.max(
                np.squeeze(ce_dict_processed[pump1_wl][dc])
            )
            ce_dict_best_loc_for_each_ando_sweep[pump1_wl][dc] = np.argmax(
                np.squeeze(ce_dict_processed[pump1_wl][dc])
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
