import numpy as np
from scipy.stats import permutation_test
import os
from typing import List, Tuple

from umcglib.froc import partial_auc, plot_multiple_froc, partial_auc, plot_multiple_roc


# Used for calculating the test statistics for the permutation test
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def print_datatypes_in_np_file(path):
    data = np.load(path, allow_pickle=True)
    for key in data.item():
        print(f"\n{key}:\n\t", end='', flush=True)
        value = data.item().get(key)
        if type(value) == int or type(value) == float:
            print(value)
        elif type(value) == dict:
            for i in value:
                print(i, value[i], flush=True)
                print(f"{i}: {value[i]}", flush=True)
        else:
            print(f"{value.shape} --- {type(value)}")
    p_auc_froc_all = list(data.item().get('p_auc_froc_all'))
    print(type(p_auc_froc_all), flush=True)


def get_mean_PAUCFROC_from_stats_old(path):
    data = np.load(path, allow_pickle=True)
    paucs_froc = data.item().get('p_auc_froc_all')
    return float(np.mean(paucs_froc))


def get_mean_PAUCFROC_from_stats_new(path):
    data = np.load(path, allow_pickle=True)
    pauc_froc = partial_auc(data.item().get('sensitivity'), data.item().get('fp_per_patient'))
    return pauc_froc


def get_from_np_dict(path, key, verbatim=False):
    data = np.load(path, allow_pickle=True)
    value = data.item().get(key)
    if verbatim:
        print(f"{key}: {value.shape} with type: {type(value)}")
    return value


def closest_fp_and_idx(false_positives: List, fpp_value: float):
    arr = np.asarray(false_positives)

    # before the target fpp_value the values are negative after they are positive
    diff_arr = arr - fpp_value

    # make all positive values really big
    diff_arr[diff_arr > 0] = 999

    # Now the index on the left of the closest number is chosen instead of the
    # possibility that it could be on the right
    i = (np.abs(diff_arr)).argmin()

    return arr[i], i


def interpolate_senss_to_one_fpp_range_for_folds(
    fold_senss: List[List],     # A list of lists with each list being the sensitivies for a fold
    fold_fps: List[List],       # A list of lists with each list being the false positives for a fold
    fp_arange: np.ndarray
):

    # Holds for each fold the number of false positives we are interested in.
    interpolated_senss = np.zeros(shape=(len(fold_senss), len(fp_arange)))

    for fold_idx, fold_fp in enumerate(fold_fps):
        for fp_idx, fp_point in enumerate(fp_arange):

            # for each fold and fp point find the closest fp idx.
            closest_fp, found_idx = closest_fp_and_idx(fold_fp, fp_point)

            # The found index should be used to retrieve the respective sensitivity
            respective_sensitivity = fold_senss[fold_idx][found_idx]

            # Assign the sensitivity to the array that
            interpolated_senss[fold_idx, fp_idx] = respective_sensitivity

    return interpolated_senss


def get_ci_low_and_high_paucs(
    sensitivities_per_fold: np.ndarray,         # 2D array (n_folds x fp_range) has sensitivities
    false_positives: np.ndarray,                # 1D array
    z: float = 1.96
):
    # Calculate the pAUC under the FROC for each fold
    n_folds = sensitivities_per_fold.shape[0]

    # Get the partial AUCs.
    paucs = np.asarray([partial_auc(list(sensitivities_per_fold[i, :]), list(false_positives)) for i in range(n_folds)])
    x = np.mean(paucs)
    n = len(paucs)
    s = np.std(paucs)

    return x - z * (s / n),  x + z * (s / n)


def collapse_sensitivies_per_fold_vs_fprange_to_cis(inter_senss_arr, z=1.96):

    # number of folds:
    n = inter_senss_arr.shape[0]

    cis_low = np.mean(inter_senss_arr, axis=0) - (z * (np.std(inter_senss_arr, axis=0) / np.sqrt(n)))
    cis_high = np.mean(inter_senss_arr, axis=0) + (z * (np.std(inter_senss_arr, axis=0) / np.sqrt(n)))

    return cis_low, cis_high


def plot_froc_on_stats():
    # Define the false positives range we are interested in.
    # Determines the resolution of the curve. A low stepsize creates a smoother curve
    fp_range = np.arange(0.1, 2.6, 0.001)

    # List of sensitivities, standard deviations and respective fp range per fold.
    mean_senss_all_folds, fps_all_folds, ci_senss_lo, ci_senss_hi = [], [], [], []

    ci_low_paucs, ci_high_paucs = [], []

    for paths_of_fold in paths_per_fold:
        
        fold_senss, fold_fps = [], []
        for stats_path in paths_of_fold:
            fold_senss.append(list(get_from_np_dict(stats_path, 'sensitivity')))
            fold_fps.append(list(get_from_np_dict(stats_path, 'fp_per_patient')))

        # Obtain 2D numpy array with measured sensitivities per fold for the defined false positive range
        inter_senss_arr = interpolate_senss_to_one_fpp_range_for_folds(
            fold_senss = fold_senss,
            fold_fps   = fold_fps,
            fp_arange  = fp_range,
        )
        
        # get singular low and high confidence interval for this model. There are multiple folds. So we can collapse it to a ci_low, mean and ci_high
        ci_low_pauc, ci_high_pauc = get_ci_low_and_high_paucs(inter_senss_arr, fp_range)
        ci_low_paucs.append(ci_low_pauc)
        ci_high_paucs.append(ci_high_pauc)

        input(np.max(np.mean(inter_senss_arr, axis=0)))

        mean_senss_all_folds.append(np.mean(inter_senss_arr, axis=0))
        fps_all_folds.append(fp_range)

        # These are the confidence intervals for the whole curve and not for the area under the curve
        ci_lows, ci_highs = collapse_sensitivies_per_fold_vs_fprange_to_cis(inter_senss_arr)

        # std_senss_lo.append(np.mean(inter_senss_arr, axis=0) - 2 * np.std(inter_senss_arr, axis=0))
        # std_senss_hi.append(np.mean(inter_senss_arr, axis=0) + 2 * np.std(inter_senss_arr, axis=0))
        ci_senss_lo.append(ci_lows)
        ci_senss_hi.append(ci_highs)

    fname_froc = f"figs_workspace/froc_combined_mu_std_v4.png"

    model_names = [f"R1", f"R4 IFFT", f"R4 U-Net Recon", f"R8 IFFT", f"R8 U-Net Recon"]

    plot_multiple_froc(
        sensitivities   = mean_senss_all_folds,
        fp_per_patient  = fps_all_folds,
        ci_low          = ci_senss_lo,
        ci_high         = ci_senss_hi,
        model_names     = model_names,
        log_x           = True,
        save_as         = fname_froc,
        xlims           = (0.1, 2.5),
        title           = 'FROC Lesion-Based Diagnosis on a Multi-Site Test Set',
        cis_legend_low  = None, #ci_low_paucs,
        cis_legend_high = None, #ci_high_paucs,
    )
    print(f"num patients in test set: {get_from_np_dict(stats_path, 'num_patients')}")
    print(f"Wrote multi froc plot to: {fname_froc}")


def plot_roc_on_stats():
    # Define the false positives rate range we are interested in.
    # Determines the resolution of the curve. A low stepsize creates a smoother curve
    fpr_range = np.arange(0.0, 1.0, 0.001)

    # List of sensitivities, standard deviations and respective fp range per fold.
    mean_tpr_all_folds, fpr_all_folds, std_tpr_lo, std_tpr_hi = [], [], [], []

    for paths_of_fold in paths_per_fold:
        
        fold_tpr, fold_fpr = [], []
        for idx, stats_path in enumerate(paths_of_fold):
            fold_tpr.append(list(get_from_np_dict(stats_path, 'roc_tpr')))
            fold_fpr.append(list(get_from_np_dict(stats_path, 'roc_fpr')))

        # Obtain 2D numpy array with measured sensitivities per fold for the defined false positive range
        inter_senss_arr = interpolate_senss_to_one_fpp_range_for_folds(
            fold_senss = fold_tpr,
            fold_fps   = fold_fpr,
            fp_arange  = fpr_range,
        )

        mus = np.mean(inter_senss_arr, axis=0)
        std = np.std(inter_senss_arr, axis=0)

        mean_tpr_all_folds.append(mus)
        fpr_all_folds.append(fpr_range)
        std_tpr_lo.append(mus - 2 * std)
        std_tpr_hi.append(mus + 2 * std)

    fname_roc = f"figs_workspace/roc_combined_mu_std_v2.png"
    plot_multiple_roc(
        tpr         = mean_tpr_all_folds,
        fpr         = fpr_all_folds,
        ci_low      = std_tpr_lo,
        ci_high     = std_tpr_hi,
        xlims       = (0.0, 100.0),
        ylims       = (0.0, 100.0),
        model_names = ["R1", "R4", "R4 U-Net Recon", "R8", "R8 U-Net Recon"],
        save_as     = fname_roc,
        title       = "Patient Based Diagnosis ROC on Test Set",
    )
    print(f"num patients in test set: {get_from_np_dict(stats_path, 'num_patients')}")
    print(f"Wrote multi froc plot to: {fname_roc}")


def do_permutation_test_on_stats():
    recon_dirs = ['R1_', 'R4_', 'R4_80_recon_r4', 'R8_', 'R8_81_recon_r8']

    mus_r1s, mus_r4s, mus_r4s_dl, mus_r8s, mus_r8s_dl = [], [], [], [], []
    for i in range(5):
        mus_r1s.append(get_mean_PAUCFROC_from_stats_new(os.path.join('train_output', '91_diag_t2_r1_norm', f"fold_{i}", 'test_preds', 'R1_', 'stats.npy')))
        mus_r4s.append(get_mean_PAUCFROC_from_stats_new(os.path.join('train_output', '91_diag_t2_r1_norm', f"fold_{i}", 'test_preds', 'R4_', 'stats.npy')))
        mus_r4s_dl.append(get_mean_PAUCFROC_from_stats_new(os.path.join('train_output', '91_diag_t2_r1_norm', f"fold_{i}", 'test_preds', 'R4_80_recon_r4', 'stats.npy')))
        mus_r8s.append(get_mean_PAUCFROC_from_stats_new(os.path.join('train_output', '91_diag_t2_r1_norm', f"fold_{i}", 'test_preds', 'R8_', 'stats.npy')))
        mus_r8s_dl.append(get_mean_PAUCFROC_from_stats_new(os.path.join('train_output', '91_diag_t2_r1_norm', f"fold_{i}", 'test_preds', 'R8_81_recon_r8', 'stats.npy')))

    print(f"\nmus_r1s: {[round(x, 2) for x in mus_r1s]}")
    print(f"mus_r4s: {[round(x, 2) for x in mus_r4s]}")
    print(f"mus_r4s_dl: {[round(x, 2) for x in mus_r4s_dl]}")
    print(f"mus_r8s: {[round(x, 2) for x in mus_r8s]}")
    print(f"mus_r8s_dl: {[round(x, 2) for x in mus_r8s_dl]}\n")

    if True:
        rng = np.random.default_rng(seed=124)
        if True:
            res = permutation_test((np.asarray(mus_r1s), np.asarray(mus_r4s)), statistic, vectorized=True, n_resamples=10000, alternative='greater', random_state=rng)
            print(f"mus_r1s vs mus_r4s")
            print(f"p-value: {round(res.pvalue, 3)}")
            # print(f"statistic: {round(res.statistic, 3)}")
            print()

            res = permutation_test((np.asarray(mus_r1s), np.asarray(mus_r4s_dl)), statistic, vectorized=True, n_resamples=10000, alternative='greater', random_state=rng)
            print(f"mus_r1s vs mus_r4s_dl")
            print(f"p-value: {round(res.pvalue, 3)}")
            # print(f"statistic: {round(res.statistic, 3)}")
            print()
        if True:
            res = permutation_test((np.asarray(mus_r1s), np.asarray(mus_r8s_dl)), statistic, vectorized=True, n_resamples=10000, alternative='greater', random_state=rng)
            print(f"mus_r1s vs mus_r8s_dl")
            print(f"p-value: {round(res.pvalue, 3)}")
            # print(f"statistic: {round(res.statistic, 3)}")
            print()
        if True:
            res = permutation_test((np.asarray(mus_r4s), np.asarray(mus_r8s)), statistic, vectorized=True, n_resamples=10000, alternative='greater', random_state=rng)
            print(f"mus_r4s vs mus_r8s")
            print(f"p-value: {round(res.pvalue, 3)}")
            # print(f"statistic: {round(res.statistic, 3)}")
            print()
        if True:
            res = permutation_test((np.asarray(mus_r4s_dl), np.asarray(mus_r4s)), statistic, vectorized=True, n_resamples=10000, alternative='greater', random_state=rng)
            print(f"mus_r4s vs mus_r4s_dl")
            print(f"p-value: {round(res.pvalue, 3)}")
            # print(f"statistic: {round(res.statistic, 3)}")
            print()
        if True:
            res = permutation_test((np.asarray(mus_r8s_dl), np.asarray(mus_r8s)), statistic, vectorized=True, n_resamples=10000, alternative='greater', random_state=rng)
            print(f"mus_r8s vs mus_r8s_dl")
            print(f"p-value: {round(res.pvalue, 3)}")
            # print(f"statistic: {round(res.statistic, 3)}")
            print()

################################################################################

if __name__ == '__main__':

    paths_r1 = [
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_0', 'test_preds', 'R1_', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_1', 'test_preds', 'R1_', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_2', 'test_preds', 'R1_', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_3', 'test_preds', 'R1_', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_4', 'test_preds', 'R1_', 'stats.npy'),    
    ]

    paths_r4 = [
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_0', 'test_preds', 'R4_', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_1', 'test_preds', 'R4_', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_2', 'test_preds', 'R4_', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_3', 'test_preds', 'R4_', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_4', 'test_preds', 'R4_', 'stats.npy'),    
    ]

    paths_r4_dl = [
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_0', 'test_preds', 'R4_80_recon_r4', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_1', 'test_preds', 'R4_80_recon_r4', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_2', 'test_preds', 'R4_80_recon_r4', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_3', 'test_preds', 'R4_80_recon_r4', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_4', 'test_preds', 'R4_80_recon_r4', 'stats.npy'),    
    ]

    paths_r8 = [
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_0', 'test_preds', 'R8_', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_1', 'test_preds', 'R8_', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_2', 'test_preds', 'R8_', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_3', 'test_preds', 'R8_', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_4', 'test_preds', 'R8_', 'stats.npy'),    
    ]

    paths_r8_dl = [
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_0', 'test_preds', 'R8_81_recon_r8', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_1', 'test_preds', 'R8_81_recon_r8', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_2', 'test_preds', 'R8_81_recon_r8', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_3', 'test_preds', 'R8_81_recon_r8', 'stats.npy'),
        os.path.join('train_output', '91_diag_t2_r1_norm', 'fold_4', 'test_preds', 'R8_81_recon_r8', 'stats.npy'),
    ]

    # Aggregate all paths per fold
    paths_per_fold       = [paths_r1, paths_r4, paths_r4_dl, paths_r8, paths_r8_dl]
    paths_per_fold_names = ["paths_r1", "paths_r4", "paths_r4_dl", "paths_r8", "paths_r8_dl"]

    # print_datatypes_in_np_file(paths_per_fold[0][0])
    # print(get_from_np_dict(paths_per_fold[0][0], 'average_precision'))
    # print(get_from_np_dict(paths_per_fold[0][0], 'p_auc_95_boot_0.1_2.5_ci_low'))
    # print(get_from_np_dict(paths_per_fold[0][0], 'p_auc_95_boot_0.1_2.5_ci_high'))
    # print(get_from_np_dict(paths_per_fold[0][0], 'auc_ci_low_roc'))
    # print(get_from_np_dict(paths_per_fold[0][0], 'auc_ci_high_roc'))


    plot_froc_on_stats()
    # plot_roc_on_stats()
    # do_permutation_test_on_stats()