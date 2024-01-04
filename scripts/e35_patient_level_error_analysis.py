import numpy as np
import os
import pandas as pd
import csv


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


def get_from_np_dict(path, key, verbatim=False):
    data = np.load(path, allow_pickle=True)
    value = data.item().get(key)
    if verbatim:
        print(f"{key}: {value.shape} with type: {type(value)}")
    return value



def get_error_test_set(path, verbatim = False):

    labl = get_from_np_dict(path, "roc_patient_level_label")
    conf = get_from_np_dict(path, "roc_patient_level_conf")

    pat_labels, pat_confs, errors = [], [], []
    for pat_idx in range(len(labl)):
        pat_labl = labl[pat_idx]
        pat_pred = conf[pat_idx]
        error = round(pat_labl - pat_pred, 3)
        
        pat_labels.append(pat_labl)
        pat_confs.append(pat_pred)
        errors.append(error)

        if verbatim:
            print(f"Pat: {pat_idx}\nLabl: {pat_labl}\nConf: {pat_pred}\nError: {error}\n")
            input(f"Press any key to continue...")

    return pat_labels, pat_confs, errors


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

    pat_labels, pat_confs_r1, errors_r1     = get_error_test_set(paths_r1[0], False)
    pat_labels, pat_confs_r4dl, errors_r4dl = get_error_test_set(paths_r4_dl[0], False)

    data = {
        "pat_labels" : pat_labels,
        "pat_confs_r1" : pat_confs_r1,
        "pat_confs_r4dl": pat_confs_r4dl,
        "error_r1" : errors_r1,
        "error_r4dl" : errors_r4dl,
        "diff_error_r1_vs_r4dl" : list(np.asarray(errors_r1) - np.asarray(errors_r4dl))
    }

    csv_path = os.path.join('stats', f'error_analysis_e35.csv')
    df = pd.DataFrame(data=data)
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    print(f">Wrote to: {csv_path}")