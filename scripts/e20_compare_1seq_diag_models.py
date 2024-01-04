from json import load
from multiprocessing import set_start_method, get_context
import argparse
import SimpleITK as sitk
import time
import random
import numpy as np
import os
from fastMRI_PCa.utils.utils import does_table_exist
from optuna.samplers import TPESampler
import optuna
from tqdm import tqdm
from itertools import product

from functools import partial
from tensorflow.keras.models import load_model
from typing import List, Tuple

# from fastMRI_PCa.models import ssim_loss, ssim_metric, psnr_metric, weighted_binary_cross_entropy
# from fastMRI_PCa.data import resample, center_crop, normalize, undersample, binarize_s
from fastMRI_PCa.utils import get_rand_exp_decay_mask
from fastMRI_PCa.visualization import save_slice_3d, write_array2nifti_t2, save_array2d_to_image

from umcglib.utils import print_, read_yaml_to_dict, print_stats_np, list_from_file, get_random_str, apply_parallel, set_gpu
from umcglib.froc import calculate_froc, plot_multiple_froc, partial_auc
from umcglib.binarize import dynamic_threshold
from umcglib.plotting import plot_roc
from umcglib.losses import weighted_binary_cross_entropy
import umcglib.images as im


################################  README  ######################################
# OLD -  This contains plot functionality for training losses and metrics. This script
# should be called on a target directory with a .csv file with train results.
# Certain types of metrics and losses are detected and plotted. And also plotted
# against each other in the same plot on new validation data.


def parse_input_args():
    parser = argparse.ArgumentParser(description='Parse arguments for training a Reconstruction model')

    parser.add_argument('-fn',
                        '--fold_num',
                        type=int,
                        help='The current fold number to be run. Not the be confused with the total number of folds')

    parser.add_argument('-d',
                        '--dirs',
                        nargs='+',
                        type=str,
                        help='Directories with model training data (Each folder should contain a training_log.csv file). Statistics will be gathered from these folders.')

    parser.add_argument('-sb',
                        dest='show_blob',
                        action='store_true',
                        default=False,
                        help="Use this if you want a visualization of detected lesions as blobs to nifti files.")

    parser.add_argument('-opt',
                        dest='do_opt',
                        action='store_true',
                        default=False,
                        help="Use this flag if you want to determine values for the dynamic threshold and minimum confidence.")

    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help="Use this if the code should be run in debug mode. So paramters will be scaled down so that it runs much faster.")

    parser.add_argument('-c',
                        '--model_select_crit',
                        type=str,
                        default='val_loss',
                        help='Model selection critertia. Can be validation loss or partial AUC from FROC (pAUC_FROC). Options: "val_loss" or "pAUC_FROC".')

    parser.add_argument('-f',
                        '--fname',
                        type=str,
                        default='multi_froc',
                        help='File name of the output figure. It will be stored into the figs_workspace folder.')

    parser.add_argument('-nt',
                        '--num_trials',
                        type=int,
                        default=5,
                        help='The number of hyper optimization trials that will be tried to be performed within one execution of the script. This script is meant to run for a long time and restarted without the --is_new_study flag if futher trials are desired.')

    args = parser.parse_args()

    assert(args.model_select_crit in ["val_loss", "pAUC_FROC"]), "Ensure the model selection criteria is a valid option."

    print_(f"args:\n {args}")

    return args


################################################################################


def save_some_blobs_to_nifti(
    n_imgs,
    dyn_thresh,
    min_conf,
    x_true_val,
    y_true_val,
    y_pred_val,
    diag_dir
):
    if DEBUG:
        blob_dir = f"figs_workspace/debug/{diag_dir}"
    else:
        blob_dir = f"figs_workspace/{diag_dir}"
    os.makedirs(blob_dir, exist_ok=True)
    acc = "None"

    if diag_dir in ["67_diag_t2_ifft_cc_r1"]:
        acc = 1
    if diag_dir in ["68_diag_t2_ifft_cc_r4", "70_diag_t2_cc_r4dl"]:
        acc = 4
    if diag_dir in ["69_diag_t2_ifft_cc_r8", "71_diag_t2_cc_r8dl"]:
        acc = 8

    for idx_pat in range(n_imgs):    
        label_map, confidences = dynamic_threshold(
            prediction=y_pred_val[idx_pat],
            min_voxels=25,
            outer_deadzone=3,
            dynamic_threshold_factor=dyn_thresh,
            minimum_confidence=min_conf
        )

        print_(f"mri idx: {idx_pat}")
        for key in confidences:
            print_(f"{key}: {confidences[key]}")

        lbl_fname = os.path.join(blob_dir, f"pat{idx_pat}_acc{acc}_lblpred_th{dyn_thresh}_mc{min_conf}.nii.gz")
        lbl_s = sitk.GetImageFromArray((label_map.T).astype(np.float32))
        lbl_s.SetSpacing([0.5, 0.5, 3.0])
        sitk.WriteImage(lbl_s, lbl_fname)
        print_(f"Wrote nifti blob map to: {lbl_fname}")

        seg_fname =  os.path.join(blob_dir, f"pat{idx_pat}_acc{acc}_seg_th{dyn_thresh}_mc{min_conf}.nii.gz")
        seg_img_s = sitk.GetImageFromArray((y_true_val[idx_pat].T).astype(np.float32))
        seg_img_s.SetSpacing([0.5, 0.5, 3.0])
        sitk.WriteImage(seg_img_s, seg_fname)
        print_(f"Wrote nifti seg map to: {seg_fname}")

        pred_fname =  os.path.join(blob_dir, f"pat{idx_pat}_acc{acc}_pred_th{dyn_thresh}_mc{min_conf}.nii.gz")
        pred_img_s = sitk.GetImageFromArray((y_pred_val[idx_pat].T).astype(np.float32))
        pred_img_s.SetSpacing([0.5, 0.5, 3.0])
        sitk.WriteImage(pred_img_s, pred_fname)
        print_(f"Wrote nifti pred map to: {pred_fname}")

        inp_fname =  os.path.join(blob_dir, f"pat{idx_pat}_acc{acc}_inp_th{dyn_thresh}_mc{min_conf}.nii.gz")
        inp_img_s = sitk.GetImageFromArray((x_true_val[idx_pat].T).astype(np.float32))
        inp_img_s.SetSpacing([0.5, 0.5, 3.0])
        sitk.WriteImage(inp_img_s, inp_fname)
        print_(f"Wrote nifti input map to: {inp_fname}")
        
        print_(f"num lesions detected: {len(confidences)} according to dynamic threshold: {dyn_thresh} with conf: {min_conf}")
        print_(f"-- done with blob maps mri{idx_pat} -- ")


def load_or_create_study(
    is_new_study: bool,
    study_dir: str,
):
    # Create an optuna if it does not exist.
    storage = f"sqlite:///{study_dir}/{DB_FNAME}"
    if is_new_study:
        print_(f"Creating a NEW study. With name: {storage}")
        study = optuna.create_study(storage=storage,
                                    study_name=study_dir,
                                    direction='maximize',
                                    sampler=TPESampler(n_startup_trials=N_STARTUP_TRIALS))
    else:
        print_(f"LOADING study {storage} from database file.")
        study = optuna.load_study(storage=storage,
                                  study_name=study_dir)

    return study


def p_auc_froc_obj(trial, y_true_val, y_pred_val):

    dyn_thresh = trial.suggest_float('dyn_thresh', 0.0, 1.0)
    min_conf   = trial.suggest_float('min_conf', 0.0, 1.0)

    stats = calculate_froc(y_true=y_true_val,
                           y_pred=y_pred_val,
                           preprocess_func=dynamic_threshold,
                           dynamic_threshold_factor=dyn_thresh,
                           minimum_confidence=min_conf)
    
    sens, fpp = stats['sensitivity'], stats['fp_per_patient']
    p_auc_froc = partial_auc(sens, fpp, low=0.1, high=2.5)

    print_(f"dyn_threshold: {dyn_thresh}, min_conf{min_conf}")
    print_(f"Trial {trial.number} pAUC FROC: {p_auc_froc}")

    return p_auc_froc


def preprocess(
    t2,
    window_size,
    spacing=(0.5, 0.5, 3.),
    crop=True, 
    to_numpy=True,
    seg=None,
    norm="znorm",
    acceleration=None,
    centre_sampling=None
):

    if acceleration != None: 
        mask = get_rand_exp_decay_mask(
            width           = t2.GetSize()[0],
            height          = t2.GetSize()[1],
            sampling        = 1.0/acceleration,
            centre_sampling = centre_sampling,
            seed            = SEED,
            verbatim        = False
        )
        t2 = im.undersample_kspace(t2, mask)

    # Preprocess the ADC image, then resample the rest to it
    t2 = im.resample(
        image=t2, 
        min_shape=window_size, 
        method=sitk.sitkLinear, 
        new_spacing=spacing)

    if crop:
        t2 = im.center_crop(t2, window_size)

    if seg is not None:
        seg = im.resample_to_reference(seg, t2, sitk.sitkNearestNeighbor)

    # Return the SITK preprocessed images if requested
    if not to_numpy: 
        return t2, seg

    # Extract numpy arrays
    t2  = sitk.GetArrayFromImage(t2).T

    # Stack the inputs, add new axis to seg
    img_n = t2[..., np.newaxis]
    
    # Z-Normalize at crop level
    if norm == "znorm":
        img_n -= np.mean(img_n)
        img_n /= np.std(img_n)
    
    if norm == "rescale_0_1":
        img_n = (1.0*(img_n - np.min(img_n))/np.ptp(img_n))
    
    # Extract batch for the segmentation if provided
    if seg is not None: 
        seg = sitk.GetArrayFromImage(seg).T
        seg = (seg[..., None] > 0.5) * 1.

    return img_n, seg.astype(np.float32)


def read_and_preprocess(
    paths, 
    window_size=None, 
    spacing=(0.5, 0.5, 3.), 
    crop=True,
    to_numpy=True,
    norm="znorm",
    acceleration=None,
    centre_sampling=None
):
    # Explode the inputs
    if len(paths) == 1:
        t2_path = paths[0]
        seg_path = None
        seg = None
    if len(paths) == 2:
        t2_path, seg_path = paths

    # Read each image
    t2 = sitk.ReadImage(t2_path, sitk.sitkFloat32)
    if seg_path is not None:
        seg = sitk.ReadImage(seg_path, sitk.sitkFloat32)

    return preprocess(t2,
                      seg=seg,
                      window_size=window_size,
                      spacing=spacing,
                      crop=crop,
                      to_numpy=to_numpy,
                      norm=norm,
                      acceleration=acceleration,
                      centre_sampling=centre_sampling)


def load_data(
    t2_files: List[str],
    seg_files: List[str],
    sampling_window: List[int],
    spacing: Tuple[float,float,float],
    norm: str,
    acceleration: float,
    centre_sampling: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]: 

    # Zip the list and read + preprocess in parallel
    print_(f"> Reading images.. {len(t2_files)}")
    combined_prepped = []

    num_read_splits = NUM_READ_SPLITS
    num_read_at_once = len(seg_files) // num_read_splits + 1
    combined_list = list(zip(t2_files, seg_files))
    combined_prepped = []
    for i in range(num_read_splits):
        _from = i * num_read_at_once
        _to = (i + 1) * num_read_at_once
        print_("From idx:", _from, " To idx:", _to)
        cur_paths = combined_list[_from:_to]
        combined_prepped += apply_parallel(
            cur_paths,
            read_and_preprocess,
            N_CPUS,
            window_size=sampling_window,
            crop=True,
            spacing=spacing,
            norm=norm,
            acceleration=acceleration,
            centre_sampling=centre_sampling
        )
        
    prepped_imgs, prepped_segs = [], []
    for i, s in combined_prepped:
        prepped_imgs.append(i)
        prepped_segs.append(s)

    print_(f"Data Loaded with stats:")
    print_(f"\tImgs num: {len(prepped_imgs)}")
    print_(f"\tSegs num: {len(prepped_segs)}")
    print_stats_np(prepped_imgs[0], "First observation Imgs:")
    print_stats_np(prepped_segs[0], "First observation Segs:")
    
    return prepped_imgs, prepped_segs


def predict_ensemble(input_n: np.ndarray, models: list, weight_map: np.ndarray = None):
    # Predict twice with each model. The first time predict on the normal image, 
    # then predict on the mirrored version of the image (along the X axis).
    predictions = [m.predict(input_n, batch_size=BATCH_SIZE) for m in models]
    predictions +=[m.predict(input_n[::-1], batch_size=BATCH_SIZE)[::-1] for m in models]

    # Aggregate the results by taking the mean
    averaged_prediction = np.mean(predictions, axis=0)

    if weight_map is not None:
        averaged_prediction[0, ..., 0] *= weight_map

    return averaged_prediction


def predict_sliding(
    input_n: np.ndarray,
    models: list,   #Model objects
    window_size: Tuple[int,int,int],
    window_norm: str,
    weight_map: np.ndarray = None,
):
    print_("> Making sliding window predictions...")
    # Calculate the possible crop origins to cover the full image with overlap
    possible_offsets = [
        a_len - c_len for a_len, c_len in zip(input_n.shape, window_size)]

    x_list, y_list, z_list = [], [], []
    x, y, z = 0,0,0

    while x <= possible_offsets[0]:
        x_list += [x]
        x += max(min(window_size[0]//10, possible_offsets[0]-x), 1)
    while y <= possible_offsets[1]:
        y_list += [y]
        y += max(min(window_size[1]//10, possible_offsets[1]-y), 1)
    while z <= possible_offsets[2]:
        z_list += [z]
        z += max(min(window_size[2]//4, possible_offsets[2]-z), 1)
    
    all_crop_origins = list(product(x_list, y_list, z_list))

    # Sliding window prediction
    full_prediction = np.zeros(input_n.shape[:3])
    summed_weights = np.zeros_like(full_prediction)
    for x, y, z in tqdm(all_crop_origins, desc="Collecting predictions..."):
        img_crop = np.copy(input_n[
            np.newaxis,
            x:x+window_size[0],
            y:y+window_size[1],
            z:z+window_size[2]])
    
        if window_norm == "znorm":
            img_crop -= np.mean(img_crop)
            img_crop /= np.std(img_crop)
        
        if window_norm == "rescale_0_1":
            img_crop = (1.0*(img_crop - np.min(img_crop))/np.ptp(img_crop))

        # Softmax predictions are collected for each of the CV models
        crop_pred = predict_ensemble(img_crop, models, weight_map)

        full_prediction[
            x:x+window_size[0],
            y:y+window_size[1],
            z:z+window_size[2]
            ] += crop_pred.squeeze()

        if weight_map == None:
            weight_map = 1.0

        summed_weights[
            x:x+window_size[0],
            y:y+window_size[1],
            z:z+window_size[2]
            ] += weight_map

    full_prediction /= summed_weights
    print_("> Done collecting predictions:", full_prediction.shape)
    return np.expand_dims(full_prediction, axis=3).astype(np.float32)


def recon_model_predict(prepped_imgs: List, t2_recon_dir: str, fold_num: int):
        print_("Predicting with the reconstruction model on understampled T2w images. (validation loss model)")
        t2_model_path = f"models/{t2_recon_dir}/best-direct-fold{fold_num}_val_loss.h5"
        recon_model = load_model(t2_model_path, compile=False)
        recons = np.squeeze(recon_model.predict(np.stack(prepped_imgs, 0), batch_size=BATCH_SIZE))
        prepped_imgs = [recons[mri_idx] for mri_idx in range(recons.shape[0])]
        prepped_imgs = np.expand_dims(prepped_imgs, axis=4)
        recon_model.summary()
        return prepped_imgs


################################################################################
SEED = 1234

if __name__ == '__main__':

    args = parse_input_args()
    FOLD_NUM = args.fold_num
    DIAG_DIRS = args.dirs
    MODEL_SELECT_CRIT = args.model_select_crit
    T2_LIST_PATH  = "data/path_lists/pirads_4plus/current_t2w.txt"
    DWI_LIST_PATH = "data/path_lists/pirads_4plus/current_dwi.txt"
    ADC_LIST_PATH = "data/path_lists/pirads_4plus/current_adc.txt"
    SEG_LIST_PATH = "data/path_lists/pirads_4plus/seg_new.txt"
    IDX_YAML_PATH = r"data/path_lists/pirads_4plus/train_val_test_idxs.yml" 
    VALSETKEY = "val_set0"
    PERC_LOAD = 1.0
    DB_FNAME = "dyn_thres_min_conf_opt_v1_r16n.db"
    N_STARTUP_TRIALS = 10
    N_CPUS = 6
    NUM_READ_SPLITS = 2
    BATCH_SIZE = 1
    DEBUG = True if args.debug else False
    set_gpu(gpu_idx=1)

    if DEBUG:
        N_CPUS = 4
        PERC_LOAD = 0.08

    # For multiprocessing
    set_start_method("spawn")

    model_sensitivities, model_false_positives = [], []

    for model_idx, diag_dir in enumerate(DIAG_DIRS):

        train_dir = f"train_output/{diag_dir}/fold_{FOLD_NUM}/"
        diag_params_path = os.path.join(train_dir, "params.yml")
        diag_params = read_yaml_to_dict(diag_params_path)

        # Load Reconstruction model parameters
        if diag_params['t2_recon_dir'] != None:
            recon_params_path = f"train_output/{diag_params['t2_recon_dir']}/params.yml"
            recon_params = read_yaml_to_dict(recon_params_path)
            t2_model_path = f"models/{diag_params['t2_recon_dir']}/best-direct-fold{FOLD_NUM}_val_loss.h5"
        else:
            recon_params = diag_params
            t2_model_path = None
        
        # Indexes
        indexes = read_yaml_to_dict(IDX_YAML_PATH)
        val_idxs = indexes[VALSETKEY]
        if DEBUG:
            val_idxs = val_idxs[:int(PERC_LOAD * len(val_idxs))]
        print_(f"validation set indexes {val_idxs}\nWith length: {len(val_idxs)}")
        
        # Load data
        t2_files = [l.strip() for l in open(T2_LIST_PATH)]
        t2_files = list(np.asarray(t2_files)[val_idxs])
        seg_files = [l.strip() for l in open(SEG_LIST_PATH)]
        seg_files = list(np.asarray(seg_files)[val_idxs])

        prepped_imgs, prepped_segs = load_data(
            t2_files        = t2_files,
            seg_files       = seg_files,
            sampling_window = list(diag_params['sampling_window']),
            spacing         = tuple(diag_params['spacing']),
            norm            = recon_params['norm'],
            acceleration    = recon_params['acceleration'],
            centre_sampling = recon_params['centre_sampling']
        )

        # Predict with Reconstruction model
        if diag_params['t2_recon_dir'] != None:
            prepped_imgs = recon_model_predict(prepped_imgs, diag_params['t2_recon_dir'], FOLD_NUM)

        X_true = np.stack(prepped_imgs, axis=0)
        Y_true = np.stack(prepped_segs, axis=0)
        Y_pred = np.zeros(X_true.shape, dtype=np.float32)

        # Predict with Diagnostic model
        diag_model_path = os.path.join(train_dir, "models", f"best_val_loss.h5")
        diag_model = load_model(diag_model_path, compile=False)
        diag_model.summary()

        for img_idx in range(X_true.shape[0]):
            Y_pred[img_idx] = predict_sliding(
                input_n     = X_true[img_idx],
                models      = [diag_model],
                window_size = tuple(diag_params['window_size']),
                window_norm = diag_params['norm']
            )

        if DEBUG:   # visualize some training data and predictions
            for mri_idx in range(Y_pred.shape[0]):
                fname_debug = f"temp/idx{mri_idx}_y_pred.nii.gz"
                pred = sitk.GetImageFromArray(np.squeeze(Y_pred[mri_idx]).T)
                sitk.WriteImage(pred, fname_debug)
                print_(f"Wrote to: {fname_debug}")

                fname_debug = f"temp/idx{mri_idx}_x_true.nii.gz"
                x = sitk.GetImageFromArray(np.squeeze(X_true[mri_idx]).T)
                sitk.WriteImage(x, fname_debug)
                print_(f"Wrote to: {fname_debug}")

                fname_debug = f"temp/idx{mri_idx}_y_true.nii.gz"
                y = sitk.GetImageFromArray(np.squeeze(Y_true[mri_idx]).T)
                sitk.WriteImage(y, fname_debug)
                print_(f"Wrote to: {fname_debug}")

        # Data preparation
        X_true = np.squeeze(X_true)
        Y_true = np.squeeze(Y_true)
        Y_pred = np.squeeze(Y_pred)

        print_stats_np(X_true, "x true")

        # Create/load a study for opt of dynamic threshold and min confidence.
        if args.do_opt:
            study_dir = f"sqliteDB/optuna_dbs"
            table_exists = does_table_exist('trials', f"{study_dir}/{DB_FNAME}")
            study = load_or_create_study(is_new_study=not table_exists, study_dir=study_dir)
    
        # Calculate FROC
        dyn_thresh = 0.75
        min_conf = 0.15
        stats = calculate_froc(y_true=Y_true,
                               y_pred=Y_pred,
                               preprocess_func=dynamic_threshold,
                               dynamic_threshold_factor=dyn_thresh,
                               minimum_confidence=min_conf)
        
        # Visualize blobs with dynamic threshold and min confidence
        if args.show_blob:
            n_obs = len(X_true) if DEBUG else 15
            save_some_blobs_to_nifti(n_obs, dyn_thresh, min_conf, X_true, Y_true, Y_pred, diag_dir)

        if not args.do_opt:
            roc_path = f"figs_workspace/debug/{args.fname}_roc_dd{diag_dir}.png" if DEBUG else f"figs_workspace/{args.fname}_roc_dd{diag_dir}.png"
            plot_roc(
                roc_tpr=stats['roc_tpr'],
                roc_fpr=stats['roc_fpr'],
                save_path=roc_path,
                roc_auc=stats['patient_auc'],
                title = f"ROC - {diag_dir}"
            )
        
        sens, fpp = stats['sensitivity'], stats['fp_per_patient']
        model_sensitivities.append(sens)
        model_false_positives.append(fpp)

        p_auc = partial_auc(sens, fpp, low=0.1, high=2.5)

        # Try to find the best value for the dynamic threshold and min_confidence
        if args.do_opt:
            opt_func = lambda trail: p_auc_froc_obj(trail, Y_true, Y_pred)
            study.optimize(opt_func, n_trials=args.num_trials)
    
    print_(model_sensitivities)
    print_(model_false_positives)

    fname_froc = f"figs_workspace/debug/{args.fname}_froc.png" if DEBUG else f"figs_workspace/{args.fname}_froc.png"
    plot_multiple_froc(
        sensitivities=model_sensitivities,
        fp_per_patient=model_false_positives,
        model_names=args.dirs,
        log_x=True,
        save_as=fname_froc,
        xlims=(0.1, 2.5)
    )
    print_(f"Wrote multi froc plot to: {fname_froc}")
    
    print_("--- DONE ---")