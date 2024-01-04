from multiprocessing import set_start_method
import argparse
import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm
from itertools import product

from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization
from typing import List, Tuple
import glob
import copy

from fastMRI_PCa.utils.k_space_masks import get_rand_exp_decay_mask_ac_matrix

from umcglib.utils import print_, read_yaml_to_dict, print_stats_np, apply_parallel, set_gpu
from umcglib.froc import calculate_froc, plot_multiple_froc, partial_auc
from umcglib.binarize import dynamic_threshold
from umcglib.plotting import plot_roc, write_array2nifti_t2
import umcglib.images as im


################################  README  ######################################
# NEW - This script will load a diagnostic T2 model trained on R1 images.
# It will plot the FROC curve of different accelerations given to this model.
# 
# e23_comp_t2_accs_to_r1.py = Experiment 23 compare performance of a diagnostic
# model trained on T2W images with different accelerations to a model trained
# on R1.

def parse_input_args():
    parser = argparse.ArgumentParser(description='Parse arguments for training a Reconstruction model')

    parser.add_argument('-fn',
                        '--fold_num',
                        type=int,
                        help='The current diagnostic fold number to be run. Not the be confused with the total number of folds')

    parser.add_argument('-d',
                        '--r1_dir',
                        type=str,
                        help='Directory where a T2 diagnostic dual attention net model is trained on unaccelerated images. (R1n)')

    parser.add_argument('-accs',
                        '--accelerations',
                        nargs='+',
                        type=float,
                        help='A list of accelerations that need to be plotted against R1. (R1 is acceleration factor 1.0 Unaccelerated).')

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

    parser.add_argument('-rds',
                        '--recon_dirs',
                        nargs='+',
                        type=str,
                        help='A list of directories where reconstruction models are stored.')

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
    centre_sampling=None,
    t2_nifti_path=None,
    dicom_db_path: str = r'sqliteDB/dicoms.db',
):

    t2_og = copy.deepcopy(t2)       # t2 original image (unaccelerated)

    if acceleration != None: 
        mask = get_rand_exp_decay_mask_ac_matrix(
            width           = t2.GetSize()[0],
            height          = t2.GetSize()[1],
            sampling        = 1/acceleration,
            centre_sampling = centre_sampling,
            seed            = SEED,
            exp_scale       = 0.4,      # determined emperically
            nifti_path      = t2_nifti_path,
            dicom_db_path   = dicom_db_path,
            tablename       = 'dicom_headers_v2',
            verbatim        = False,
        )
        t2 = im.undersample_kspace(t2, mask)

    t2 = im.resample(
        image       = t2, 
        min_shape   = window_size, 
        method      = sitk.sitkLinear, 
        new_spacing = spacing)

    t2_og = im.resample(
        image       = t2_og,
        min_shape   = window_size,
        method      = sitk.sitkLinear, 
        new_spacing = spacing)

    if crop:
        t2    = im.center_crop(t2, window_size)
        t2_og = im.center_crop(t2_og, window_size)

    if seg is not None:
        seg = im.resample_to_reference(seg, t2, sitk.sitkNearestNeighbor)

    # Return the SITK preprocessed images if requested
    if not to_numpy: 
        return t2, seg

    # Extract numpy arrays
    t2    = sitk.GetArrayFromImage(t2).T
    t2_og = sitk.GetArrayFromImage(t2_og).T

    # Stack the inputs, add new axis to seg
    img_n = t2[..., np.newaxis]
    t2_og = t2_og[..., np.newaxis]
    
    # Z-Normalize at crop level
    if norm == "znorm":
        img_n -= np.mean(img_n)
        img_n /= np.std(img_n)
        t2_og -= np.mean(t2_og)
        t2_og /= np.std(t2_og)
    
    if norm == "rescale_0_1":
        img_n = (1.0*(img_n - np.min(img_n))/np.ptp(img_n))
        t2_og = (1.0*(t2_og - np.min(t2_og))/np.ptp(t2_og))
    
    # Extract batch for the segmentation if provided
    if seg is not None: 
        seg = sitk.GetArrayFromImage(seg).T
        seg = (seg[..., None] > 0.5) * 1.

    return img_n, seg.astype(np.float32), t2_og


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

    return preprocess(
        t2              = t2,
        seg             = seg,
        window_size     = window_size,
        spacing         = spacing,
        crop            = crop,
        to_numpy        = to_numpy,
        norm            = norm,
        acceleration    = acceleration,
        centre_sampling = centre_sampling,
        t2_nifti_path   = t2_path
    )


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
        
    prepped_imgs, prepped_segs, not_undersampled_imgs = [], [], []
    for i, s, nu in combined_prepped:
        prepped_imgs.append(i)
        prepped_segs.append(s)
        not_undersampled_imgs.append(nu)

    print_(f"Data Loaded with stats:")
    print_(f"\tImgs num: {len(prepped_imgs)}")
    print_(f"\tSegs num: {len(prepped_segs)}")
    print_stats_np(prepped_imgs[0], "First observation Imgs:")
    print_stats_np(prepped_segs[0], "First observation Segs:")
    
    return prepped_imgs, prepped_segs, not_undersampled_imgs


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


def recon_model_predict_slid(
    prepped_imgs: List,
    t2_recon_dir: str,
):
        print_("Predicting with the reconstruction model on understampled T2w images. (validation loss model)")

        t2_model_path = f"train_output/{t2_recon_dir}/fold{0}/models/best_loss_fold{0}.h5"
        recon_model = load_model(t2_model_path, compile=False)

        for img_idx in range(len(prepped_imgs)):
            prepped_imgs[img_idx] = predict_sliding(
                input_n     = np.squeeze(prepped_imgs[img_idx]),
                models      = [recon_model],
                window_size = (160, 160, 16),
                window_norm = "rescale_0_1"
            )

        return prepped_imgs


def recon_model_predict(
    prepped_imgs: List,
    t2_recon_dir: str,
    recon_fold: int = 0
):
    print_("Predicting with the reconstruction model on understampled T2w images. (validation loss model)")
    t2_model_path = f"train_output/{t2_recon_dir}/fold{recon_fold}/models/best_loss_fold{recon_fold}.h5"
    recon_model = load_model(t2_model_path, compile=False)
    return recon_model.predict(np.stack(prepped_imgs, axis=0).squeeze())


def get_accelerations_from_recon_dirs(recon_dirs: List[str]):
    return [f"{x.split('_')[2]}dl" for x in recon_dirs]


def save_predictions_to_file(
    unaccelerated,
    usampled,
    recons,
    segs_crop,
    segs_pred_crop,
    acceleration,
    save_dir,
    seg_crop_s_list #List of sitk images
):
    for mri_idx in range(len(usampled)):

        fname_temp = os.path.join(save_dir, f"{mri_idx}_1unaccelerated_inp_R{acceleration}.nii.gz")
        img = sitk.GetImageFromArray(np.squeeze(unaccelerated[mri_idx]).T)
        sitk.WriteImage(img, fname_temp)
        print_(f"Wrote to: {fname_temp}")

        fname_temp = os.path.join(save_dir, f"{mri_idx}_2usampled_R{acceleration}.nii.gz")
        img = sitk.GetImageFromArray(np.squeeze(usampled[mri_idx]).T)
        sitk.WriteImage(img, fname_temp)
        print_(f"Wrote to: {fname_temp}")

        fname_temp = os.path.join(save_dir, f"{mri_idx}_3recons_R{acceleration}.nii.gz")
        img = sitk.GetImageFromArray(np.squeeze(recons[mri_idx]).T)
        img.CopyInformation(seg_crop_s_list[mri_idx])
        sitk.WriteImage(img, fname_temp)
        print_(f"Wrote to: {fname_temp}")

        fname_temp = os.path.join(save_dir, f"{mri_idx}_4segs_crop_R{acceleration}.nii.gz")
        img = sitk.GetImageFromArray(np.squeeze(segs_crop[mri_idx]).T)
        img.CopyInformation(seg_crop_s_list[mri_idx])
        sitk.WriteImage(img, fname_temp)
        print_(f"Wrote to: {fname_temp}")

        fname_temp = os.path.join(save_dir, f"{mri_idx}_5segs_pred_crop_R{acceleration}.nii.gz")
        img = sitk.GetImageFromArray(np.squeeze(segs_pred_crop[mri_idx]).T)
        img.CopyInformation(seg_crop_s_list[mri_idx])
        sitk.WriteImage(img, fname_temp)
        print_(f"Wrote to: {fname_temp}\n")


def calc_froc_on_new_data(
    t2_files: list,
    seg_files: list,
    diag_model,
    diag_params: dict,
    t2_recon_dir: str = None,
    acceleration: float = 1.0,
):
    norm = diag_params['norm']
    centre_sampling = diag_params['centre_sampling']

    if t2_recon_dir != None:
        recon_params_path = f"train_output/{t2_recon_dir}/fold{0}/train_recon_config.yml"
        recon_params      = read_yaml_to_dict(recon_params_path)
        norm              = recon_params['normalization']
        acceleration      = float(recon_params['acceleration'])
        centre_sampling   = recon_params['centre_sampling']

    # Make dir for validation en test predictions
    test_dir = os.path.join(R1_TRAIN_DIR, "test_preds")
    os.makedirs(test_dir, exist_ok=True)
    # Recon dir path
    t2_recon_dir = "" if t2_recon_dir == None else t2_recon_dir
    recon_test_preds_dir = os.path.join(test_dir, f"R{int(acceleration)}_" + t2_recon_dir)
    os.makedirs(recon_test_preds_dir, exist_ok=True)

    # Emperically determined
    dyn_thresh = 0.75
    min_conf = 0.15

    # count the number of test predictions in the test directory
    num_files = len(glob.glob(os.path.join(recon_test_preds_dir, "*.nii.gz")))

    if num_files//3 != len(t2_files):

        # Load the data as (256, 256, 16)
        usampled, segs, uaccel = load_data(
            t2_files        = t2_files,
            seg_files       = seg_files,
            sampling_window = [256, 256, 16],   # hardcodec because we do not always have access to recon_params
            spacing         = tuple(diag_params['spacing']),
            norm            = norm,
            acceleration    = acceleration,
            centre_sampling = centre_sampling
        )
        num_obs = len(usampled)

        # Reconstruction on (256, 256, 16)
        if t2_recon_dir != None and t2_recon_dir != "":
            recon = recon_model_predict(usampled, t2_recon_dir)
            print(f"recon len: {len(recon)} and shape {recon[0].shape}")

        uaccel   = np.stack(uaccel, axis=0)
        usampled = np.stack(usampled, axis=0)
        segs     = np.stack(segs, axis=0)

        # should be (180, 180, 16, 1)
        crop_shape = (diag_configs['sampling_window'][0], diag_configs['sampling_window'][1], 16, 1)
        recons_crop    = np.zeros((num_obs,) + crop_shape)
        segs_pred_crop = np.zeros((num_obs,) + crop_shape)
        segs_crop      = np.zeros((num_obs,) + crop_shape)
        usampled_crop  = np.zeros((num_obs,) + crop_shape) 

        # Reconstruction Center cropping
        if t2_recon_dir != None and t2_recon_dir != "":
            for i in range(num_obs):
                recon_s        = sitk.GetImageFromArray(np.squeeze(recon[i]).T)
                recon_crop_s   = im.center_crop(recon_s, shape = crop_shape)
                recon_crop_n   = sitk.GetArrayFromImage(recon_crop_s).T
                recons_crop[i] = np.expand_dims(recon_crop_n, axis=3)

        # Segs Centre cropping
        seg_crop_s_list = []
        for i in range(num_obs):
            seg_s        = sitk.GetImageFromArray(np.squeeze(segs[i]).T)
            seg_crop_s   = im.center_crop(seg_s, shape = crop_shape)
            seg_crop_n   = sitk.GetArrayFromImage(seg_crop_s).T
            segs_crop[i] = np.expand_dims(seg_crop_n, axis=3)
            seg_crop_s_list.append(seg_crop_s)

        # undersampled crop
        for i in range(num_obs):
            usampled_s       = sitk.GetImageFromArray(np.squeeze(usampled[i]).T)
            usampled_crop_s  = im.center_crop(usampled_s, shape = crop_shape)
            usampled_crop_n  = sitk.GetArrayFromImage(usampled_crop_s).T
            usampled_crop[i] = np.expand_dims(usampled_crop_n, axis=3)

        # Predict with detection model using a sliding window
        if t2_recon_dir != None and t2_recon_dir != "":
            for i in range(num_obs):
                segs_pred_crop[i] = predict_sliding(   # Predict on reconstructions
                    input_n     = recons_crop[i],
                    models      = [diag_model],
                    window_size = tuple(diag_configs['window_size']),
                    window_norm = diag_configs['norm'])
        else:
            for i in range(num_obs):
                segs_pred_crop[i] = predict_sliding(  # Predict on accelerated images
                    input_n     = usampled_crop[i],
                    models      = [diag_model],
                    window_size = tuple(diag_configs['window_size']),
                    window_norm = diag_configs['norm'])

        stats = calculate_froc(
            y_true                   = segs_crop.squeeze(),
            y_pred                   = segs_pred_crop.squeeze(),
            preprocess_func          = dynamic_threshold,
            dynamic_threshold_factor = dyn_thresh,
            minimum_confidence       = min_conf,
            bootstrap                = 100 if DEBUG else 10000,
            p_auc_lims               = [(0.1, 2.5)]
        )

        np_path = os.path.join(recon_test_preds_dir, "stats.npy")
        np.save(np_path, stats, allow_pickle=True)
        print(f"\n> Wrote stats to file: {np_path} \n")

        save_predictions_to_file(
            unaccelerated   = uaccel.squeeze(),
            usampled        = usampled.squeeze(),
            recons          = recons_crop.squeeze(),
            segs_crop       = segs_crop.squeeze(),
            segs_pred_crop  = segs_pred_crop.squeeze(),
            acceleration    = acceleration,
            save_dir        = recon_test_preds_dir,
            seg_crop_s_list = seg_crop_s_list,
        )

    else:
        temp_path = os.path.join(recon_test_preds_dir, "stats.npy")
        print_(f">Trying to load: {temp_path}")
        stats = np.load(temp_path, allow_pickle=True)

    if DEBUG:
        roc_path = f"temp/dt{dyn_thresh}_mc{min_conf}_acc{acceleration}_val_froc.png"
    else:
        os.makedirs(os.path.join(R1_TRAIN_DIR, "figs"), exist_ok=True)
        roc_path = os.path.join(R1_TRAIN_DIR, "figs", f"dt{dyn_thresh}_mc{min_conf}val_froc.png")
    plot_roc(
        roc_tpr   = stats['roc_tpr'],
        roc_fpr   = stats['roc_fpr'],
        save_path = roc_path,
        roc_auc   = stats['patient_auc'],
        title     = f"ROC - R{int(acceleration)}n" if t2_recon_dir == None else f"ROC - R{int(acceleration)}DL"
    )
    
    sens1, fpp1 = stats['sensitivity'], stats['fp_per_patient']
    # p_auc = partial_auc(sens1, fpp1, low=0.1, high=2.5)

    return sens1, fpp1, stats['sens_95_boot_ci_low'], stats["sens_95_boot_ci_high"]


def read_config(config_path, verbatim=True):
    configs = read_yaml_to_dict(config_path)
    
    if verbatim:
        print_("\n>Configs")
        for key in configs:
            value = configs[key]
            print_(f"\t>{key} ({type(value)}):\t\t{value}")
        print_()

    return configs


################################################################################
SEED = 12345

if __name__ == '__main__':

    args = parse_input_args()
    ACCELERATIONS = args.accelerations
    RECON_DIRS = args.recon_dirs
    FOLD_NUM = args.fold_num
    DIAG_DIR = args.r1_dir
    DO_OPT = args.do_opt
    SHOW_BLOB = args.show_blob
    FNAME = args.fname
    MODEL_SELECT_CRIT = args.model_select_crit
    T2_LIST_PATH  = "data/path_lists/pirads_4plus/current_t2w.txt"
    DWI_LIST_PATH = "data/path_lists/pirads_4plus/current_dwi.txt"
    ADC_LIST_PATH = "data/path_lists/pirads_4plus/current_adc.txt"
    SEG_LIST_PATH = "data/path_lists/pirads_4plus/seg_new.txt"
    IDX_YAML_PATH = r"data/path_lists/pirads_4plus/train_val_test_idxs.yml" 
    VALSETKEY  = "val_set0"
    TESTSETKEY = "test_set0"
    PERC_LOAD = 1.0
    N_CPUS = 10
    NUM_READ_SPLITS = 2
    BATCH_SIZE = 1
    DEBUG = True if args.debug else False
    
    if DEBUG:
        N_CPUS = 4
        PERC_LOAD = 0.025
        set_gpu(gpu_idx=0)

    # For multiprocessing
    set_start_method("spawn")

    # Store sensitivities and false positives for all accelerations.
    model_sensitivities, model_false_positives = [], []
    cis_low, cis_high = [], []
    
    # Load the diagnostic T2-R1 model parameters
    R1_TRAIN_DIR = f"train_output/{DIAG_DIR}/fold_{FOLD_NUM}/"
    diag_configs = read_config(os.path.join(R1_TRAIN_DIR, "params.yml"))

    # Load the Diagnostic model
    diag_model_path = os.path.join(R1_TRAIN_DIR, "models", f"best_val_loss.h5")
    diag_model = load_model(diag_model_path, compile=False)

    # Indexes = validation set + test set
    indexes   = read_yaml_to_dict(IDX_YAML_PATH)
    val_idxs  = indexes[VALSETKEY]
    test_idxs = indexes[TESTSETKEY]
    test_idxs = test_idxs + val_idxs    # we merge these two sets, because the validation set has not been used for model selection

    if DEBUG:
        test_idxs = test_idxs[:int(PERC_LOAD * len(test_idxs))]
    print_(f"validation set indexes {test_idxs}\nWith length: {len(test_idxs)}")

    # Load data
    t2_files = [l.strip() for l in open(T2_LIST_PATH)]
    t2_files = list(np.asarray(t2_files)[test_idxs])
    seg_files = [l.strip() for l in open(SEG_LIST_PATH)]
    seg_files = list(np.asarray(seg_files)[test_idxs])

    # Loop over different accelerations (normal reconstructions)
    for model_idx1, acc in enumerate(ACCELERATIONS):
        sens, fps, ci_low, ci_high = calc_froc_on_new_data(
            t2_files     = t2_files,
            seg_files    = seg_files,
            diag_model   = diag_model,
            diag_params  = diag_configs,
            acceleration = acc,
        )
        model_sensitivities.append(sens)
        model_false_positives.append(fps)
        cis_low.append(ci_low)
        cis_high.append(ci_high)

    # Loop over Deep learning reconstructions.
    if RECON_DIRS != None:
        for model_idx2, recon_dir in enumerate(RECON_DIRS):
            sens, fps, ci_low, ci_high = calc_froc_on_new_data(
                t2_files     = t2_files,
                seg_files    = seg_files,
                diag_model   = diag_model,
                diag_params  = diag_configs,
                t2_recon_dir = recon_dir,
                acceleration = 0.0     # will be overloaded in the function
            )
            model_sensitivities.append(sens)
            model_false_positives.append(fps)
            cis_low.append(ci_low)
            cis_high.append(ci_high)

    if RECON_DIRS != None:
        m_names = [f"r{int(acc)}n" for acc in ACCELERATIONS] + RECON_DIRS
    else:
        m_names = [f"r{int(acc)}n" for acc in ACCELERATIONS]

    if True: 
        m_names = ['R1 (unaccelerated)', 'R4 (IFFT)', 'R4 DL Recon']
        m_names = ['R1 (unaccelerated)', 'R4 (IFFT)', 'R8 (IFFT)', 'R4 DL Recon', 'R8 DL Recon']
    
    fname_froc = f"figs_workspace/debug/{FNAME}_froc.png" if DEBUG else f"figs_workspace/{FNAME}_froc_foldnum{int(FOLD_NUM)}.png"
    plot_multiple_froc(
        sensitivities  = model_sensitivities,
        fp_per_patient = model_false_positives,
        ci_low         = cis_low,
        ci_high        = cis_high,
        model_names    = m_names,
        log_x          = True,
        save_as        = fname_froc,
        xlims          = (0.1, 2.5),
        title          = 'FROC Lesion-Based Diagnosis on a Multi-Site Test Set',
        height         = 9
    )
    print_(f"Wrote multi froc plot to: {fname_froc}")
    
    print_("--- DONE ---")
    