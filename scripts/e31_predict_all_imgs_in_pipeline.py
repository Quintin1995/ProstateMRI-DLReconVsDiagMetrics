from multiprocessing import set_start_method
import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm
from itertools import product
from typing import List, Tuple
import copy
import pandas as pd
import csv

from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization

from fastMRI_PCa.utils.k_space_masks import get_rand_exp_decay_mask_ac_matrix
from fastMRI_PCa.models import psnr_metric, ssim_metric

from umcglib.utils import print_, read_yaml_to_dict, print_stats_np, apply_parallel, set_gpu
import umcglib.images as im


def read_config(config_path, verbatim=True):
    configs = read_yaml_to_dict(config_path)
    
    if verbatim:
        print_("\n>Configs")
        for key in configs:
            value = configs[key]
            print_(f"\t>{key} ({type(value)}):\t\t{value}")
        print_()

    return configs


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
    t2_og = copy.deepcopy(t2)

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


def recon_model_predict_slid(
    prepped_imgs: List,
    t2_recon_dir: str
):
    print_("Predicting with the reconstruction model on understampled T2w images. (validation loss model)")

    t2_model_path = f"train_output/{t2_recon_dir}/fold0/models/best_loss_fold0.h5"
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
    t2_recon_dir: str
):
    print_("Predicting with the reconstruction model on understampled T2w images. (validation loss model)")
    t2_model_path = f"train_output/{t2_recon_dir}/fold0/models/best_loss_fold0.h5"
    recon_model = load_model(t2_model_path, compile=False)
    return recon_model.predict(np.stack(prepped_imgs, axis=0).squeeze(), batch_size=1)


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


def log_ssim_psnr(img1_4d, img2_4d, test_idxs, recon_fold, diag_model_num, diag_fold, acceleration, recon_method):
    # img1: First image batch. 4-D Tensor of shape [batch, height, width, channels] with only Positive Pixel Values.
    # img2: Second image batch. 4-D Tensor of shape [batch, height, width,channels] with only Positive Pixel Values.
    # test_idxs: List of indexes to record the ssim value for.
    # recon_fold: Fold number used for reconstruction, usually just 0
    # diag_model_num: Number of folder name of the diagnostic model used. 91 in most cases (in train_output)
    # acceleration: Acceleration just to create the accelerated image
    # Recon_method: Reconstruction method used. Either 'U-Net' or 'IFFT'

    metrics_csv_path = os.path.join('stats', f"ssim_psnr_test_set_rf{recon_fold}_dn{diag_model_num}_N{img1_4d.shape[0]}.csv")
    if os.path.exists(metrics_csv_path):
        df = pd.read_csv(metrics_csv_path)

    print(f"Calculating SSIM and PSNR")
    print(f"size of ssim and psnr arrays img1_4d: {img1_4d.shape}, img2_4d: {img2_4d.shape}")

    idxs_out, test_idxs_out, recon_methods, fold_nums, accelerations, metrics, metric_vals = [],[],[],[],[],[],[]

    for img_idx in range(len(img1_4d)):

        # Take single image and retain 4d'nes
        img1 = np.expand_dims(img1_4d[img_idx], 0)
        img2 = np.expand_dims(img2_4d[img_idx], 0)

        # SSIM
        idxs_out.append(img_idx)
        test_idxs_out.append(test_idxs[img_idx])
        recon_methods.append(recon_method)
        fold_nums.append(diag_fold)
        accelerations.append(acceleration)
        metrics.append("ssim")
        ssim = round(float(np.asarray(ssim_metric(y_true=img1, y_pred=img2))), 2)
        metric_vals.append(ssim)

        # PSNR
        idxs_out.append(img_idx)
        test_idxs_out.append(test_idxs[img_idx])
        recon_methods.append(recon_method)
        fold_nums.append(diag_fold)
        accelerations.append(acceleration)
        metrics.append("psnr")
        psnr = round(float(np.asarray(psnr_metric(y_true=img1, y_pred=img2))), 2)
        metric_vals.append(psnr)

        # print(f"{str(img_idx).zfill(3)}: SSIM: {ssim}, PSNR: {psnr}")

    metric_data = {
        "idx": idxs_out,
        "test_idx": test_idxs_out,
        "recon_method": recon_method,
        "fold_num": fold_nums,
        "acceleration": accelerations,
        "metric": metrics,
        "metric_val": metric_vals,
    }

    if os.path.exists(metrics_csv_path):
        new_df = pd.DataFrame(data=metric_data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(data=metric_data)

    df.to_csv(metrics_csv_path, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')


################################################################################
SEED = 12345

if __name__ == '__main__':

    DEBUG = False

    DO_R1 = True            # perform undersampling on R1

    #### MANUAL PARAMS
    t2_recon_dir = "80_recon_r4"   # 80_recon_r4 or 81_recon_r8
    diag_fold    = 0               # done: ((fold3_r4, fold3_r8), (fold0_r4, fold0_r8), (fold1_r4, fold1_r8), (fold2_r4, fold2_r8), (fold4_r4, x))
                                   # done:
    recon_fold   = 0
    load_perc    = 0.025 if DEBUG else 1.0
    set_gpu(gpu_idx=1)

    dyn_thresh   = 0.75
    min_conf     = 0.15

    # Constant params
    IDX_YAML_PATH = r"data/path_lists/pirads_4plus/train_val_test_idxs.yml" 
    VALSETKEY  = "val_set0"
    TESTSETKEY = "test_set0"
    T2_LIST_PATH  = "data/path_lists/pirads_4plus/current_t2w.txt"
    SEG_LIST_PATH = "data/path_lists/pirads_4plus/seg_new.txt"
    NUM_READ_SPLITS = 2
    N_CPUS = 8 if DEBUG else 12
    BATCH_SIZE = 1

    # For multiprocessing
    set_start_method("spawn")

    # Read diagnostic model parameters and model
    R1_TRAIN_DIR = f"train_output/91_diag_t2_r1_norm/fold_{diag_fold}/"
    diag_configs = read_config(os.path.join(R1_TRAIN_DIR, "params.yml"))
    diag_model_path = os.path.join(R1_TRAIN_DIR, "models", f"best_val_loss.h5")
    diag_model = load_model(diag_model_path, compile=False)
    
    # Indexes = validation set + test set
    indexes = read_yaml_to_dict(IDX_YAML_PATH)
    val_idxs = indexes[VALSETKEY]
    test_idxs = indexes[TESTSETKEY]
    test_idxs = test_idxs + val_idxs    # we merge these two sets, because the validation set has not been used for model selection
    test_idxs = test_idxs[:int(len(test_idxs) * load_perc)]

    # Load paths
    t2_files = [l.strip() for l in open(T2_LIST_PATH)]
    t2_files = list(np.asarray(t2_files)[test_idxs])
    seg_files = [l.strip() for l in open(SEG_LIST_PATH)]
    seg_files = list(np.asarray(seg_files)[test_idxs])

    # Load recon params and overrride parameters
    recon_params = read_yaml_to_dict(f"train_output/{t2_recon_dir}/fold{recon_fold}/train_recon_config.yml")
    norm = recon_params['normalization']
    acceleration = float(recon_params['acceleration']) if not DO_R1 else 1.0

    # Define dir where images should be saved.
    fig_save_dir = os.path.join("figs_workspace", f"whole_pipeline_imgs_91fold{diag_fold}_r{int(acceleration)}")
    os.makedirs(fig_save_dir, exist_ok=True)
    
    # Load undersampled data with segmentation and unacclerated images
    undersampled, segs, unacclerated_inp = load_data(
        t2_files        = t2_files,
        seg_files       = seg_files,
        sampling_window = list(recon_params['target_shape']),
        spacing         = tuple(diag_configs['spacing']),
        norm            = norm,
        acceleration    = acceleration,
        centre_sampling = recon_params['centre_sampling']
    )

    print(f"\nundersampled len: {len(undersampled)} and shape: {undersampled[0].shape}")
    print(f"segs len: {len(segs)} and shape {segs[0].shape}")
    print(f"unacclerated_inp len: {len(unacclerated_inp)} and shape{unacclerated_inp[0].shape}")

    unacclerated_inp = np.stack(unacclerated_inp, axis=0)
    if not DO_R1:
        reconstructed    = recon_model_predict(undersampled, t2_recon_dir)
    undersampled     = np.stack(undersampled, axis=0)     
    segs             = np.stack(segs, axis=0)
    seg_pred         = np.zeros(undersampled.shape, dtype=np.float32)

    # Calculate SSIM and PSNR (unacc vs undersampled Rx) and (unacc vs recon Rx)
    log_ssim_psnr(
        img1_4d        = unacclerated_inp,
        img2_4d        = undersampled,
        test_idxs      = test_idxs,
        recon_fold     = recon_fold,
        diag_model_num = 91,
        diag_fold      = diag_fold,
        acceleration   = acceleration,
        recon_method   = "IFFT"
    )
    if not DO_R1:
        log_ssim_psnr(
            img1_4d        = unacclerated_inp,
            img2_4d        = reconstructed,
            test_idxs      = test_idxs,
            recon_fold     = recon_fold,
            diag_model_num = 91,
            diag_fold      = diag_fold,
            acceleration   = acceleration,
            recon_method   = "UNet"
        )

    print(f"\nunacclerated_inp shape: {unacclerated_inp.shape}")
    if not DO_R1:
        print(f"reconstructed len: {len(reconstructed)} and shape {reconstructed[0].shape}")
    print(f"undersampled shape: {undersampled.shape}")
    print(f"segs shape: {segs.shape}")
    print(f"seg_pred shape: {seg_pred.shape}\n")

    # Reconstruction Center cropping before we apply the detection model
    crop_shape_seg = (diag_configs['sampling_window'][0], diag_configs['sampling_window'][1], 16, 1)
    cropped_recons = np.zeros((undersampled.shape[0],) + crop_shape_seg)
    seg_pred_crop  = np.zeros((undersampled.shape[0],) + crop_shape_seg)
    
    print(f"seg_pred_crop shape: {seg_pred_crop.shape}\n")
    
    if not DO_R1:
        for img_idx in range(undersampled.shape[0]):
            recon_s = sitk.GetImageFromArray(np.squeeze(reconstructed[img_idx]).T)
            recon_crop_s = im.center_crop(recon_s, shape = crop_shape_seg)
            recon_crop_n =  sitk.GetArrayFromImage(recon_crop_s).T
            cropped_recons[img_idx] = np.expand_dims(recon_crop_n, axis=3)

    # Segs cropping before we apply the detection model
    segs_crop = np.zeros((undersampled.shape[0],) + crop_shape_seg)
    print(f"segs_crop shape: {segs_crop.shape}\n")
    for img_idx in range(undersampled.shape[0]):
        seg_s = sitk.GetImageFromArray(np.squeeze(segs[img_idx]).T)
        seg_crop_s = im.center_crop(seg_s, shape = crop_shape_seg)
        seg_crop_n =  sitk.GetArrayFromImage(seg_crop_s).T
        segs_crop[img_idx] = np.expand_dims(seg_crop_n, axis=3)

    # detection model prediction with sliding window
    for img_idx in range(undersampled.shape[0]):
        seg_pred_crop[img_idx] = predict_sliding(
            input_n     = cropped_recons[img_idx],
            models      = [diag_model],
            window_size = tuple(diag_configs['window_size']),
            window_norm = diag_configs['norm']
        )
    print(f"seg_pred shape: {seg_pred.shape}")

    # write all 5 images types as nifti to file
    for mri_idx in range(seg_pred_crop.shape[0]):

        fname_temp = os.path.join(fig_save_dir, f"{mri_idx}_1unacclerated_inp_R{acceleration}.nii.gz")
        img = sitk.GetImageFromArray(np.squeeze(unacclerated_inp[mri_idx]).T)
        sitk.WriteImage(img, fname_temp)
        print_(f"Wrote to: {fname_temp}")

        fname_temp = os.path.join(fig_save_dir, f"{mri_idx}_2undersampled_R{acceleration}.nii.gz")
        img = sitk.GetImageFromArray(np.squeeze(undersampled[mri_idx]).T)
        sitk.WriteImage(img, fname_temp)
        print_(f"Wrote to: {fname_temp}")

        if not DO_R1:
            fname_temp = os.path.join(fig_save_dir, f"{mri_idx}_3reconstructed_R{acceleration}.nii.gz")
            img = sitk.GetImageFromArray(np.squeeze(reconstructed[mri_idx]).T)
            sitk.WriteImage(img, fname_temp)
            print_(f"Wrote to: {fname_temp}")

        fname_temp = os.path.join(fig_save_dir, f"{mri_idx}_4segs_crop_R{acceleration}.nii.gz")
        img = sitk.GetImageFromArray(np.squeeze(segs_crop[mri_idx]).T)
        img.CopyInformation(seg_crop_s)
        sitk.WriteImage(img, fname_temp)
        print_(f"Wrote to: {fname_temp}")

        fname_temp = os.path.join(fig_save_dir, f"{mri_idx}_5seg_pred_crop_R{acceleration}.nii.gz")
        img = sitk.GetImageFromArray(np.squeeze(seg_pred_crop[mri_idx]).T)
        img.CopyInformation(seg_crop_s)
        sitk.WriteImage(img, fname_temp)
        print_(f"Wrote to: {fname_temp}\n")