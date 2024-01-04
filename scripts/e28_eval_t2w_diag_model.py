import argparse
import os
from typing import Tuple 
import numpy as np
import SimpleITK as sitk
from multiprocessing import set_start_method
import numpy as np
from typing import Tuple
import copy
from itertools import product
from typing import List, Tuple

from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.keras.models import load_model

from fastMRI_PCa.utils import get_rand_exp_decay_mask, get_rand_exp_decay_mask_ac_matrix
from umcglib.froc import calculate_froc, plot_multiple_froc, partial_auc
import umcglib.images as im
from umcglib.predict import predict_sliding, predict_ensemble
from umcglib.utils import print_stats_np, apply_parallel
from umcglib.utils import print_stats_np_list, read_yaml_to_dict, set_gpu, print_
from umcglib.plotting import write_array2nifti_t2, plot_roc
from umcglib.binarize import dynamic_threshold
from fastMRI_PCa.utils import dump_dict_to_yaml
from tqdm import tqdm

# from scripts.e24_train_t2w_recon_model_no import load_and_preprocess_t2


################################  README  ######################################
# NEW - This script will load a reconstruction model and reconstruct some
# validation images. Calculate the SSIM and PSNR per image and aroundt the 
# lesion. Load a diagnostic model and perform FROC and ROC analysis.
################################################################################


def parse_input_args():
    parser = argparse.ArgumentParser(description='Argument parser for evaluation of the diagnostic model when given reconstructions.')

    parser.add_argument(
        '-d',
        '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help="Whether to run the code in debug mode or not. Use the flag if run in debug mode."
    )

    parser.add_argument(
        '-td',
        '--train_dir',
        type=str,
        required=True,
        help='FilePath to train directory where the model is trained and stored. Example: train_output/75_recon_slid_wind_r1dl/fold0',
    )

    parser.add_argument(
        '-mf',
        '--model_fname',
        type    = str,
        default = "best_val_loss.h5",
        help    = 'Name of the model .h5 file to be loaded for the validation set.',
    )

    parser.add_argument(
        '-f',
        '--fname',
        type=str,
        default='multi_froc',
        help='File name of the output figure. It will be stored into the figs_workspace folder.')
    
    args = parser.parse_args()
    print_(f"args:\n {args}")
    
    return args


def read_config(config_path, verbatim=True):
    
    configs = read_yaml_to_dict(config_path)
    
    # fix some params
    # configs['window_size'] = tuple(configs['window_size'])
    configs['target_shape'] = tuple(configs['target_shape'])
    
    configs['seed'] = None if not configs['use_seed'] else configs['seed']
    
    if verbatim:
        print_("\n>Configs")
        for key in configs:
            value = configs[key]
            print_(f"\t>{key} ({type(value)}):\t\t{value}")
        print_()

    return configs


def get_spacings(t2_files, val_set_key):

    if DEBUG:
        fname = os.path.join(VAL_DIR, f'val_spacings_{val_set_key}_DEBUG.txt')
    else:
        fname = os.path.join(VAL_DIR, f'val_spacings_{val_set_key}.txt')
    
    if not os.path.isfile(fname):
        spacings = [(0.0, 0.0, 0.0) for _ in range(len(t2_files))]

        for idx, filename in enumerate(t2_files):
            spacings[idx] = list(sitk.ReadImage(filename, sitk.sitkFloat32).GetSpacing())
            print(f"{idx}: Found spacing: {spacings[idx]}")

        with open(fname, 'w') as fp:
            for item in spacings:
                fp.write(f"{round(item[0], 2)},{round(item[1], 2)},{round(item[2], 2)}\n")
            print(f'>Wrote spacing to file: {fname}')
    else:
        spacings = [tuple([float(part) for part in l.strip().split(',')]) for l in open(fname)]

    return spacings

    
def write_heatmap2nifti(heatmap, filename, spacing = None):
    mri_sitk = sitk.GetImageFromArray(heatmap.squeeze().T)
    if spacing != None:
        mri_sitk.SetSpacing(spacing)
    path = os.path.join(VAL_DIR, filename)
    sitk.WriteImage(mri_sitk, path)
    print_(f">-Wrote nifti to path: {path}")


def write_seg2nfiti(seg_path, diag_configs, t2_ac_n, val_idx):
    seg_s = sitk.ReadImage(seg_path, sitk.sitkFloat32)
    seg_s = im.resample(
        image       = seg_s, 
        min_shape   = diag_configs['window_size'], 
        method      = sitk.sitkNearestNeighbor, 
        new_spacing = diag_configs['target_space'],
    )
    seg_s = im.center_crop(seg_s, diag_configs['window_size'])
    seg_s.CopyInformation(sitk.GetImageFromArray(t2_ac_n.T))
    seg_s.SetSpacing(diag_configs['target_space'])
    seg_out_path = os.path.join(VAL_DIR, f"{val_idx}_seg.nii.gz")
    sitk.WriteImage(seg_s, seg_out_path)
    print_(f">-Wrote nifti to path: {seg_out_path}")


def read_and_preprocess(
    paths, 
    window_size=None, 
    spacing=(0.5, 0.5, 3.), 
    crop=True,
    to_numpy=True,
    norm="znorm",
    acceleration=None,
    centre_sampling=None,
    seed: int = 12345.
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
                      centre_sampling=centre_sampling,
                      seed=seed,
                      nifti_path=t2_path)


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


def preprocess(
    t2,
    window_size,
    spacing = (0.5, 0.5, 3.),
    crop = True, 
    to_numpy = True,
    seg = None,
    norm = "znorm",
    acceleration = None,
    centre_sampling = None,
    seed: int = 12345,
    nifti_path = "",
    dicom_db_path = r'sqliteDB/dicoms.db'
):
    t2_u = copy.deepcopy(t2)        # t2u = t2w unaccelerated

    if acceleration != None: 
        # mask = get_rand_exp_decay_mask(
        #     width           = t2.GetSize()[0],
        #     height          = t2.GetSize()[1],
        #     sampling        = 1.0/acceleration,
        #     centre_sampling = centre_sampling,
        #     seed            = seed,
        #     verbatim        = False
        # )
        mask = get_rand_exp_decay_mask_ac_matrix(
            width           = t2.GetSize()[0],
            height          = t2.GetSize()[1],
            sampling        = 1.0/acceleration,
            centre_sampling = centre_sampling,
            seed            = seed,
            exp_scale       = 0.4,      # determined emperically
            nifti_path      = nifti_path,
            dicom_db_path   = dicom_db_path,
            tablename       = 'dicom_headers_v2',
            verbatim        = False,
        )

        t2 = im.undersample_kspace(t2, mask)

    # Preprocess the ADC image, then resample the rest to it
    t2 = im.resample(
        image       = t2, 
        min_shape   = window_size, 
        method      = sitk.sitkLinear, 
        new_spacing = spacing,
    )
    t2_u = im.resample(
        image       = t2_u, 
        min_shape   = window_size, 
        method      = sitk.sitkLinear, 
        new_spacing = spacing,
    )

    if crop:
        t2 = im.center_crop(t2, window_size)
        t2_u = im.center_crop(t2_u, window_size)

    if seg is not None:
        seg = im.resample_to_reference(seg, t2, sitk.sitkNearestNeighbor)

    # Return the SITK preprocessed images if requested
    if not to_numpy: 
        return t2_u, t2, seg

    # Extract numpy arrays
    t2  = sitk.GetArrayFromImage(t2).T
    t2_u  = sitk.GetArrayFromImage(t2_u).T

    # Stack the inputs, add new axis to seg
    img_n = t2[..., np.newaxis]
    img_n_u = t2_u[..., np.newaxis]
    
    # Z-Normalize at crop level
    if norm == "znorm":
        img_n -= np.mean(img_n)
        img_n /= np.std(img_n)

        img_n_u -= np.mean(img_n_u)
        img_n_u /= np.std(img_n_u)
    
    if norm == "rescale_0_1":
        img_n = (1.0*(img_n - np.min(img_n))/np.ptp(img_n))
        img_n_u = (1.0*(img_n_u - np.min(img_n_u))/np.ptp(img_n_u))
    
    # Extract batch for the segmentation if provided
    if seg is not None: 
        seg = sitk.GetArrayFromImage(seg).T
        seg = (seg[..., None] > 0.5) * 1.

    return img_n_u, img_n, seg.astype(np.float32)


def load_data(
    t2_files: List[str],
    seg_files: List[str],
    sampling_window: List[int],
    spacing: Tuple[float,float,float],
    norm: str,
    acceleration: float,
    centre_sampling: float,
    seed: int = 12345,
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
            N_WORKERS,
            window_size=sampling_window,
            crop=True,
            spacing=spacing,
            norm=norm,
            acceleration=acceleration,
            centre_sampling=centre_sampling,
            seed=seed
        )
        
    unaccel_imgs, accel_imgs, prepped_segs = [], [], []
    for u, i, s in combined_prepped:
        unaccel_imgs.append(u)
        accel_imgs.append(i)
        prepped_segs.append(s)

    print_(f"Data Loaded with stats:")
    print_(f"\tImgs num: {len(accel_imgs)}")
    print_(f"\tSegs num: {len(prepped_segs)}")

    print_stats_np(unaccel_imgs[0], "First observations unacclerated images.")    
    print_stats_np(accel_imgs[0], "First observation accelerated Imgs:")
    print_stats_np(prepped_segs[0], "First observation Segs:")
    
    return unaccel_imgs, accel_imgs, prepped_segs


def recon_model_predict(
    t2_imgs: List,
    t2_recon_dir: str,
    fold_num: int,
):
        print_("Predicting with the reconstruction model on understampled T2w images. (validation loss model)")

        t2_model_path = os.path.join('train_output', t2_recon_dir, f'fold{fold_num}', 'models', f'best_loss_fold{fold_num}.h5')
        recon_model = load_model(t2_model_path, compile=False)

        # t2_imgs = np.stack(t2_imgs, axis=0)
        # recons = recon_model.predict(t2_imgs, batch_size=1)

        for img_idx in range(len(t2_imgs)):
            t2_imgs[img_idx] = predict_sliding(
                input_n     = np.squeeze(t2_imgs[img_idx]),
                models      = [recon_model],
                window_size = (160, 160, 16),
                window_norm = "rescale_0_1"
            )

        # recons = np.squeeze(recon_model.predict(np.stack(prepped_imgs, 0), batch_size=BATCH_SIZE))
        # prepped_imgs = [prepped_imgs[mri_idx] for mri_idx in range(recons.shape[0])]
        # prepped_imgs = np.expand_dims(prepped_imgs, axis=4)
        return t2_imgs


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


def calc_froc_on_new_data(
    model_idx: int,
    t2_files: list,
    seg_files: list,
    diag_model,
    diag_configs: dict,
    val_idxs: List[int],
    t2_recon_dir: str = None,
    acceleration: float = 1.0,
):
    norm = diag_configs['normalization']

    window_size = list(diag_configs['window_size'])
    
    # Load the T2w reconstruction model if not None
    if t2_recon_dir != None:
        # Load Reconstruction model parameters
        recon_configs_path = f"train_output/{t2_recon_dir}/train_recon_config.yml"
        recon_configs = read_yaml_to_dict(recon_configs_path)
        norm = recon_configs['normalization']
        acceleration = float(recon_configs['acceleration'])
        window_size = list(recon_configs['target_shape'])

    # Load the input images and the corresponding segmentation (also the unacclerated version)
    unaccel_imgs, acc_imgs, segs = load_data(
        t2_files        = t2_files,
        seg_files       = seg_files,
        sampling_window = window_size,
        spacing         = tuple(diag_configs['target_space']),
        norm            = norm,
        acceleration    = acceleration,
        centre_sampling = diag_configs['centre_sampling'],
        seed            = diag_configs['seed']
    )

    # Do reconstruction on the accelerated images
    if t2_recon_dir != None:
        recon_imgs = recon_model_predict(acc_imgs, t2_recon_dir, diag_configs['fold_num'])
        recon_imgs = np.stack(recon_imgs, axis=0)

    segs = np.stack(segs, axis=0)

    #predict with the diagnostic model to obtian a csPCa heatmap
    if t2_recon_dir != None:
        recon_imgs = im.center_crop_n(recon_imgs, list(diag_configs['window_size']))
        diag_imgs = diag_model.predict(recon_imgs)

    for img_idx in range(recon_imgs.shape[0]):
        recon_imgs[img_idx] = predict_sliding(
            input_n     = recon_imgs[img_idx],
            models      = [diag_model],
            window_size = tuple(diag_configs['window_size']),
            window_norm = diag_configs['normalization']
        )

    else:
        diag_imgs = diag_model.predict(np.stack(acc_imgs, axis=0))

    # Write unacclerated/accelerated/seg/recon_pred/heatmap_pred to files.
    if False:
        for i, val_idx in enumerate(val_idxs):
            write_array2nifti_t2(unaccel_imgs[i].squeeze(), VAL_DIR, f"{val_idx}_t2_unacc_r{int(acceleration)}_{t2_recon_dir}.nii.gz")
            write_array2nifti_t2(acc_imgs[i].squeeze(), VAL_DIR, f"{val_idx}_t2_acc_r{int(acceleration)}_{t2_recon_dir}.nii.gz")
            write_array2nifti_t2(segs[i].squeeze(), VAL_DIR, f"{val_idx}_t2_seg_r{int(acceleration)}_{t2_recon_dir}.nii.gz")
            write_array2nifti_t2(diag_imgs[i].squeeze(), VAL_DIR, f"{val_idx}_t2_heatmap_r{int(acceleration)}_{t2_recon_dir}.nii.gz")
            if t2_recon_dir != None:
                write_array2nifti_t2(recon_imgs[i].squeeze(), VAL_DIR, f"{val_idx}_t2_recon_r{int(acceleration)}_{t2_recon_dir}.nii.gz")

    # Data preparation
    acc_imgs = np.squeeze(acc_imgs)
    segs = np.squeeze(segs)
    diag_imgs = np.squeeze(diag_imgs)

    print_stats_np(acc_imgs, "x true")

    # Calculate FROC
    dyn_thresh = 0.75
    min_conf = 0.15
    stats = calculate_froc(
        y_true                   = segs,
        y_pred                   = diag_imgs,
        preprocess_func          = dynamic_threshold,
        dynamic_threshold_factor = dyn_thresh,
        minimum_confidence       = min_conf
    )
    dump_dict_to_yaml(stats, VAL_DIR, f"froc_stats{diag_configs['fold_num']}_r{int(acceleration)}_{t2_recon_dir}")
    
    if DEBUG:
        roc_path = f"temp/dt{dyn_thresh}_mc{min_conf}_acc{acceleration}_val_roc.png"
    else:
        roc_path = os.path.join(TRAIN_DIR, 'val_set', "figs", f"dt{dyn_thresh}_mc{min_conf}val_roc.png")
    plot_roc(
        roc_tpr=stats['roc_tpr'],
        roc_fpr=stats['roc_fpr'],
        save_path=roc_path,
        roc_auc=stats['patient_auc'],
        title = f"ROC - R{int(acceleration)}n" if t2_recon_dir == None else f"ROC - R{int(acceleration)}DL"
    )
    
    sens1, fpp1 = stats['sensitivity'], stats['fp_per_patient']
    # p_auc = partial_auc(sens1, fpp1, low=0.1, high=2.5)

    return sens1, fpp1


################################################################################


def main(args, diag_configs):
    
    # Store sensitivities and false positives for all accelerations.
    model_sensitivities, model_false_positives = [], []
    
    # Load the csPCa detection model
    diag_model = load_model(
        filepath = os.path.join(TRAIN_DIR, "models", args.model_fname),
        compile  = False
    )

    # Load validation indexes
    val_set_key = f"val_set{diag_configs['train_set_key'][-1]}"
    indexes_dict = read_yaml_to_dict(diag_configs['indexes_path'])
    val_idxs = indexes_dict[val_set_key]
    print_(f"> Validation idxs: {val_idxs}\n\twith length: {len(val_idxs)}")
    
    # Get filnames of validation files
    t2_files = [l.strip() for l in open(diag_configs['train_path_list'])]
    seg_files = [l.strip() for l in open(diag_configs['label_path_list'])]

    if DEBUG:   # load less data if in debug mode
        t2_files  = t2_files[:20]
        seg_files = seg_files[:20]
        val_idxs  = list(range(20))
        set_gpu(gpu_idx=0)
        VAL_DIR = TEMP_DIR

    # Only take the actualy valiation images from all the files.
    t2_files = list(np.asarray(t2_files)[val_idxs])
    seg_files = list(np.asarray(seg_files)[val_idxs])

    # Loop over different accelerations (normal reconstructions)
    for model_idx1, acc in enumerate(ACCELERATIONS):
        sens, fps = calc_froc_on_new_data(
            model_idx    = model_idx1,
            t2_files     = t2_files,
            seg_files    = seg_files,
            diag_model   = diag_model,
            diag_configs = diag_configs,
            acceleration = acc,
            val_idxs     = val_idxs,
        )
        model_sensitivities.append(sens)
        model_false_positives.append(fps)

    # Loop over Deep learning reconstructions.
    for model_idx2, recon_dir in enumerate(RECON_DIRS):
        sens, fps = calc_froc_on_new_data(
            model_idx    = model_idx2 + model_idx1 + 1,
            t2_files     = t2_files,
            seg_files    = seg_files,
            diag_model   = diag_model,
            diag_configs = diag_configs,
            t2_recon_dir = recon_dir,
            acceleration = 0.0,     # will be overloaded in the function
            val_idxs     = val_idxs,
        )
        model_sensitivities.append(sens)
        model_false_positives.append(fps)

    m_names = [f"r{int(acc)}n" for acc in ACCELERATIONS] + RECON_DIRS
    
    fname_froc = os.path.join(VAL_FIG_DIR, "debug", f"{FNAME}_froc.png") if DEBUG else os.path.join(VAL_FIG_DIR, f"{FNAME}_froc.png")
    # fname_froc = f"figs_workspace/debug/{FNAME}_froc.png" if DEBUG else f"figs_workspace/{FNAME}_froc.png"
    plot_multiple_froc(
        sensitivities=model_sensitivities,
        fp_per_patient=model_false_positives,
        model_names=m_names,
        log_x=True,
        save_as=fname_froc,
        xlims=(0.1, 2.5)
    )
    print_(f"Wrote multi froc plot to: {fname_froc}")
    

################################################################################


if __name__ == '__main__':
    
    RECON_DIRS = ['80_recon_r4', '81_recon_r8', '82_recon_r12']
    RECON_DIRS = ['80_recon_r4', '81_recon_r8']
    RECON_DIRS = []
    RECON_DIRS = ['80_recon_r4']
    ACCELERATIONS = [4.0]
    ACCELERATIONS = [1.0, 4.0, 8.0]
    ACCELERATIONS = [1.0]
    ACCELERATIONS = [1.0, 4.0]
    
    args = parse_input_args()
    NUM_READ_SPLITS = 2
    N_WORKERS = 4
    dicom_db_path: str = r'sqliteDB/dicoms.db'
    seg_path_list_file = "data/path_lists/pirads_4plus/seg_new.txt"
    DEBUG = args.debug
    FNAME = args.fname
    print_(f">Is Debug: {DEBUG}")

    # Fix tensorflow memory leak and multiprocessing error:
    disable_eager_execution()
    set_start_method("spawn")
    
    config_path = os.path.join(args.train_dir, "train_diag_config.yml")
    if os.path.exists(config_path):
        diag_configs = read_config(config_path)
    else:
        diag_configs = read_config(os.path.join(args.train_dir, 'params.yml'))

    TRAIN_DIR = args.train_dir
    TEMP_DIR  = os.path.join(TRAIN_DIR, "temp")
    VAL_DIR = os.path.join(TRAIN_DIR, "val_set")
    VAL_FIG_DIR = os.path.join(VAL_DIR, 'figs')
    
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(VAL_FIG_DIR, exist_ok=True)
    os.makedirs(os.path.join(VAL_FIG_DIR, 'debug'), exist_ok=True)

    main(args, diag_configs)
    print_("--- DONE ---")