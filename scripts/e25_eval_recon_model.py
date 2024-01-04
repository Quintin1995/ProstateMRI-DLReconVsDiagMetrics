import argparse
import os
from typing import Tuple 
import numpy as np
import SimpleITK as sitk
from multiprocessing import set_start_method
import numpy as np
from typing import Tuple
from tqdm import tqdm
from itertools import product

from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.keras.models import load_model

import umcglib.images as im
from umcglib.predict import predict_sliding, predict_ensemble
from umcglib.utils import print_stats_np, print_stats_np_list, read_yaml_to_dict, set_gpu, print_
from fastMRI_PCa.visualization import write_array2nifti_t2

from scripts.e24_train_t2w_recon_model_ import read_config, load_and_preprocess_t2


################################  README  ######################################
# NEW - This script will load a patch based reconstruction model and reconstruct
# some validation images and calculate statistics on it. 
# It will also compute some metrics per lesion.
################################################################################


def parse_input_args():
    parser = argparse.ArgumentParser(description='Argument parser for the reconstruction model evaluation.')
    
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
        default = "best_loss_fold0.h5",
        help    = 'Name of the model .h5 file to be loaded for the validation set.',
    )

    parser.add_argument(
        '-dmp',
        '--do_mirror_pred',
        dest='do_mirror_pred',
        action='store_true',
        default = False,
        help='For each window de prediction will be averaged with the mirrored version of the same window.',
    )

    parser.add_argument(
        '-s',
        '--xyz_steps',
        nargs='+',
        type=int,
        help='x y z steps on the input image for the slinding window. (example: 4 4 4 = 4 steps in the x dim and 4 in the other dimensions aswel.)')

    
    args = parser.parse_args()
    print_(f"args:\n {args}")
    return args


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


def write_to_file(val_idx, X_true_n, Y_true_n, Y_pred_n, spacing) -> None:
    print_stats_np(X_true_n, "\nX_true_n")
    print_stats_np(Y_true_n, "Y_true_n")
    print_stats_np(Y_pred_n, "Y_pred_s")

    X_true_s = sitk.GetImageFromArray(X_true_n.squeeze().T)
    Y_true_s = sitk.GetImageFromArray(Y_true_n.squeeze().T)
    Y_pred_s = sitk.GetImageFromArray(Y_pred_n.squeeze().T)

    X_true_s.SetSpacing(spacing)
    Y_true_s.SetSpacing(spacing)
    Y_pred_s.SetSpacing(spacing)

    fname_X_true_s = os.path.join(VAL_DIR, f"idx{val_idx}_x_true.nii.gz")
    fname_Y_true_s = os.path.join(VAL_DIR, f"idx{val_idx}_y_true.nii.gz")
    fname_Y_pred_s = os.path.join(VAL_DIR, f"idx{val_idx}_y_pred.nii.gz")

    sitk.WriteImage(X_true_s, fname_X_true_s)
    sitk.WriteImage(Y_true_s, fname_Y_true_s)
    sitk.WriteImage(Y_pred_s, fname_Y_pred_s)

    print_(f"> Wrote to: {fname_X_true_s}")
    print_(f"> Wrote to: {fname_Y_true_s}")
    print_(f"> Wrote to: {fname_Y_pred_s}")
    

def predict_sliding_median(
    input_n: np.ndarray,
    models: list,   #Model objects
    window_size: Tuple[int,int,int],
    weight_map: np.ndarray = None,
    n_steps = (10, 10, 4),
    n_out_channels: int = 1,
    batch_size: bool = 1,
    do_mirror_pred: bool = True,
    aggregate_func: str = "mean"
) -> np.ndarray:
    """
    Predicts on the input_n image with a sliding window method.

    Parameters:
    `input_n`: numpy image array np.ndarray 3D array.
    `models`: List of keras model objects used to predict with for each window.
    `window_size`: tuple of ints. The size of the sliding window
    `window_norm`: string. The normalization method used per window before prediction.
    `weight_map`: np.ndarray with weights per voxel. Should be same size as the window_size
    `n_steps`: tuple with (x,y,z) number of steps for the x,y,z directions for the sliding window
    `aggregate_func`: str = Method used for aggregating multiple predictions per voxel. mean of median for example
    """
    if weight_map != None:
        assert window_size == weight_map.shape, "window_size and weight map in sliding window pred should be of the same shape."

    # Calculate the possible crop origins to cover the full image with overlap
    possible_offsets = [
        a_len - c_len for a_len, c_len in zip(input_n.shape, window_size)]

    x_list, y_list, z_list = [], [], []
    x, y, z = 0, 0, 0

    while x <= possible_offsets[0]:
        x_list += [x]
        x += max(min(input_n.shape[0]//n_steps[0], possible_offsets[0]-x), 1)
    while y <= possible_offsets[1]:
        y_list += [y]
        y += max(min(input_n.shape[1]//n_steps[1], possible_offsets[1]-y), 1)
    while z <= possible_offsets[2]:
        z_list += [z]
        z += max(min(input_n.shape[2]//n_steps[2], possible_offsets[2]-z), 1)
    
    all_crop_origins = list(product(x_list, y_list, z_list))

    if aggregate_func == 'mean':
        full_prediction = np.zeros(input_n.shape[:3] + (n_out_channels, ), dtype=np.float32)
    if aggregate_func == 'median':
        full_prediction = np.zeros(input_n.shape[:3] + (n_out_channels, ) + (len(all_crop_origins),), dtype=np.float32)
    
    print_(f"full prediction shape: {full_prediction.shape}")
    
    summed_weights = np.zeros_like(full_prediction, dtype=np.float32)

    for crop_idx, (x, y, z) in enumerate(tqdm(all_crop_origins, desc="Collecting Crops...")):

        img_crop = np.copy(input_n[
                np.newaxis,
                x:x+window_size[0],
                y:y+window_size[1],
                z:z+window_size[2],
                np.newaxis,
            ]
        )
    
        # Softmax predictions are collected for each of the CV models
        crop_pred = predict_ensemble(
            input_n        = img_crop,
            models         = models,
            weight_map     = weight_map,
            batch_size     = batch_size,
            do_mirror_pred = do_mirror_pred,
        )

        if aggregate_func == 'mean':
            full_prediction[
                x:x+window_size[0],
                y:y+window_size[1],
                z:z+window_size[2]
            ] += crop_pred[0, ...]  #remove batch dim

        if aggregate_func == 'median':
            full_prediction[
                x:x+window_size[0],
                y:y+window_size[1],
                z:z+window_size[2],
                :,
                crop_idx
                ] = crop_pred[0, ...]  #remove batch dim

        if weight_map == None:
            weight_map = 1.0

        if aggregate_func == 'mean':
            summed_weights[
                x:x+window_size[0],
                y:y+window_size[1],
                z:z+window_size[2]
                ] += weight_map
    
    if aggregate_func == "mean":
        full_prediction /= summed_weights
    elif aggregate_func == "median":
        full_prediction = np.median(full_prediction, axis = 4)

    final_prediction = np.expand_dims(full_prediction, axis=3).astype(np.float32)
    print_("> Done collecting predictions:", full_prediction.shape)

    return final_prediction 


################################################################################


seg_path_list_file = "data/path_lists/pirads_4plus/seg_new.txt"

if __name__ == '__main__':

    # Fix tensorflow memory leak and multiprocessing error:
    disable_eager_execution()
    set_start_method("spawn")

    # Find model configs
    args = parse_input_args()
    m_configs = read_config(os.path.join(args.train_dir, "train_recon_config.yml"))

    DEBUG = args.debug
    print_(f">Is Debug: {DEBUG}")

    TRAIN_DIR = args.train_dir
    TEMP_DIR  = os.path.join(TRAIN_DIR, "temp")
    VAL_DIR = os.path.join(TRAIN_DIR, "val_set")

    # load validation indexes
    val_set_key = f"val_set{m_configs['train_set_key'][-1]}"
    indexes_dict = read_yaml_to_dict(m_configs['indexes_path'])
    val_idxs = indexes_dict[val_set_key]
    print_(f"> Validation idxs: {val_idxs}\n\twith length: {len(val_idxs)}")

    # Get filnames of validation files
    t2_files = [l.strip() for l in open(m_configs['train_path_list'])]
    seg_files = [l.strip() for l in open(seg_path_list_file)]

    if DEBUG:
        t2_files  = t2_files[:10]
        seg_files = seg_files[:10]
        val_idxs  = list(range(10))
    
    spacings = get_spacings(t2_files, val_set_key)
    VAL_DIR = os.path.join(TRAIN_DIR, "val_set", f"window_steps_{'_'.join([str(x) for x in args.xyz_steps])}")
    os.makedirs(VAL_DIR, exist_ok=True)

    # Load the model
    model = load_model(
        filepath = os.path.join(TRAIN_DIR, "models", args.model_fname),
        compile  = False
    )

    for val_idx in val_idxs:
        print_(f"\n>Calculating metrics on val image idx: {val_idx}...")
        print_(f"\t>T2 path: {t2_files[val_idx]}")
        print_(f"\t>seg path: {seg_files[val_idx]}")

        x_true_path = t2_files[val_idx]
        y_true_path = t2_files[val_idx]
        seg_path    = seg_files[val_idx]

        x_true_n, y_true_n = load_and_preprocess_t2(
            nifti_path      = x_true_path,
            sampling        = 1.0/m_configs['acceleration'],
            centre_sampling = m_configs['centre_sampling'],
            target_shape    = m_configs['target_shape'],
            target_space    = m_configs['target_space'],
            norm            = m_configs['normalization'],
            seed            = m_configs['seed'] 
        )

        print_stats_np(x_true_n, "xtrue")
        print_stats_np(y_true_n, "ytrue")

        if True:
            write_array2nifti_t2(x_true_n.squeeze(), TEMP_DIR, f"{val_idx}_xtrue.nii.gz")
            write_array2nifti_t2(y_true_n.squeeze(), TEMP_DIR, f"{val_idx}_ytrue.nii.gz")

        y_pred_n = predict_sliding_median(
            input_n        = x_true_n,
            models         = [model],
            window_size    = m_configs['window_size'],
            n_steps        = list(args.xyz_steps),
            n_out_channels = len(m_configs['out_seqs']),
            do_mirror_pred = args.do_mirror_pred,
            aggregate_func = 'mean',
        )

        write_to_file(
            val_idx = val_idx,
            X_true_n = x_true_n,
            Y_true_n = y_true_n,
            Y_pred_n = y_pred_n,
            spacing  = list(spacings[val_idx])
        )

        # Perform sliding window prediction on the validation image
        if DEBUG:
            print_("f>Breaking because of debug mode")
            break

        