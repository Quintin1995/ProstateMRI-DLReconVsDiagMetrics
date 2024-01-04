import argparse
import numpy as np
import multiprocessing
import time
import SimpleITK as sitk
import os

from typing import List
from functools import partial
from tensorflow.keras.models import load_model

from fastMRI_PCa.data import resample, center_crop, normalize, undersample
from fastMRI_PCa.utils import print_p, read_yaml_to_dict, list_from_file
from fastMRI_PCa.utils import print_stats_np, get_rand_exp_decay_mask
from fastMRI_PCa.utils.utils import create_dirs_if_not_exists
from fastMRI_PCa.visualization import save_slice_3d
from fastMRI_PCa.models import ssim_loss, ssim_metric, psnr_metric


################################  README  ######################################
# OLD
# This script predict the validation set on the given model. The
# predictions are stored in the given train_output folder.


def parse_input_args():

    help = """This script will load the following data: DWI images, ADC maps and
              undersampled T2w images. The data will be used to train a
              diagnostic network where lesions are segmented for the patient.
              Goal: Train a diagnostic U-Net."""

    parser = argparse.ArgumentParser(description=help)

    parser.add_argument('--model_dir',
                        type=str,
                        required=True,
                        help='Directory where the predictions will be saved. At the same time the name of the model folder where the saved model can be found.')

    args = parser.parse_args()
    return args


def parallel_loading_preprocess(
    nifti_paths: List[str],
    sampling: float,
    centre_sampling: float,
    target_shape: List[int],
    target_space: List[float],
    norm: str,
    do_undersample: bool):

    t = time.time()
    print(f"\nT2s: Start of parralel loading and preprocessing...")

    # Start a pool of workers.
    pool = multiprocessing.Pool(processes=N_CPUS)

    # Define a partial function that can undergo a pool.map function.
    load_and_preprocess_partial = partial(load_and_preprocess_t2,
                                          sampling=sampling,
                                          centre_sampling = centre_sampling,
                                          target_shape=target_shape,
                                          target_space=target_space,
                                          norm=norm,
                                          do_undersample=do_undersample)

    # Apply the load and preprocess function for each file in the given paths
    data_list = pool.map(load_and_preprocess_partial, nifti_paths)
    
    # Aggregate the data in the first axis.
    data = np.stack(data_list, axis=0)
    data = np.expand_dims(data, 4)

    print(f"Time parallel loading T2s time: {time.time() - t} seconds")
    print_stats_np(data, f"T2 numpy stats")

    return data


def load_and_preprocess_t2(
    nifti_path: str,
    sampling: float,
    centre_sampling: float,
    target_shape: List[int],
    target_space: List[float],
    norm: str,
    do_undersample: bool):

    print_p(f"Processing T2 mri: {nifti_path}")
    
    # Read the T2 transversal file as SimpleITK object
    t2_tra_s = sitk.ReadImage(nifti_path, sitk.sitkFloat32)

    if do_undersample and centre_sampling != 1.0:
        # Build patient mask
        mask = get_rand_exp_decay_mask(width=t2_tra_s.GetSize()[0],
                                    height=t2_tra_s.GetSize()[1],
                                    sampling=sampling,
                                    centre_sampling=centre_sampling,
                                    seed=SEED,
                                    verbatim=False)

        # Perform undersampling in k-space with the given mask.
        t2_tra_s = undersample(t2_tra_s, mask)

    # Preprocess
    t2_tra_s = resample(t2_tra_s, 
                        min_shape=target_shape, 
                        method=sitk.sitkNearestNeighbor, 
                        new_spacing=target_space)
    t2_tra_s = center_crop(t2_tra_s, target_shape)
    t2_tra_s = normalize(norm_str=norm, img_s=t2_tra_s)

    return sitk.GetArrayFromImage(t2_tra_s).T



################################################################################
# Path to file with train, validation and test indexes.
IDX_YAML_PATH = r"data/path_lists/pirads_4plus/train_val_test_idxs.yml"

# File with paths to each file.
t2_path_list_file = "data/path_lists/pirads_4plus/current_t2w.txt" 

# Number of CPUs used for loading, preprocessing and undersampling in kspace
N_CPUS = 3

SEED   = 3478

DEBUG = True

if __name__ == '__main__':
    
    args = parse_input_args()
    print_p(f"\nAll input parameters: {args}\n")

    if DEBUG:
        os.environ["CUDA_VISIBLE_DEVICES"]="1"

    # Load reconstruction model parameters
    recon_path = f"train_output/{args.model_dir}/params.yml"
    recon_params = read_yaml_to_dict(recon_path)

    # Load deterministic validation set
    indexes_dict = read_yaml_to_dict(IDX_YAML_PATH)
    idxs_set_name = "val_set0"
    val_idxs = indexes_dict[idxs_set_name]
    print(f"validation set indexes {val_idxs}\nWith length: {len(val_idxs)}")

    # Load the T2 images in parallel. Load, undersample and preprocess
    t2_paths = list(np.asarray(list_from_file(t2_path_list_file))[val_idxs])
    print(f"Found {len(t2_paths)} in {idxs_set_name}")
    perc_load = 0.07 if DEBUG else 1.0  # decrease num imgs if in debug mode.
    print_p(f"Going to load MRIs (number): {int(perc_load*len(t2_paths))}...")
    
    X_val = parallel_loading_preprocess(nifti_paths=t2_paths[:int(perc_load*len(t2_paths))],
                                        sampling=(1/recon_params["acceleration"]),
                                        centre_sampling=recon_params["centre_sampling"],
                                        target_shape=recon_params["target_shape"],
                                        target_space=recon_params["target_space"],
                                        norm=recon_params["norm"],
                                        do_undersample=True)

    # No undersampling for the label. Load and preprocess
    Y_val = parallel_loading_preprocess(nifti_paths=t2_paths[:int(perc_load*len(t2_paths))],
                                        sampling=(1/recon_params["acceleration"]),
                                        centre_sampling=recon_params["centre_sampling"],
                                        target_shape=recon_params["target_shape"],
                                        target_space=recon_params["target_space"],
                                        norm=recon_params["norm"],
                                        do_undersample=False)

    if DEBUG:
        for mri_idx in range(1):
            slice_idx = 10
            create_dirs_if_not_exists(f"temp/{args.model_dir}")
            save_slice_3d(np.squeeze(X_val[mri_idx]), slice_idx, f"temp/{args.model_dir}", f"mri{mri_idx}_slice{slice_idx}_Xvalset0")
            save_slice_3d(np.squeeze(Y_val[mri_idx]), slice_idx, f"temp/{args.model_dir}", f"mri{mri_idx}_slice{slice_idx}_Yvalset0")

    # Load T2 reconstruction model.
    recon_model = load_model(f"models/{args.model_dir}/best-direct-fold0_val_loss.h5", custom_objects={'ssim_loss': ssim_loss, 'ssim_metric': ssim_metric, 'psnr_metric': psnr_metric})
    recon_model.summary()

    # Predict data
    print_p("Prediction on batch...")
    Y_val_pred = recon_model.predict(X_val, batch_size=1)

    # Save data
    outdir = f"train_output/{args.model_dir}/preds_{idxs_set_name}"
    create_dirs_if_not_exists(outdir)
    for mri_idx in range(Y_val_pred.shape[0]):
        fname = f"{outdir}/{mri_idx}_pred.nii.gz"
        print(f"Saving MRI num {mri_idx+1}/{Y_val_pred.shape[0]}.  -> {fname}")
        save_slice_3d(np.squeeze(Y_val_pred[mri_idx, ...]), 12, outdir, f"pred_{mri_idx}_slice{12}")
        pred_s = sitk.GetImageFromArray(np.squeeze(Y_val_pred[mri_idx, ...]).T)
        pred_s.SetSpacing([0.5, 0.5, 3.0])
        sitk.WriteImage(pred_s, fname)

        if True:
            save_slice_3d(np.squeeze(Y_val[mri_idx, ...]), 12, outdir, f"label_{mri_idx}_slice{12}")
            fname = f"{outdir}/{mri_idx}_r{recon_params['acceleration']}_label.nii.gz"
            print(f"Saving MRI label {mri_idx+1}/{Y_val_pred.shape[0]}.  -> {fname}")
            pred_s = sitk.GetImageFromArray(np.squeeze(Y_val[mri_idx, ...]).T)
            pred_s.SetSpacing([0.5, 0.5, 3.0])
            sitk.WriteImage(pred_s, fname)