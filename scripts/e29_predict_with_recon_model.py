import argparse
from operator import mod
import numpy as np
import multiprocessing
import time
import SimpleITK as sitk
import os

from typing import List
from functools import partial
from tensorflow.keras.models import load_model
import copy

from fastMRI_PCa.data import resample, center_crop, normalize, undersample
from fastMRI_PCa.utils import print_p, read_yaml_to_dict, list_from_file
from fastMRI_PCa.utils import print_stats_np, get_rand_exp_decay_mask
from fastMRI_PCa.utils.utils import create_dirs_if_not_exists
# from fastMRI_PCa.visualization import save_slice_3d
from fastMRI_PCa.models import ssim_loss, ssim_metric, psnr_metric



if __name__ == '__main__':

    #params
    nifti_inp_path    = os.path.join('temp', 'test_t2.nii.gz')
    recon_model_fpath = os.path.join('models', '29_recon_r4_mag', 'best-direct-fold0_val_loss.h5')
    nifti_out_path1    = os.path.join('temp', 'test_t2_recon.nii.gz')
    nifti_out_path2    = os.path.join('temp', 'test_t2_crop.nii.gz')
    target_shape = (192, 192, 20)
    target_space = (0.5, 0.5, 3.0)
    norm         = 'rescale_0_1'

    # Load model
    recon_model = load_model(recon_model_fpath, custom_objects={'ssim_loss': ssim_loss, 'ssim_metric': ssim_metric, 'psnr_metric': psnr_metric})
    recon_model.summary()

    t2_tra_s = sitk.ReadImage(nifti_inp_path, sitk.sitkFloat32)
    t2_tra_s.SetSpacing(target_space)
    t2_tra_s_copy = copy.deepcopy(t2_tra_s)

    # preprocessing
    t2_tra_s = resample(
        image       = t2_tra_s, 
        min_shape   = target_shape, 
        method      = sitk.sitkNearestNeighbor, 
        new_spacing = target_space
    )
    t2_tra_s = center_crop(t2_tra_s, target_shape)
    t2_tra_s = normalize(norm_str=norm, img_s=t2_tra_s)
    t2_tra_n = sitk.GetArrayFromImage(t2_tra_s).T

    # predict
    data = np.expand_dims(t2_tra_n, 0)
    recons = recon_model.predict(data)

    # write to file
    for mri_idx in range(recons.shape[0]):
        print(f"Saving MRI num {mri_idx+1}/{recons.shape[0]}.  -> {nifti_out_path1}")
        pred_s = sitk.GetImageFromArray(np.squeeze(recons[mri_idx, ...]).T)
        pred_s.CopyInformation(t2_tra_s)
        pred_s = center_crop(pred_s, (160, 160, 8))
        sitk.WriteImage(pred_s, nifti_out_path1)

    t2_tra_s_copy = center_crop(t2_tra_s_copy, (160, 160, 8))
    sitk.WriteImage(t2_tra_s_copy, nifti_out_path2)


    print("done")