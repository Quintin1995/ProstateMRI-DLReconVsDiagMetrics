import numpy as np
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from typing import Tuple

from fastMRI_PCa.utils import print_p


def compare_volumes(
    vol_true: np.ndarray,
    vol_test: np.ndarray,
    verbose: False
) -> Tuple[float, float, float]:
    """ Compares the given volumes with metrics: MSE, SSIM and PSNR. Metrics are
    averaged for all slices. 
    
    Parameters:
    vol_true (np.ndarray): 3D numpy array (mri volume)
    vol_test (np.ndarray): 3D numpy array (mri volume)
    """
    
    assert vol_true.shape == vol_test.shape, "Shapes of 3D volumes are not equal"

    num_metrics = 3  # MSE, SSIM, PSNR
    metrics = np.empty((num_metrics, vol_true.shape[2]), dtype = np.float32)

    data_range = np.abs(
        max(np.max(vol_true), np.max(vol_test)) - 
        min(np.min(vol_true), np.min(vol_test)))

    for slice_idx in range(vol_true.shape[2]):
        metrics[0, slice_idx] = mse(
            vol_true[:, :, slice_idx],
            vol_test[:, :, slice_idx])
        metrics[1, slice_idx] = ssim(
            vol_true[:, :, slice_idx],
            vol_test[:, :, slice_idx])
        metrics[2, slice_idx] = psnr(
            vol_true[:, :, slice_idx],
            vol_test[:, :, slice_idx],
            data_range = data_range)

    means = np.mean(metrics, axis = 1)

    if verbose:
        print_p(f"\tMSE: {round(means[0], 3)}")
        print_p(f"\tSSIM: {round(means[1], 3)}")
        print_p(f"\tPSNR: {round(means[2], 3)}")

    return means[0], means[1], means[2]