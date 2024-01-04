import h5py

from datetime import date
import numpy as np
from scipy import fftpack
import SimpleITK as sitk
from typing import List
import re

from fastMRI_PCa.data import resample, center_crop, normalize
from fastMRI_PCa.utils import print_p
from fastMRI_PCa.visualization import *
from fastMRI_PCa.utils import dump_dict_to_yaml
from fastMRI_PCa.utils import get_rand_exp_decay_mask


################################  README  ######################################
# OLD - This script will makes a dataset of T2 images for all patients from UMCG
# and RUMC. The data is storad into a .h5 storage file. The applied k-space mask
# is the exponential distribution.
################################################################################


def get_list_from_txt_file(path: str) -> List:
    """ Returns a list of all items on each line of the text file referenced in
        the given path
    
    Parameters:
    `path (str)`: path to the text file
    """
    return [line.strip() for line in open(path, "r")]


def make_undersampled_dataset(
    path_list: str,
    target_dir: str,
    seq: str,
    target_shape: List[int] = [144, 144, 18],
    target_space: List[float] = [0.5, 0.5, 3.0],
    sampling: float = 0.5,
    centre_sampling: float = 0.5,
    norm: str = "znorm"
        ) -> None:
    """ Obtain a binary mask for k-space for a specific patient.
    
    Parameters:
    `path_list (str)`: Path to text file with dicom paths on each line.
        (data to be read into memory)
    `target_dir (str)`: Directory where the dataset will be saved
    `seq (str)`: Short name of the sequence (Used for naming)
        Options = ("t2")
    `target_shape (List[str])`: The x, y, z dimensions of the centre crop.
        To make each image/3D volume the same dimensions.
    `target_space (List[float])`: The volume will be resampled according to the
        given list in x, y, z dimensions
    `sampling (float)`: What percentage of the image should sampled in kspace?
        Example: 0.25=25% --> 75% of kspace will be masked away.
        Acceleration = 4x
    `centre_sampling` (float): Percentage of samples that should lie in the
        central region of k-space
    `norm (str)`: The normalization method used for the label and input.
        options=("znorm", "rescale01")

    Returns: None
    """

    # Dump params to .yaml so that later we know what params went into dataset construction
    params = {
        "path_list": path_list,
        "target_dir": target_dir,
        "target_shape": target_shape,
        "target_space": target_space,
        "sequence": seq,
        "sampling": sampling,
        "centre_sampling": centre_sampling,
        "normalization": norm,
        "datetime": date.today().strftime("%Y-%m-%d")
    }
    dump_dict_to_yaml(params, target_dir, filename=f"settings_{seq}_samp{int(sampling*100)}")

    # Create dataset storage file
    STORAGE_FILE = f"{target_dir}/storage_{seq}_samp{int(sampling*100)}.h5"
    hdf5_file = h5py.File(STORAGE_FILE, "w")
    storage = {}

    # Create storage file for reconstruction 
    storage[seq] = hdf5_file.create_dataset(
        name=seq, 
        shape=(0,) + (target_shape[0], target_shape[1], target_shape[2]), 
        dtype='f', 
        chunks=(1,) + (target_shape[0], target_shape[1], target_shape[2]), 
        maxshape=(None,) + (target_shape[0], target_shape[1], target_shape[2]), 
        compression="lzf")

    # Create storeage file for the label of the reconstruction. --> the original
    storage[f"{seq}_label"] = hdf5_file.create_dataset(
        name=seq + "_label", 
        shape=(0,) + (target_shape[0], target_shape[1], target_shape[2]), 
        dtype='f', 
        chunks=(1,) + (target_shape[0], target_shape[1], target_shape[2]), 
        maxshape=(None,) + (target_shape[0], target_shape[1], target_shape[2]), 
        compression="lzf")

    # Find all patient files
    nifti_paths = get_list_from_txt_file(path_list)
    print_p(f"Found {len(nifti_paths)} MRI scans. Sequence: {seq}")

    # Determine regex pattern to determine patient ID based on path of file.
    regex_patient_pattern = r'[0-9]+-[A-Z]-[0-9]+|pat[0-9]{4}'
    p = re.compile(regex_patient_pattern)

    # Start looping over files
    for mri_idx, nifti_path in enumerate(nifti_paths):

        # Get patient ID
        patient_id = p.search(nifti_path).group(0)
        print_p(f"\nProcessing mri {mri_idx}/{len(nifti_paths)} - {patient_id}")
        
        # Read the T2 transversal file as SimpleITK object
        t2_tra_s = sitk.ReadImage(nifti_path, sitk.sitkFloat32)

        # Convert to Numpy
        t2_tra_n = sitk.GetArrayFromImage(t2_tra_s).T

        # Make the label and make room for it
        t2_tra_s_label = resample(t2_tra_s, 
                                min_shape=target_shape, 
                                method=sitk.sitkNearestNeighbor, 
                                new_spacing=target_space)
        t2_tra_s_label = center_crop(t2_tra_s_label, target_shape)
        t2_tra_s_label = normalize(norm_str=norm, img_s=t2_tra_s_label)

        storage[seq+"_label"].resize(storage[seq+"_label"].shape[0] + 1, axis=0)
        storage[seq+"_label"][mri_idx] = sitk.GetArrayFromImage(t2_tra_s_label).T

        # Reconstruction has same dimenions as original nifti/MR image
        recon_n = np.zeros(t2_tra_n.shape)
        
        # Build patient mask
        mask = get_rand_exp_decay_mask(width=t2_tra_n.shape[0],
                                       height=t2_tra_n.shape[1],
                                       sampling=sampling,
                                       centre_sampling=centre_sampling,
                                       exp_scale=0.4,
                                       seed=3478,
                                       verbatim=False
        )

        # Loop over slices
        for slice_idx in range(t2_tra_n.shape[2]):

            slice = t2_tra_n[:, :, slice_idx]
            # if slice_idx == 10:
            #     save_slice_2d(slice, "temp/", "1_inp")

            # Transform to k-space (FFT)
            kspace = fftpack.fftn(slice)
            # if slice_idx == 10:
            #     save_kspace(kspace, "temp/", "2_kspace", True)

            # Shift kspace (complex part is needed)
            kspace_shift = fftpack.fftshift(kspace)
            # if slice_idx == 10:
            #     save_kspace(kspace_shift, "temp/", "3_kspace_shift", True)

            # Undersample/Mask kspace
            m_kspace = kspace_shift * mask  # masked k-space
            # if slice_idx == 10:
            #     save_kspace(m_kspace, "temp/", "4_kspace_shift_mask", True)

            # Shift kspace back (with complex part)
            m_kspace_t = fftpack.ifftshift(m_kspace)
            # if slice_idx == 10:
            #     save_kspace(m_kspace_t, "temp/", "5_masked_kspace", True)
            
            # Transform back to image space (IFFT)
            slice_recon = np.real(fftpack.ifft2(m_kspace_t))
            # if slice_idx == 10:
            #     save_slice_2d(slice_recon, "temp/", "6_out")

            # Assign slice to 4D volume.
            recon_n[:, : , slice_idx] = slice_recon

        # Transform to sitk and copy original img metadata to reconstruction.
        recon_s = sitk.GetImageFromArray(recon_n.T)
        recon_s.CopyInformation(t2_tra_s)

        # Resample, center crop and z-normalization all on sitk objects.
        recon_s = resample(recon_s, 
                            min_shape=target_shape, 
                            method=sitk.sitkNearestNeighbor, 
                            new_spacing=target_space)
        recon_s = center_crop(recon_s, target_shape)
        recon_s = normalize(norm_str=norm, img_s=recon_s)

        # Make room for the next image and assign it
        storage[seq].resize(storage[seq].shape[0] + 1, axis=0)
        storage[seq][mri_idx] = sitk.GetArrayFromImage(recon_s).T

    hdf5_file.close()


################################################################################


make_undersampled_dataset(
    path_list = r"data/combined_t2_list.txt",
    target_dir = r"data/interim/3_rand_exp_dist",
    seq = "t2",
    target_shape = [256, 256, 24],
    target_space = [0.5, 0.5, 3.0],
    sampling = 0.25,
    norm = "rescale_0_1"
    )

print_p(" -- DONE --")