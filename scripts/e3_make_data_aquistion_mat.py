from os import path
import h5py

from datetime import date
from yaml import dump
import numpy as np
from scipy import fftpack
import SimpleITK as sitk
from typing import List
import re
import sqlite3

from fastMRI_PCa.data import resample, center_crop, normalize
from fastMRI_PCa.utils import get_kspace_mask_exp_decay
from fastMRI_PCa.utils import print_p
from fastMRI_PCa.visualization import *
from fastMRI_PCa.utils import dump_dict_to_yaml


################################  README  ######################################
# OLD - This script will makes a dataset of T2 images for all patients from UMCG and
# RUMC. A mask in kspace is created and cropped according to their respective
# acquisition matrix found in the DICOM header. The data is storad into a .h5
# storage file.
################################################################################


def get_list_from_txt_file(path: str) -> List:
    """ Returns a list of all items on each line of the text file referenced in
        the given path
    
    Parameters:
    `path (str)`: path to the text file
    """
    return [line.strip() for line in open(path, "r")]


def get_acquisition_matrices(
    cur: sqlite3.Cursor,
    patient_id: str,
    tablename: str,
    verbatim=False):
    """ Returns a list of all k-space aquisition matrices for the given patient.
        The acquisition matrix values can be found in the given tablename and 
        should be searched for by using an SQLite Query.
    
    Parameters:
    `cur (Sqlite Cursor object)`: cursor object as an object by sqlite3 of an
        already connected database
    `patient_id (str)`: Unique patient id, used to be able to find the patient in
        the database.
    `tablename (str)`: Name of the table in an SQLite database.
    """

    # Define and execute query to find acquisition matrices per patient.
    query = f"SELECT [0018|1310] FROM {tablename} WHERE ([0008|103e] like '%tra%' or [path] like '%tra%') and ([0008|103e] like '%t2%' or [0008|103e] like '%T2%') and [0010|0020] like '%{patient_id}%';"
    results = cur.execute(query).fetchall() #list of tuples
    
    # Make list where parsed acquistion matrices will be stored.
    ac_matrices = []
    
    if verbatim:
        for idx, res in enumerate(results):
            print_p(f"DB results{idx} = {res}")

    # There can be more than one result. If patient had the same sequence acquired multiple times for example.
    for idx, res in enumerate(results):
        acq_mat = results[idx][0].split("\\")  # Acquisition Matrix in k-space

        # Multi-valued: frequency rows\frequency columns\phase rows\phase columns.
        freq_rows = int(acq_mat[0])
        freq_cols = int(acq_mat[1])
        phase_rows = int(acq_mat[2])
        phase_cols = int(acq_mat[3])

        ac_tup = (max(freq_rows, phase_rows), max(freq_cols, phase_cols))
        ac_matrices.append(ac_tup)
    return ac_matrices


def get_patient_specific_mask(
    cur: sqlite3.Cursor,
    patient_id: str,
    mask_perc: float,
    mri_dims: tuple,
    tablename: str,
    method: str,
    verbatim=False) -> np.ndarray:
    """ Obtain a binary mask for k-space for a specific patient.
    
    Parameters:
    `vcur (Sqlite Cursor object)`: cursor object as an object by sqlite3 of an
        already connected database
    `patient_id (str)`: Unique patient id, used to be able to find the patient in
        the database.
    `mask_perc (float)`: What percentage of the acquistion matrix should be
        masked? 0.75=75% and means only 25% of simulated k-space will be retained.
    `mri_dims (tuple)`: The dimenions of the input image (x, y, z)
    `tablename (str)`: Name of the table in an SQLite database.
    `method (str)`: Which type of k-space mask should be used?
        options=("rect", "exp_decay")

    Returns `mask (np.ndarray)`: A mask that can be applied to k-space for the
        given patient.
    """

    # Get acquisition matrices that match from DICOM database as a list
    acquistion_matrices = get_acquisition_matrices(
        cur=cur,
        patient_id=patient_id,
        tablename=tablename,
        verbatim=verbatim)

    for acquistion_matrix in acquistion_matrices:
        try:
            # Determine where the acquisition matrix should be located in image space.
            xdiff = abs(mri_dims[0] - acquistion_matrix[0])//2
            ydiff = abs(mri_dims[1] - acquistion_matrix[1])//2
            mask = np.zeros((mri_dims[0], mri_dims[1]))

            if method == "rect":
                cropped_mask = np.ones(acquistion_matrix)
            if method == "exp_decay":
                cropped_mask = get_kspace_mask_exp_decay(acquistion_matrix, mask_perc)
            
            if verbatim:
                print_p(f"\tacquisition matrix = {acquistion_matrix}")
                print_p(f"\timg dims = {mri_dims[0]}, {mri_dims[1]}")
                print_p(f"\txdiff = {xdiff}")
                print_p(f"\tydiff = {ydiff}")
                print_p(f"\tcropped_mask.shape[0] + xdiff = {cropped_mask.shape[0] + xdiff}")
                print_p(f"\tcropped_mask.shape[1] + ydiff = {cropped_mask.shape[1] + ydiff}")
            
            # Put the mask in the centre of mri dimensions.
            mask[xdiff:cropped_mask.shape[0] + xdiff, ydiff:cropped_mask.shape[1] + ydiff] = cropped_mask
            return mask
        except:
            print_p(f"The acquisition matrix does not fit in img space. ERROR. {acquistion_matrix}")
            continue
    return mask


def make_undersampled_dataset(
    path_list: str,
    target_dir: str,
    tablename: str,
    db_path: str,
    seq: str,
    target_shape: List[int] = [144, 144, 18],
    target_space: List[float] = [0.5, 0.5, 3.0],
    mask_perc = 0.5,
    norm: str = "znorm",
) -> None:
    """ Obtain a binary mask for k-space for a specific patient.
    
    Parameters:
    `path_list (str)`: Path to text file with dicom paths on each line.
        (data to be read into memory)
    `target_dir (str)`: Directory where the dataset will be saved
    `tablename (str)`: Name of SQLite table where dicoms headers are stored
    `db_path (str)`: Path to the database (.db) file
    `seq (str)`: Short name of the sequence (Used for naming)
        Options = ("t2")
    `target_shape (List[str])`: The x, y, z dimensions of the centre crop.
        To make each image/3D volume the same dimensions.
    `target_space (List[float])`: The volume will be resampled according to the
        given list in x, y, z dimensions
    `mask_perc (float)`: What percentage of the image should be masked?
        Example: 0.75=75% --> 25% of kspace will be retained. Acceleration = 4x
    `norm (str)`: The normalization method used for the label and input.
        options=("znorm", "rescale01")

    Returns: None
    """

    # Dump params to .yaml so that later we know what params went into dataset construction
    params = {
        "path_list": path_list,
        "target_dir": target_dir,
        "tablename": tablename,
        "db_path": db_path,
        "target_shape": target_shape,
        "target_space": target_space,
        "sequence": seq,
        "mask_percentage": mask_perc,
        "normalization": norm,
        "datetime": date.today().strftime("%Y-%m-%d")}
    dump_dict_to_yaml(params, target_dir, filename=f"settings_{seq}_u{int(mask_perc*100)}")

    # Create dataset storage file
    STORAGE_FILE = f"{target_dir}/storage_{seq}_u{int(mask_perc*100)}.h5"
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

    # Determine regex pattern to determinen patient ID based on path of file.
    regex_patient_pattern = r'[0-9]+-[A-Z]-[0-9]+|pat[0-9]{4}'
    p = re.compile(regex_patient_pattern)

    # Create connection with DICOM sqlite DB, we'll take acquisition matrices
    con = sqlite3.connect(db_path)
    cur = con.cursor()

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
        
        # Build patient mask based on dims of the acquisition matrix in k-space
        mask = get_patient_specific_mask(
            cur=cur,
            patient_id=patient_id,
            mask_perc=mask_perc,
            mri_dims=t2_tra_n.shape,
            tablename = tablename,
            method="exp_decay",
            verbatim=True)

        # Loop over slices
        for slice_idx in range(t2_tra_n.shape[2]):

            slice = t2_tra_n[:, :, slice_idx]
            # save_slice_2d(slice, "reports/figures/temp/", "1")

            # Transform to k-space (FFT)
            kspace = fftpack.fftn(slice)
            # save_kspace(kspace, "reports/figures/temp/", "2", True)

            # Shift kspace (complex part is needed)
            kspace_shift = fftpack.fftshift(kspace)
            # save_kspace(kspace_shift, "reports/figures/temp/", "3", True)

            # Undersample/Mask kspace
            m_kspace = kspace_shift * mask  # masked k-space
            # save_kspace(m_kspace, "reports/figures/temp/", "4", True)

            # Shift kspace back (with complex part)
            m_kspace_t = fftpack.ifftshift(m_kspace)
            # save_kspace(m_kspace_t, "reports/figures/temp/", "5", True)
            
            # Transform back to image space (IFFT)
            slice_recon = np.real(fftpack.ifft2(m_kspace_t))
            # save_slice_2d(slice_recon, "reports/figures/temp/", "6")

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


#########################################


make_undersampled_dataset(
    path_list = r"data/combined_t2_list.txt",
    target_dir = r"data/interim/diff_norm",
    tablename = "dicom_headers_v2",
    db_path = r'sqliteDB/dicoms.db',
    seq = "t2",
    target_shape = [256, 256, 24],
    target_space = [0.5, 0.5, 3.0],
    mask_perc = 0.75,
    norm = "rescale_0_1"
    )