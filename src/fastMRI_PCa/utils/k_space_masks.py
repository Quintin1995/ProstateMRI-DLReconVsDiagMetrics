import numpy as np
import math
import re
import sqlite3
from umcglib.utils import print_
import os
from umcglib.kspace import write_np2nifti

def mask_vector_exp_decay(dim = 20, power = 0.5) -> tuple:
    """Constructs a list/vector of dimension dim with 'power' determining the
    decay of 1's occuring in this vector from left to right.
        
    Parameters:
    `dim (int)`: The length of the returned vector.
    `power (float)`: The power of an exponential function. It determines how
    quickly 1's are reduced further to the right of the created vector.
    
    Returns:
    `tuple (list, float)`: (vector, undersampling_percentage)
    """

    # Initialize empty list of half mask size
    v = [0 for _ in range(dim)]
    wait = 0
    
    for i in range(dim):
        # When waiting time reaches 0, add 1 to list.
        # Otherwise add a 0, and update wait according to exponential function.
        if wait < 1:
            v[i] = 1
            wait = i/dim**power
        else:
            wait -= 1

    # Calculate undersampling proportion
    r = 1 - sum(v) / dim
    return v, r


def mask_from_vector_exp_decay(vec, y_dim) -> np.ndarray:
    """ Creates a 2D numpy array mask from a single vector.
    
    Parameters:
    `vec (list)`: A list/vector (0.5 width of the final mask) of ones and zeros
    that will be copied reversed and concatenated into a longer list and
    extended upwards to create a 2d mask
    `y_dim (int)`: The length of the extension upwards (height)

    Returns:
    `mask (np.ndarray)`: 2D numpy mask with 1's and 0's 
    """

    vec = np.asarray(vec, dtype = np.uint8)
    mask = np.zeros((y_dim, vec.shape[0] * 2))

    mask[:, 0:vec.shape[0]] = vec[None, ::-1]
    mask[:, vec.shape[0]:]  = vec[None, :]
    
    return mask


def optimize_mask_exp_decay(
    x_dim,
    y_dim,
    undersampling_target = 0.1,
    steps = 100,
    verbatim=False):
    """ Optimizes a vector of 1's and 0's to best fit the wanted undersampling
    percentage. This is done by random increments in the power function that
    creates a mask vector
    
    Parameters:
    `x_dim (int)`: width of desired mask
    `y_dim (int)`: height of desired mask
    `undersampling_target (float)`: desired undersampling percentage (0.1 = 10%)
    So 10% of the mask will contain a 1.0
    `steps (int)`: How many times can the algorithm retry to find a better vector
    to approximate the undersampling target

    Returns:
    `mask (np.ndarray)`: 2D numpy mask with 1's and 0's and a good approximation
    of the undersampling target
    `r (float)`: Best approximation of undersampling target percentage
    """

    best = {'diff': float('inf'), 'r': -1, "vec" : [], 'pow': 1}
    power = 0
    for i in range(1, steps):
        vec, r = mask_vector_exp_decay(x_dim // 2, power)
        diff = abs(undersampling_target - r)
        if diff < best['diff']:
            if verbatim:
                print(f"> step {i}: new best")
                print(f"> Diff {diff}; r {r}; pow {power}")
            best = {'diff': diff, 'r': r, 'vec': vec, 'pow': power}
        else:
            power = (best['pow'] + (0.5 - np.random.random()))
    mask = mask_from_vector_exp_decay(best['vec'], y_dim)
    
    return mask, best['r']


def get_kspace_mask_exp_decay(
    dims: tuple,
    perc: float,
    verbatim=False,
    seed=3478) -> np.ndarray:
    """ Optimize a numpy mask with dims as dimensions (2D) where percentage of
    the mask is 1.0 and the rest is zero filled.
    The mask is vertical. So only vertical lines in k-space will be masked.

    Parameters:
    `dims (tuple)`: width and height of the mask
    `perc (float)`: percentage of mask that will be masked. Float [0.0 and 1.0.]
    `seed (int)`: seed for the np.random.random().

    Returns:
    `mask (np.ndarray)`: 2D numpy mask with 1's and 0's and a good approximation
    of the undersampling target
    """
    
    np.random.seed(seed)
    print(f"Exponential Decay k-space mask: {dims[0]}, {dims[1]}")

    mask, r = optimize_mask_exp_decay(
        x_dim=dims[1], 
        y_dim=dims[0], 
        undersampling_target=perc, 
        steps=1000)

    if verbatim:
        print(f"Undersampling: {r}")

    return mask


def get_kspace_mask_rect(width: int, height: int) -> np.ndarray:
    """ Returns a rectangular mask for k-space where the centre of the mask are
    1's. The given width and height are 2x zero padded. The centre 25% are 1s.
    
    Parameters:
    `width (int)`: Width of the mask
    `height (int)`: Height of the mask
    """

    assert width == height, "Widht en height of rectangular mask are not equal."
    print(f"Rectangular k-space mask: {width}, {height}")
    w, h = width, height
    mask = np.zeros((w, h))
    mask[int(w*0.25) : int(w*0.75), int(h*0.25) : int(h*0.75)] = 1.0
    return mask


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Inverts the given mask. 0.0 becomes 1.0 and 1.0 becomes 0.0.
    
    Parameters:
    `mask (np.ndarray)`: Binary 2D array.
    """
    return np.abs(mask - 1)


def get_acquisition_matrices(
    cur: sqlite3.Cursor,
    patient_id: str,
    tablename: str,
    verbatim: bool = False,
):
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
    # query = f"SELECT [0018|1310] FROM {tablename} WHERE ([0008|103e] like '%tra%' or [path] like '%tra%') and ([0008|103e] like '%t2%' or [0008|103e] like '%T2%') and [0010|0020] like '%{patient_id}%' and [0018|1030] not like '%snel_bij bewogen%';"
    query = f"SELECT [0018|1310] FROM {tablename} WHERE ([0008|103e] like '%tra%' or [path] like '%tra%') and ([0008|103e] like '%t2%' or [0008|103e] like '%T2%') and [0010|0020] like '%{patient_id}%' order by [0018|1030];"
    
    results = cur.execute(query).fetchall() #list of tuples
    
    # Make list where parsed acquistion matrices will be stored.
    ac_matrices = []
    
    if verbatim:
        for idx, res in enumerate(results):
            print_(f"DB results{idx} = {res}")

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


def get_rand_exp_decay_mask_ac_matrix(
    width: int,
    height: int,
    sampling: float,
    centre_sampling: float,
    seed: int,
    nifti_path: str,
    dicom_db_path: str,
    exp_scale: float = 0.4,  # determined emperically
    tablename: str = 'dicom_headers_v2',
    verbatim = False,
):

    # Find the patient ID in the nifti path with regex
    regex_patient_pattern = r'[0-9]+-[A-Z]-[0-9]+|pat[0-9]{4}'
    p = re.compile(regex_patient_pattern)
    patient_id = p.search(nifti_path).group(0)

    con = sqlite3.connect(dicom_db_path)
    cur = con.cursor()

    # Obtain the most relevant k-space acquistion matrix as list of tuples from the DICOM header database
    acquistion_matrices = get_acquisition_matrices(
        cur        = cur,
        patient_id = patient_id,
        tablename  = tablename,
        verbatim   = verbatim,
    )

    if len(acquistion_matrices) == 0:
        print("ERROR no acquisition matrix found\n"*30)
        print(nifti_path)

    if False:
        print_(f"\t>Found acquistion matrix: {acquistion_matrices}")
        print_(f"\t>path : {nifti_path}")

    for ac_shape in acquistion_matrices:

        # Add z-dim to the mask (is expected by the poisson mask 3D function)
        ac_shape = (ac_shape[0], ac_shape[1], 1)

        # Try to fit the first acquistion matrix in the MRI image, otherwise continue with the next one and see if that one fits.
        try:
            # Determine where the acquisition matrix should be located in image space.
            xdiff = abs(width - ac_shape[0])//2
            ydiff = abs(height - ac_shape[1])//2
            mask = np.zeros((width, height))

            # Obtain the k-space mask from .npy files or create them (this is a slow function) It is a advised to pre-create them.
            cropped_mask = get_rand_exp_decay_mask(
                width           = ac_shape[1],
                height          = ac_shape[0],
                sampling        = sampling,
                centre_sampling = centre_sampling,
                seed            = seed,
                exp_scale       = exp_scale,  # determined emperically
                verbatim        = verbatim,
            )

            # Do some printing for debugging
            if verbatim:
                write_np2nifti(cropped_mask, os.path.join('temp', 'cropped_mask.nii.gz'))
                print_(f"\t>expected sampling: {sampling}")
                print_(f"\t>actual   sampling: {np.sum(cropped_mask)/(cropped_mask.shape[0]*cropped_mask.shape[1])}")
            
            if verbatim:
                print_(f"\tacquisition matrix = {ac_shape}")
                print_(f"\timg dims = {width}, {height}")
                print_(f"\txdiff = {xdiff}")
                print_(f"\tydiff = {ydiff}")
                print_(f"\tcropped_mask.shape[0] + xdiff = {cropped_mask.shape[0] + xdiff}")
                print_(f"\tcropped_mask.shape[1] + ydiff = {cropped_mask.shape[1] + ydiff}")
            
            # Put the mask in the centre of mri dimensions.
            mask[xdiff:cropped_mask.shape[0] + xdiff, ydiff:cropped_mask.shape[1] + ydiff] = cropped_mask.squeeze()

            if verbatim:
                write_np2nifti(cropped_mask, os.path.join('temp', 'cropped_mask_in_kspace.nii.gz'))

            return mask
        
        except:
            print_(f"ERROR - The acquisition matrix does not fit in img space. ERROR. ac_shape: {ac_shape} and crop shape: {cropped_mask.shape}")
            continue

    return mask


def get_rand_exp_decay_mask(
    width: int,
    height: int,
    sampling: float,
    centre_sampling: float,
    seed: int,
    exp_scale: float = 0.4,  # determined emperically
    verbatim = False
):  
    np.random.seed(seed)

    # Create height vector - because of horizontal mask.
    vec = np.zeros((height,))

    # Set Central region to 1.0
    central_region_perc = sampling * centre_sampling
    left_idx  = math.ceil(height//2 - height//2 * central_region_perc)
    right_idx = math.ceil(height//2 + height//2 * central_region_perc)
    vec[left_idx : right_idx] = 1.0

    # Add a k-space line until sampling percentage is reached
    while sum(vec)/height < sampling:
        idx = np.random.exponential(exp_scale, 1)
        idx = int(idx * left_idx)
        
        if np.random.random() > 0.5:
            idx = left_idx - idx
        else:
            idx = right_idx + idx
        try:
            vec[idx] = 1.0
        except:
            continue

    mask = np.zeros((height, width))
    mask[:, :] = vec[:, np.newaxis]

    if verbatim:
        print(f"\nAccomplished undersampling percentage = {sum(vec)/height * 100}%")

    return mask