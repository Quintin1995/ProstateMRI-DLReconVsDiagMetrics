import argparse
import os
from sklearn.model_selection import KFold
from typing import List, Tuple, Optional, Iterator
import numpy as np
import SimpleITK as sitk
from shutil import copyfile
import time
import multiprocessing
from functools import partial
from multiprocessing import set_start_method
import copy
import re
import sqlite3
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv3D, concatenate, Activation, add
from tensorflow.keras.layers import Conv3DTranspose, LeakyReLU, Dense, multiply
from tensorflow.keras.layers import MaxPooling3D, UpSampling3D, Permute, Reshape
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1 import disable_eager_execution

import umcglib.images as im
from umcglib.utils import print_stats_np, print_stats_np_list, read_yaml_to_dict, set_gpu, print_
from umcglib.augment import augment
from umcglib.losses import ssim_loss, get_msssim_loss
from umcglib.metrics import ssim_metric, psnr_metric
from umcglib.kspace import get_poisson_mask_3d, write_np2nifti
from fastMRI_PCa.utils.k_space_masks import get_rand_exp_decay_mask_ac_matrix
from umcglib.utils import get_random_str
from umcglib.plotting import save_array2d_to_image

################################  README  ######################################
# NEW - This script will train a T2W U-Net reconstruction model with a window
# based train method.
################################################################################


def parse_input_args():
    parser = argparse.ArgumentParser(description='Argument parser for the reconstruction model.')

    parser.add_argument(
        '-c',
        '--config_fpath',
        type=str,
        required=True,
        help='Path to config file with hyper parameters for a reconstruction training session.',
    )

    args = parser.parse_args()

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


def write_array2nifti_t2(
    mri_vol: np.ndarray,
    figdir: str,
    filename: str,
    target_space: List[float] = [0.5, 0.5, 3.0]
) -> None:
    """ Writes a nifti (T2) to file based on a 3D numpy array. Target spacing is
    set if it is not given.
    
    `mri_vol (np.ndarray)`: Mri volume. 3D numpy array
    `figdir (str)`: Figure dicectory where the file will be saved.
    `filename (str)`: Name of the saved nifti in the figdir.
    `target_space (List[int])`: Voxel size of mri volume
    """
    mri_sitk = sitk.GetImageFromArray(mri_vol.squeeze().T)
    mri_sitk.SetSpacing(target_space)
    path = os.path.join(figdir, filename)
    sitk.WriteImage(mri_sitk, path)
    print_(f">-Wrote nifti to path: {path}")


def get_train_val_idxs_from_kfold(all_idxs, fold_num, num_folds, seed):

    print_(f"\n>All to load indexes: {all_idxs}\n\t with length {len(all_idxs)}")
    
    kfold = KFold(num_folds, shuffle=True, random_state=seed)
    k_train_idxs, k_valid_idxs = list(kfold.split(all_idxs))[fold_num]
    
    # Sklearn Kfold return indexes of the index set. So all indexes should be indexed by the returned indexes.
    train_idxs = list(np.asarray(all_idxs)[k_train_idxs])
    valid_idxs = list(np.asarray(all_idxs)[k_valid_idxs])
    
    print_(f"Dataset division:")
    print_("\t- Train indexes:", train_idxs, f"\n\twith length: {len(train_idxs)}")
    print_("\t- Valid indexes:", valid_idxs, f"\n\twith length: {len(valid_idxs)}")
    return train_idxs, valid_idxs


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
    # query = f"SELECT [0018|1310] FROM {tablename} WHERE ([0008|103e] like '%tra%' or [path] like '%tra%') and ([0008|103e] like '%t2%' or [0008|103e] like '%T2%') and [0010|0020] like '%{patient_id}%';"
    # query = f"SELECT [0018|1310] FROM {tablename} WHERE ([0018|1030] like '%tra%' or [path] like '%tra%') and ([0008|103e] like '%t2%' or [0008|103e] like '%T2%') and [0010|0020] like '%{patient_id}%';"
    query = f"SELECT [0018|1310] FROM {tablename} WHERE (([0018|1030] like '%tra%' or [path] like '%tra%') or ([0008|103e] like '%t2_tse_traobl%')) and ([0008|103e] like '%t2%' or [0008|103e] like '%T2%') and [0010|0020] like '%{patient_id}%' and (([0018|1030] is NULL) or [0018|1030] not like '%snel_bij bewogen%');"
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


############################### Callbacks ######################################


#Tensorflow Callback for exporting nifti predictions after each epoch.
class IntermediateImagesRecon(Callback):

    def __init__(self,
        validation_set,
        prefix: str,
        train_outdir: str,
        input_sequences: List[str],
        output_sequences: List[str],
        num_images=10
    ):

        self.prefix = prefix
        self.num_images = num_images
        self.validation_set = (
            validation_set[0][:num_images, ...],        # input
            validation_set[1][:num_images, ...]         # label
            )
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences

        self.out_dir = f"{train_outdir}"

        if self.num_images > validation_set[0].shape[0]:
            self.num_images = validation_set[0].shape[0]

        print_(f"> IntermediateImages: Exporting images and targets to {self.out_dir}")
        for i in range(self.num_images):

            # Write the input to file
            for seq_idx, seq in enumerate(self.input_sequences):
                img_s = sitk.GetImageFromArray(self.validation_set[0][i, ..., seq_idx].T)
                img_s.SetSpacing([0.5, 0.5, 3.0])
                sitk.WriteImage(img_s, f"{self.out_dir}/{prefix}{i:03d}_{seq}_inp.nii.gz")

            # Write the label to file
            for seq_idx, seq in enumerate(self.output_sequences):
                seg_s = sitk.GetImageFromArray(self.validation_set[1][i, ..., seq_idx].T)
                seg_s.SetSpacing([0.5, 0.5, 3.0])
                sitk.WriteImage(seg_s, f"{self.out_dir}/{prefix}{i:03d}_{seq}_lab.nii.gz")


    def on_epoch_end(self, epoch, logs={}):
        print(f"\nWriting predictions to {self.out_dir}")
        
        # Predict on the validation_set
        self.predictions = self.model.predict(self.validation_set, batch_size=1)
        
        # Error maps
        error_maps = self.predictions - self.validation_set[1]

        # Do predictions for each output sequence.
        for i in range(self.num_images):
            for seq in self.input_sequences:
                pred_s = sitk.GetImageFromArray(self.predictions[i, ..., 0].T)
                pred_s.SetSpacing([0.5, 0.5, 3.0])
                sitk.WriteImage(pred_s, f"{self.out_dir}/{self.prefix}{i:03d}_{seq}_pred.nii.gz")
            
            # Calculate the error map for each sequence.
            for seq in self.input_sequences:
                error_map_s = sitk.GetImageFromArray(error_maps[i, ..., 0].T)
                error_map_s.SetSpacing([0.5, 0.5, 3.0])
                sitk.WriteImage(error_map_s, f"{self.out_dir}/{self.prefix}{i:03d}_{seq}_error_map.nii.gz")


################################### Models #####################################


def squeeze_excite_block(input, ratio=8):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, 1, filters)

    se = GlobalAveragePooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((4, 1, 2, 3))(se)

    x = multiply([init, se])
    return x


def build_dual_attention_unet(
    input_shape,
    num_classes,
    l2_regularization = 0.0001,
    instance_norm = False,
    last_activation: str = 'sigmoid',
    ):

    def conv_layer(x, kernel_size, out_filters, strides=(1,1,1)):
        x = Conv3D(out_filters, kernel_size, 
                strides             = strides,
                padding             = 'same',
                kernel_regularizer  = l2(l2_regularization), 
                kernel_initializer  = 'he_normal',
                use_bias            = False
                )(x)
        return x
    
    in_defaults = {
        "axis": -1,
        "center": True, 
        "scale": True,
        "beta_initializer": "random_uniform",
        "gamma_initializer": "random_uniform"
    }

    def conv_block(input, out_filters, strides=(1,1,1), with_residual=False, with_se=False, activation='relu'):
        # Strided convolution to convsample
        x = conv_layer(input, (3,3,3), out_filters, strides)
        x = Activation('relu')(x)

        # Unstrided convolution
        x = conv_layer(x, (3,3,3), out_filters)

        # Add a squeeze-excite block
        if with_se:
            se = squeeze_excite_block(x)
            x = add([x, se])
            
        # Add a residual connection using a 1x1x1 convolution with strides
        if with_residual:
            residual = conv_layer(input, (1,1,1), out_filters, strides)
            x = add([x, residual])
            
        if instance_norm:
            x = InstanceNormalization(**in_defaults)(x)
            
        if activation == 'leaky':
            x = LeakyReLU(alpha=.1)(x)
        else:
            x = Activation('relu')(x)
        
        # Activate whatever comes out of this
        return x

    # If we already have only one input, no need to combine anything
    inputs = Input(input_shape)

    # Downsampling
    conv1 = conv_block(inputs, 16)
    conv2 = conv_block(conv1, 32, strides=(2,2,2), with_residual=True, with_se=True) #72x72x18
    conv3 = conv_block(conv2, 64, strides=(2,2,2), with_residual=True, with_se=True) #36x36x18
    conv4 = conv_block(conv3, 128, strides=(2,2,2), with_residual=True, with_se=True) #18x18x9
    conv5 = conv_block(conv4, 256, strides=(2,2,2), with_residual=True, with_se=True) #9x9x9
    
    # First upsampling sequence
    up1_1 = Conv3DTranspose(128, (3,3,3), strides=(2,2,2), padding='same')(conv5) #18x18x9
    up1_2 = Conv3DTranspose(128, (3,3,3), strides=(2,2,2), padding='same')(up1_1) #36x36x18
    up1_3 = Conv3DTranspose(128, (3,3,3), strides=(2,2,2), padding='same')(up1_2) #72x72x18
    bridge1 = concatenate([conv4, up1_1]) #18x18x9 (128+128=256)
    dec_conv_1 = conv_block(bridge1, 128, with_residual=True, with_se=True, activation='leaky') #18x18x9

    # Second upsampling sequence
    up2_1 = Conv3DTranspose(64, (3,3,3), strides=(2,2,2), padding='same')(dec_conv_1) # 36x36x18
    up2_2 = Conv3DTranspose(64, (3,3,3), strides=(2,2,2), padding='same')(up2_1) # 72x72x18
    bridge2 = concatenate([conv3, up1_2, up2_1]) # 36x36x18 (64+128+64=256)
    dec_conv_2 = conv_block(bridge2, 64, with_residual=True, with_se=True, activation='leaky')
    
    # Final upsampling sequence
    up3_1 = Conv3DTranspose(32, (3,3,3), strides=(2,2,2), padding='same')(dec_conv_2) # 72x72x18
    bridge3 = concatenate([conv2, up1_3, up2_2, up3_1]) # 72x72x18 (32+128+64+32=256)
    dec_conv_3 = conv_block(bridge3, 32, with_residual=True, with_se=True, activation='leaky')
    
    # Last upsampling to make heatmap
    up4_1 = Conv3DTranspose(16, (3,3,3), strides=(2,2,2), padding='same')(dec_conv_3) # 72x72x18
    dec_conv_4 = conv_block(up4_1, 16, with_residual=False, with_se=True, activation='leaky') #144x144x18 (16)

    # Reduce to a single output channel with a 1x1x1 convolution
    single_channel = Conv3D(num_classes, (1, 1, 1))(dec_conv_4)  

    # Apply sigmoid activation to get binary prediction per voxel
    act  = Activation(last_activation)(single_channel)

    # Model definition
    model = Model(inputs=inputs, outputs=act)
    return model


def build_unet(
    window_size,
    num_classes, 
    l2_regularization = 0.0001, 
    instance_norm = False,
    last_activation: str = 'sigmoid',
):
    # Default parameters for conv layers
    c_defaults = {
        "kernel_size" : (3,3,3),
        "kernel_initializer" : 'he_normal',
        "padding" : 'same'
    }
    in_defaults = {
        "axis": -1,
        "center": True, 
        "scale": True,
        "beta_initializer": "random_uniform",
        "gamma_initializer": "random_uniform"
    }

    # Create NAMED input layers for each sequence
    ct_input  = Input(window_size)

    # Contraction path
    # he_normal defines initial weights - it is a truncated normal distribution (Gaussian dist.)
    # sets padding to same, meaning that input dimensions are the same as output dimensions
    c1 = Conv3D(16, kernel_regularizer = l2(l2_regularization), **c_defaults)(ct_input)
    c1 = Conv3D(16, kernel_regularizer = l2(l2_regularization), **c_defaults)(c1)
    if instance_norm:
        c1 = InstanceNormalization(**in_defaults)(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(32, kernel_regularizer = l2(l2_regularization), **c_defaults)(p1)
    c2 = Conv3D(32, kernel_regularizer = l2(l2_regularization), **c_defaults)(c2)
    if instance_norm:
        c2 = InstanceNormalization(**in_defaults)(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(64, kernel_regularizer = l2(l2_regularization), **c_defaults)(p2)
    c3 = Conv3D(64, kernel_regularizer = l2(l2_regularization), **c_defaults)(c3)
    if instance_norm:
        c3 = InstanceNormalization(**in_defaults)(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(128, kernel_regularizer = l2(l2_regularization), **c_defaults)(p3)
    c4 = Conv3D(128, kernel_regularizer = l2(l2_regularization), **c_defaults)(c4)
    if instance_norm:
        c4 = InstanceNormalization(**in_defaults)(c4)
    c4 = Activation('relu')(c4)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    c5 = Conv3D(256, kernel_regularizer = l2(l2_regularization), **c_defaults)(p4)
    c5 = Conv3D(256, kernel_regularizer = l2(l2_regularization), **c_defaults)(c5)
    if instance_norm:
        c5 = InstanceNormalization(**in_defaults)(c5)
    c5 = Activation('relu')(c5)

    # Upwards U part
    u6 = UpSampling3D((2, 2, 2))(c5)
    u6 = concatenate([u6, c4], axis=-1)
    c6 = Conv3D(128, kernel_regularizer = l2(l2_regularization), **c_defaults)(u6)
    c6 = Conv3D(128, kernel_regularizer = l2(l2_regularization), **c_defaults)(c6)
    if instance_norm:
        c6 = InstanceNormalization(**in_defaults)(c6)
    c6 = Activation('relu')(c6)

    u7 = UpSampling3D((2, 2, 2))(c6)
    u7 = concatenate([u7, c3], axis=-1)
    c7 = Conv3D(64, kernel_regularizer = l2(l2_regularization), **c_defaults)(u7)
    c7 = Conv3D(64, kernel_regularizer = l2(l2_regularization), **c_defaults)(c7)
    if instance_norm:
        c7 = InstanceNormalization(**in_defaults)(c7)
    c7 = Activation('relu')(c7)

    u8 = UpSampling3D((2, 2, 2))(c7)
    u8 = concatenate([u8, c2], axis=-1)
    c8 = Conv3D(32, kernel_regularizer = l2(l2_regularization), **c_defaults)(u8)
    c8 = Conv3D(32, kernel_regularizer = l2(l2_regularization), **c_defaults)(c8)
    if instance_norm:
        c8 = InstanceNormalization(**in_defaults)(c8)
    c8 = Activation('relu')(c8)

    u9 = UpSampling3D((2, 2, 2))(c8)
    u9 = concatenate([u9, c1], axis=-1)
    c9 = Conv3D(16, kernel_regularizer = l2(l2_regularization), **c_defaults)(u9)
    c9 = Conv3D(16, kernel_regularizer = l2(l2_regularization), **c_defaults)(c9)
    if instance_norm:
        c9 = InstanceNormalization(**in_defaults)(c9)
    c9 = Activation('relu')(c9)

    # Perform 1x1x1 convolution and reduce the feature maps to N channels.
    output_layer = Conv3D(num_classes, (1, 1, 1), 
        padding='same', 
        activation=last_activation
        )(c9)

    unet = Model(
        inputs=ct_input,
        outputs=output_layer
        )

    return unet


def get_poisson_ac_matrix_mask(
    mri_dims: tuple,
    nifti_path: str,
    sampling: float,
    seed: int,
    centre_sampling: float,
    # cur,                    # sqlite3 cursor object.
    tolerance = 0.02,
    n_neighbours: int = 42,
    tablename: str = 'dicom_headers_v2',
    dicom_db_path: str = r'sqliteDB/dicoms.db',
    verbatim: bool = False
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
    if verbatim:
        print_(f"\t>Found acquistion matrix: {acquistion_matrices}")

    for ac_shape in acquistion_matrices:
        
        # Add z-dim to the mask (is expected by the poisson mask 3D function)
        ac_shape = (ac_shape[0], ac_shape[1], 1)

        # Try to fit the first acquistion matrix in the MRI image, otherwise continue with the next one and see if that one fits.
        try:
            # Determine where the acquisition matrix should be located in image space.
            xdiff = abs(mri_dims[0] - ac_shape[0])//2
            ydiff = abs(mri_dims[1] - ac_shape[1])//2
            mask = np.zeros((mri_dims[0], mri_dims[1]))

            # for some reason the odd acquistion matrix dimensions fail for the poisson disk mask. So we decrease the acquistion matrix by 1 value for the odd dimenion
            if ac_shape[0] % 2 == 1:
                ac_shape = (ac_shape[0]-1, ac_shape[1], ac_shape[2])
            if ac_shape[1] % 2 == 1:
                ac_shape = (ac_shape[0], ac_shape[1]-1, ac_shape[2])

            # Obtain the k-space mask from .npy files or create them (this is a slow function) It is a advised to pre-create them.
            cropped_mask = get_poisson_mask_3d(
                shape        = ac_shape,
                accel        = (1/sampling),
                n_neighbours = n_neighbours,
                seed         = seed,
                centre_samp  = centre_sampling,
                tol          = tolerance,
            )

            # Do some printing for debugging
            if verbatim:
                write_np2nifti(cropped_mask, os.path.join('temp', 'cropped_mask.nii.gz'))
                print_(f"\t>expected sampling: {sampling}")
                print_(f"\t>actual   sampling: {np.sum(cropped_mask)/(cropped_mask.shape[0]*cropped_mask.shape[1]*cropped_mask.shape[2])}")
            
            if verbatim:
                print_(f"\tacquisition matrix = {ac_shape}")
                print_(f"\timg dims = {mri_dims[0]}, {mri_dims[1]}")
                print_(f"\txdiff = {xdiff}")
                print_(f"\tydiff = {ydiff}")
                print_(f"\tcropped_mask.shape[0] + xdiff = {cropped_mask.shape[0] + xdiff}")
                print_(f"\tcropped_mask.shape[1] + ydiff = {cropped_mask.shape[1] + ydiff}")
            
            # Put the mask in the centre of mri dimensions.
            mask[xdiff:cropped_mask.shape[0] + xdiff, ydiff:cropped_mask.shape[1] + ydiff] = cropped_mask.squeeze()

            if verbatim:
                write_np2nifti(cropped_mask, os.path.join('temp', 'AFTER1.nii.gz'))

            return mask
        
        except:
            print_(f"ERROR - The acquisition matrix does not fit in img space. ERROR. {ac_shape}")
            continue

    return mask


def write_mask_to_file(mask):
    mask = mask.squeeze()
    if mask.ndim == 3:
        mask = mask[0, :, :]
        
    fig = plt.figure()
    plt.imshow(mask.T, cmap="gray")
    plt.title(f'20% central sampling mask', size=12)
    plt.axis('on')
    path = os.path.join('temp', f"20perc_mask_dims_{mask.shape[0]}_{mask.shape[1]}_{get_random_str(6)}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f">-Wrote slice to file: {path}")
    plt.close()


################################### Preprocess #################################


def load_and_preprocess_t2(
    nifti_path: str,
    sampling: float,
    centre_sampling: float,
    target_shape: List[int],
    target_space: List[float],
    norm: str,
    seed: int,
    dicom_db_path: str,
    verbatim: bool = False,
):

    print_(f"\n>Processing T2 mri xy pair: {nifti_path}")
    
    x_true_s = sitk.ReadImage(nifti_path, sitk.sitkFloat32)
    y_true_s = copy.deepcopy(x_true_s)

    if sampling != 1.0:

        if True:
            print(f">>Image dimenions: x_true_s.GetSize()[0]: {x_true_s.GetSize()[0]} and x_true_s.GetSize()[1]: {x_true_s.GetSize()[1]}")

        mask = get_rand_exp_decay_mask_ac_matrix(
            width           = x_true_s.GetSize()[0],
            height          = x_true_s.GetSize()[1],
            sampling        = sampling,
            centre_sampling = centre_sampling,
            seed            = seed,
            exp_scale       = 0.4,      # determined emperically
            nifti_path      = nifti_path,
            dicom_db_path   = dicom_db_path,
            tablename       = 'dicom_headers_v2',
            verbatim        = False,
        )

        if False:
            write_mask_to_file(mask)

        # mask = get_poisson_ac_matrix_mask(
        #     mri_dims        = (x_true_s.GetSize()[0], x_true_s.GetSize()[1]),
        #     nifti_path      = nifti_path,
        #     sampling        = sampling,
        #     seed            = seed,
        #     centre_sampling = centre_sampling,
        #     dicom_db_path   = dicom_db_path,
        #     verbatim        = verbatim,
        # )

        # Perform undersampling in k-space with the given mask.
        x_true_s = im.undersample_kspace(img_s=x_true_s, mask=mask, verbatim=False)

    # Each image can have a different amount of slices, either the target_shape
    # z-dim will be taken or the actual actual amount of slices in the image if 
    # there are less.
    # min_z = min(target_shape[2], x_true_s.GetSize()[2])
    # target_shape = (target_shape[0], target_shape[1], min_z)
    
    x_true_s = im.resample(
        image       = x_true_s, 
        min_shape   = target_shape, 
        method      = sitk.sitkNearestNeighbor, 
        new_spacing = target_space
    )
    y_true_s = im.resample(
        image       = y_true_s, 
        min_shape   = target_shape, 
        method      = sitk.sitkNearestNeighbor, 
        new_spacing = target_space
    )

    x_true_s = im.center_crop(image=x_true_s, shape=target_shape)
    y_true_s = im.center_crop(image=y_true_s, shape=target_shape)
    
    if norm == "rescale_0_1":
        x_true_n = sitk.GetArrayFromImage(x_true_s).T
        x_true_n = (1.0*(x_true_n - np.min(x_true_n))/np.ptp(x_true_n))
        # save_array2d_to_image(x_true_n[:,:,10].squeeze().T, 'slice 10', 'temp', f"test_x_true_n_slice10_{get_random_str(6)}.png")

        y_true_n = sitk.GetArrayFromImage(y_true_s).T
        y_true_n = (1.0*(y_true_n - np.min(y_true_n))/np.ptp(y_true_n))
        # save_array2d_to_image(y_true_n[:,:,10].squeeze().T, 'slice 10', 'temp', f"test_y_true_n_slice10_{get_random_str(6)}.png")

    return x_true_n.astype(np.float32), y_true_n.astype(np.float32)


def load_all_data(
    nifti_paths: List[str],
    sampling: float,
    centre_sampling: float,
    target_shape: List[int],
    target_space: List[float],
    norm: str,
    n_workers: int,
    seed: int,
    dicom_db_path: str = r'sqliteDB/dicoms.db',
):
    print(f"\n>T2s: Start of parallel loading and preprocessing...")
    t = time.time()

    # Start a pool of workers.
    pool = multiprocessing.Pool(processes=n_workers)

    # con = sqlite3.connect(dicom_db_path)
    # cur = con.cursor()

    # Define a partial function that can undergo a pool.map function.
    load_and_preprocess_partial = partial(load_and_preprocess_t2,
        sampling        = sampling,
        centre_sampling = centre_sampling,
        target_shape    = target_shape,
        target_space    = target_space,
        norm            = norm,
        seed            = seed,
        dicom_db_path   = dicom_db_path,
        verbatim        = True if DEBUG else False
    )

    # Apply the load and preprocess function for each file in the given paths
    data = pool.map(load_and_preprocess_partial, nifti_paths, chunksize=20)
    pool.close()

    # Unzip the loaded data
    x_true, y_true = zip(*data)
    x_true = list(x_true)
    y_true = list(y_true)

    for i in range(len(x_true)):
        x_true[i] = np.expand_dims(x_true[i], 4)
        y_true[i] = np.expand_dims(y_true[i], 4)
        
    print(f"Time parallel loading T2s time: {time.time() - t} seconds")
    print_stats_np_list(x_true, arr_name="x_true")
    print_stats_np_list(y_true, arr_name="y_true")

    return x_true, y_true


def get_window_generator(
    x_true: List[np.ndarray],
    y_true: List[np.ndarray],
    shape: Tuple,
    input_sequences: List[str],
    output_sequences: List[str],
    seed: int,
    batch_size: Optional[int] = 5, 
    indexes: Optional[List[int]] = None, 
    shuffle: bool = False, 
    do_augmentation = True,
    rotation_freq: float = 0.1,    # Ask chris which parameters he used.
    tilt_freq: float = 0.1,
    noise_freq: float = 0.3,
    noise_mult: float = 1e-3,
    mirror_freq: float = 0.5,
) -> Iterator[Tuple[dict, dict]]:
    """
    Returns a (training) generator for use with model.fit().
    augmentation: Apply augmentation or not (bool).
    """

    n_obs = len(x_true)

    if indexes == None or DEBUG:
        indexes = list(range(n_obs))

    if type(indexes) == int:
        indexes = list(range(indexes))

    if batch_size == None:
        batch_size = len(indexes)

    ep_idx = 0      # within epoch idx
    batch_idx = 0   # keep a counter of all generated batches

    rng = np.random.default_rng(seed)

    # Prepare empty batch placeholder with named inputs and outputs
    input_batch = np.zeros((batch_size,) + shape + (len(input_sequences),), dtype=np.float32)   # T2, adc, dwi
    output_batch = np.zeros((batch_size,) + shape + (len(output_sequences),), dtype=np.float32) # seg

    # Loop infinitely to keep generating batches
    while True:
        
        # Prepare each observation in a batch
        for img_idx in range(batch_size):
            
            # Shuffle the order of images if all indexes have been seen
            if ep_idx == 0 and shuffle:
                rng.shuffle(indexes)

            current_index = indexes[ep_idx]

            # Insert the augmented images into the input batch
            img_crop, lab_crop = (
                np.zeros(shape + (len(input_sequences),)),
                np.zeros(shape + (len(output_sequences),))
            )

            for seq_idx, seq in enumerate(input_sequences):
                img_crop[:, :, :, seq_idx] = x_true[current_index][:, :, :, seq_idx]

            for seq_idx, seq in enumerate(output_sequences):
                lab_crop[:, :, :, seq_idx] = y_true[current_index][:, :, :, seq_idx]

            if do_augmentation:
                img_crop, lab_crop = augment(
                    img            = img_crop,
                    seg            = lab_crop,
                    noise_chance   = noise_freq,
                    noise_mult_max = noise_mult,
                    rotate_chance  = rotation_freq,
                    tilt_chance    = tilt_freq,
                    mirror_chance  = mirror_freq,
                    seed           = seed + batch_idx + img_idx,
                )

            input_batch[img_idx]  = img_crop
            output_batch[img_idx] = lab_crop

            # Increase the current index and modulo by the number of rows
            # so that we stay within bounds
            ep_idx = (ep_idx + 1) % len(indexes)

        batch_idx += 1
        yield np.clip(input_batch, 0, 1), np.clip(output_batch, 0, 1)


################################################################################


def train(
    train_path_list: str,
    seed: int = None,
    fold_num: int = 0,
    num_folds: int = 5,
    idxs_path: str = "data/path_lists/pirads_4plus/train_val_test_idxs.yml",
    train_set_key: str = "train_set0",
    centre_sampling: float = 0.5,
    target_shape: Tuple = (192, 192, 20),
    target_space: List = [0.5, 0.5, None],
    normalization: str = "rescale_0_1",
    acceleration: float = 4.0,
    n_workers: int = 10,
    # window_size: List = (160, 160, 48),
    batch_size: int = 13,
    inp_seqs: List = ['t2'],
    out_seqs: List = ['t2_true'],
    rotation_freq: float = 0.25,
    tilt_freq: float = 0.040,
    noise_freq: float = 0.60,
    noise_mult: float = 0.0030,
    mirror_freq: float = 0.5,
    loss: str = "ssim",
    optimizer: str = "adam",
    learning_rate: float = 0.0004,
    unet_type: str = "simple",
    l2_regularization: float = 0.0001,
    do_instance_norm: bool = True,
    last_activation: str = 'sigmoid',
    do_early_stop: bool = True,
    early_stop_pat: int = 50,
    early_stop_var: str = "val_loss",
    early_stop_mode: str = "min",
    max_epochs: int = 2000,
    validation_interval: int = 1,
    verbose: int = 2,
    dicom_db_path = r'sqliteDB/dicoms.db',
    **kwargs,
):
    print_("> Fold number:", fold_num, "of", num_folds)

    if DEBUG:
        train_path_list = train_path_list[:-4] + "_debug" + ".txt"

    # Read path to the data
    t2_files = [l.strip() for l in open(train_path_list)]
    all_idxs = read_yaml_to_dict(idxs_path)[train_set_key]

    if DEBUG:
        # percent_load        = 0.15
        all_idxs            = all_idxs[:len(t2_files)]
        validation_interval = 2
        n_workers           = 1
        batch_size          = 2
        verbose             = 1
        max_epochs          = 100
        set_gpu(gpu_idx=0) if not DEBUG else set_gpu(gpu_idx=1)
        
    # Determine train and validation indexes
    train_idxs, valid_idxs = get_train_val_idxs_from_kfold(all_idxs, fold_num, num_folds, seed)

    t2_files.reverse()

    # Read all relevant data into RAM in parallel
    x_true, y_true = load_all_data(
        nifti_paths     = t2_files if not DEBUG else t2_files[:len(train_idxs)],
        sampling        = (1/acceleration),
        centre_sampling = centre_sampling,
        target_shape    = target_shape,
        target_space    = target_space,
        norm            = normalization,
        n_workers       = n_workers,
        seed            = seed,
        dicom_db_path   = dicom_db_path,
    )

    # Create window based train and validation generator
    train_generator = get_window_generator(
        x_true            = x_true,
        y_true            = y_true,
        shape             = target_shape,
        input_sequences   = inp_seqs,
        output_sequences  = out_seqs,
        seed              = seed,
        batch_size        = batch_size,
        indexes           = train_idxs,
        shuffle           = True if not DEBUG else False,
        do_augmentation   = True,
        rotation_freq     = rotation_freq,
        tilt_freq         = tilt_freq,
        noise_freq        = noise_freq,
        noise_mult        = noise_mult,
        mirror_freq       = mirror_freq,
    )
    valid_generator = get_window_generator(
        x_true            = x_true,
        y_true            = y_true,
        shape             = target_shape,
        input_sequences   = inp_seqs,
        output_sequences  = out_seqs,
        seed              = seed + 1_000_000,  # Different seed for the validation set
        batch_size        = len(valid_idxs),
        indexes           = valid_idxs,
        shuffle           = False,
        do_augmentation   = False,
        rotation_freq     = None,
        tilt_freq         = None,
        noise_freq        = None,
        noise_mult        = None,
        mirror_freq       = None,
    )
    train_set = next(train_generator)
    valid_set = next(valid_generator)
    print_stats_np(train_set[0], "X_train")
    print_stats_np(train_set[1], "Y_train")
    print_stats_np(valid_set[0], "X_valid")
    print_stats_np(valid_set[1], "Y_valid")

    # Write some samples of the train and valid batch data to dir: samples/
    if True:
        for i in range(min(batch_size, len(valid_idxs))):
            write_array2nifti_t2(train_set[0][i], SAMPLE_DIR, f"{i}_xtrue_batch_train.nii.gz")
            write_array2nifti_t2(train_set[1][i], SAMPLE_DIR, f"{i}_ytrue_batch_train.nii.gz")
            write_array2nifti_t2(valid_set[0][i], SAMPLE_DIR, f"{i}_xtrue_batch_valid.nii.gz")
            write_array2nifti_t2(valid_set[1][i], SAMPLE_DIR, f"{i}_ytrue_batch_valid.nii.gz")


    # Create the model and show summary
    if unet_type == "simple":
        dnn = build_unet(
            window_size       = target_shape + (1,), 
            num_classes       = 1,
            l2_regularization = l2_regularization,
            instance_norm     = do_instance_norm,
            last_activation   = last_activation,
        )
    if unet_type == "dual_attention":
        dnn = build_dual_attention_unet(
            input_shape       = target_shape + (1,),
            num_classes       = 1,
            l2_regularization = l2_regularization,
            instance_norm     = do_instance_norm,
            last_activation   = last_activation,
        )
    
    dnn.summary(line_length=160)

    # Start construction the model reconstruction model
    if optimizer == "adam":
        optimizer = Adam(learning_rate)

    if loss == "ssim":
        loss = ssim_loss
    if loss == "msssim":
        loss = get_msssim_loss

    metrics = [ssim_metric, psnr_metric, 'mse']

    dnn.compile(
        optimizer = optimizer,
        loss      = loss,
        metrics   = metrics,
    )

    callbacks = []

    if do_early_stop:
        # Stop training after X epochs without improvement
        callbacks += [
            EarlyStopping(
                patience = early_stop_pat,
                monitor  = early_stop_var,
                mode     = early_stop_mode,
                verbose  = 1
            )
        ]

    # This callback predicts a number of imgs for test set after an epoch
    # and shows a few slices in PNG files in the "output/" folder
    callbacks += [
        IntermediateImagesRecon(
            validation_set   = valid_set, 
            prefix           = f"",
            train_outdir     = OUTPUT_DIR,
            num_images       = len(valid_idxs) if not DEBUG else 20,
            input_sequences  = inp_seqs,
            output_sequences = out_seqs,
        )
    ]

    callbacks += [CSVLogger(os.path.join(LOG_DIR, f"train_fold{fold_num}.csv"))]

    # Add a modelcheckpoint callback for each metric and the loss
    for metric_name in [m.__name__ for m in metrics[:2]] + ["loss"]:
        callbacks += [
            ModelCheckpoint(
                filepath        = os.path.join(MODEL_DIR, f"best_{metric_name}_fold{fold_num}.h5"),
                monitor         = f"val_{metric_name}", 
                save_best_only  = True, 
                mode            = 'min' if "loss" in metric_name else "max",
                verbose         = 1
            )
        ]

    dnn.fit(
        x               = train_generator,
        validation_data = valid_set,
        steps_per_epoch = len(train_idxs) // batch_size * validation_interval, 
        epochs          = max_epochs,
        callbacks       = callbacks,
        verbose         = verbose
    )

    print_("[I] Completed Training.")


################################################################################


if __name__ == '__main__':

    set_start_method("spawn")

    args = parse_input_args()
    configs = read_config(args.config_fpath)
    DEBUG = configs['is_debug']
    TRAIN_DIR = os.path.join(configs['train_dir'], f"fold{configs['fold_num']}")

    if DEBUG:
        TRAIN_DIR = os.path.join(TRAIN_DIR, 'debug')

    print_(">Creating folder structure")
    OUTPUT_DIR       = os.path.join(TRAIN_DIR, "output/")
    SAMPLE_DIR       = os.path.join(TRAIN_DIR, "samples/")
    LOG_DIR          = os.path.join(TRAIN_DIR, "logs/")
    MODEL_DIR        = os.path.join(TRAIN_DIR, "models/")
    UNDERSAMPLE_DIR  = os.path.join(TRAIN_DIR, "undersamples/")
    TEMP_DIR         = os.path.join(TRAIN_DIR, "temp/")
    FIGS_DIR         = os.path.join(TRAIN_DIR, "figs/")
    VALIDATION_DIR   = os.path.join(TRAIN_DIR, "val_set/")
    TEST_DIR         = os.path.join(TRAIN_DIR, "test_set/")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(UNDERSAMPLE_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # copy the config that started the program to the experiment folder.
    dst = os.path.join(TRAIN_DIR, args.config_fpath.split('/')[-1])
    copyfile(args.config_fpath, dst)

    # for on peregrine - To ensure we do not go OOM (RAM)
    disable_eager_execution()
    
    train(**configs)
