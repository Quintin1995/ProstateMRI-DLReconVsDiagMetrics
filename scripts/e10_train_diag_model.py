import argparse
import numpy as np
import time
import multiprocessing
import SimpleITK as sitk
import os

from datetime import date
from focal_loss import BinaryFocalLoss
from sklearn.model_selection import KFold
from typing import List
from functools import partial

from tensorflow.keras.models import load_model
from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, Activation, Dense, concatenate, add
from tensorflow.keras.layers import Conv3DTranspose, LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras.layers import GlobalAveragePooling3D, Reshape, Dense, multiply, Permute
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from fastMRI_PCa.models import ssim_loss, ssim_metric, psnr_metric
from fastMRI_PCa.data import resample, center_crop, normalize, binarize_s
from fastMRI_PCa.visualization import save_slice_3d
from fastMRI_PCa.utils import print_p, print_stats_np, dump_dict_to_yaml
from fastMRI_PCa.utils import list_from_file, get_rand_exp_decay_mask, create_dirs_if_not_exists
from fastMRI_PCa.data import undersample, get_generator, IntermediateImagesDiag
from fastMRI_PCa.utils import read_yaml_to_dict

from umcglib.callbacks import FROCCallback


################################  README  ######################################
# OLD -This script will load the following data: DWI images, ADC maps and
# undersampled T2w images. The data will be used to train a diagnostic network
# where lesions are segmented for the patient. Model = Dual Attention U-net from
# Anindo.
# Goal: Train a diagnostic U-Net.


################################### FUNCTIONS ##################################


def parse_input_args():

    help = """This script will load the following data: DWI images, ADC maps and
              undersampled T2w images. The data will be used to train a
              diagnostic network where lesions are segmented for the patient.
              Goal: Train a diagnostic U-Net."""

    parser = argparse.ArgumentParser(description=help)

    parser.add_argument('-fn',
                        '--fold_num',
                        type=int,
                        help='The current fold number to be run. Not the be confused with the total number of folds')

    parser.add_argument('-nf',
                        '--num_folds',
                        type=int,
                        help='The number of folds in total.')

    parser.add_argument('-to',
                        '--train_outdir',
                        type=str,
                        required=True,
                        help='Directory where training data will be saved. And a dir in Models will be made where the best model is saved.')

    parser.add_argument('--job',
                        dest='is_job',
                        action='store_true',
                        help="Whether the current code is run via a job script. Use --job for when it is not job.sh script ")

    parser.add_argument('--no_job',
                        dest='is_job',
                        action='store_false',
                        help="Whether the current code is run via a job script. Use --no_job for when it is not job.sh script")

    parser.add_argument('-rd',
                        '--t2_recon_dir',
                        type=str,
                        help='OPTIONAL Directory where weights for T2 reconstruction model is stored. Will look in the models/ folder.')
    
    parser.add_argument('-a',
                        '--acceleration',
                        type=float,
                        help='Acceleration of the simulated acquisition. R = 1/sampling. (float). ONLY FILL IN WHEN NOT CHOOSING A DEEP LEARNING RECONSTRUCTION MODEL.' )

    parser.add_argument('-cs',
                        '--centre_sampling',
                        type=float,
                        default=0.5,
                        help='Percentage of SAMPLED k-space that lies central in k-space. (float). ONLY FILL IN WHEN NOT CHOOSING A DEEP LEARNING RECONSTRUCTION MODEL.' )
    
    parser.add_argument('-tsh',
                        '--target_shape',
                        nargs='+',
                        type=int,
                        default=[192, 192, 20],
                        help='Centre crop [x,y,z] that will be used for training. ONLY FILL IN WHEN NOT CHOOSING A DEEP LEARNING RECONSTRUCTION MODEL.')

    parser.add_argument('-tsp',
                        '--target_space',
                        nargs='+',
                        type=float,
                        default=[0.5, 0.5, 3.0],
                        help='Resampling spacing. ONLY FILL IN WHEN NOT CHOOSING A DEEP LEARNING RECONSTRUCTION MODEL.')

    parser.add_argument('--norm',
                        type=str,
                        default="znorm",
                        help='Normalization method used for the diagnostic model.')

    args = parser.parse_args()
    return args


def load_and_preprocess_nifti(
    nifti_path: str,
    seq: str,
    target_shape: List[int],
    target_space: List[float],
    norm: str):

    print_p(f"Processing {seq} mri: {nifti_path}")
    
    # Read the DWI transversal file as SimpleITK object
    mri_tra_s = sitk.ReadImage(nifti_path, sitk.sitkFloat32)

    # Resample, center crop and z-normalization all on sitk objects.
    mri_tra_s = resample(mri_tra_s, 
                        min_shape=target_shape, 
                        method=sitk.sitkNearestNeighbor, 
                        new_spacing=target_space)
    mri_tra_s = center_crop(mri_tra_s, target_shape)
    if seq != 'seg':
        mri_tra_s = normalize(norm_str=norm, img_s=mri_tra_s)
    else:
        mri_tra_s = binarize_s(mri_tra_s)

    return sitk.GetArrayFromImage(mri_tra_s).T


def parallel_loading_preprocess(
    nifti_paths: List[str],
    seq: str,
    sampling: float,
    centre_sampling: float,
    target_shape: List[int],
    target_space: List[float],
    norm: str,
    do_undersample: bool):

    t = time.time()
    print(f"\n{seq}s: Start of parralel loading and preprocessing...")

    # Start a pool of workers.
    pool = multiprocessing.Pool(processes=N_CPUS)

    # Define a partial function that can undergo a pool.map function.
    if seq == 'T2':
        partial_f = partial(load_and_preprocess_t2,
                            sampling=sampling,
                            centre_sampling = centre_sampling,
                            target_shape=target_shape,
                            target_space=target_space,
                            norm=norm,
                            do_undersample=do_undersample)
    else:
        partial_f = partial(load_and_preprocess_nifti,
                            seq=seq,
                            target_shape=target_shape,
                            target_space=target_space,
                            norm=norm)

    # Apply the load and preprocess function for each file in the given paths
    data_list = pool.map(partial_f, nifti_paths)
    pool.close()
    
    # Aggregate the data in the first axis.
    data = np.stack(data_list, axis=0)

    print(f"Time parallel loading {seq}s time: {time.time() - t} seconds")
    print_stats_np(data, f"{seq} numpy stats\n")

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

    if do_undersample:
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


def load_reconstructed_t2s(
    nifti_paths: List[str],
    seq: str,
    sampling: float,
    centre_sampling: float,
    target_shape: List[int],
    target_space: List[float],
    norm: str,
    t2_recon_dir: str):

    # Load the T2 images (reconstructions) in parallel as numpy array
    recons_ifft = parallel_loading_preprocess(nifti_paths=nifti_paths,
                                              seq=seq,
                                              sampling=sampling,
                                              centre_sampling=centre_sampling,
                                              target_shape=target_shape,
                                              target_space=target_space,
                                              norm=norm,
                                              do_undersample=True)

    # if t2_recon_dir not given as input argument we will simply load the IFFT reconstruction for the diagnostic model.
    if t2_recon_dir == None:
        print("Normal Reconstructions loaded. (not Deep Learning reconstructions but simple IFFT reconstructions.)")
        return np.squeeze(recons_ifft)
    else:
        # Load T2 reconstruction model.
        t2_model_path = f"models/{t2_recon_dir}/best-direct-fold0_val_loss.h5"
        recon_model = load_model(t2_model_path, custom_objects={'ssim_loss': ssim_loss, 'ssim_metric': ssim_metric, 'psnr_metric': psnr_metric})
        recon_model.summary()

        if DEBUG:
            recons_dl = np.squeeze(recon_model.predict(recons_ifft, batch_size=1))
        else:
            recons_dl = np.squeeze(recon_model.predict(recons_ifft, batch_size=10))
    
        print_stats_np(recons_dl, "Recons_dl")
        print("Deep Learning Reconstructions loaded.")
        return recons_dl


def load_data(norm: str,
              sampling: float,
              centre_sampling: float,
              target_shape: List[int],
              target_space: List[float],
              first_idx: int,
              last_idx: int,
              t2_recon_dir: str):
    
    # T2 images: Load - Undersample - preprocess --> Reconstructions.
    t2_paths = list_from_file("data/path_lists/pirads_4plus/current_t2w.txt")
    last_idx = len(t2_paths) if last_idx == -1 else last_idx

    t2s_recons = load_reconstructed_t2s(t2_paths[first_idx:last_idx],
                                        seq='T2',
                                        sampling=sampling,
                                        centre_sampling=centre_sampling,
                                        target_shape=target_shape,
                                        target_space=target_space,
                                        norm=norm,
                                        t2_recon_dir=t2_recon_dir)
    
    # DWI images: Load - preprocess.
    dwi_paths = list_from_file("data/path_lists/pirads_4plus/current_dwi.txt")
    dwis = parallel_loading_preprocess(dwi_paths[first_idx:last_idx],
                                       seq='dwi',
                                       sampling=None,
                                       centre_sampling=None,
                                       target_shape=target_shape,
                                       target_space=target_space,
                                       norm=norm,
                                       do_undersample=None)

    # ADC images: Load - preprocess.
    adc_paths = list_from_file("data/path_lists/pirads_4plus/current_adc.txt")
    adcs = parallel_loading_preprocess(adc_paths[first_idx:last_idx],
                                       seq='ADC',
                                       sampling=None,
                                       centre_sampling=None,
                                       target_shape=target_shape,
                                       target_space=target_space,
                                       norm=norm,
                                       do_undersample=None)

    # LABEL Segmentation images: Load - preprocess.
    seg_paths = list_from_file("data/path_lists/pirads_4plus/seg_new.txt")
    segs = parallel_loading_preprocess(seg_paths[first_idx:last_idx],
                                       seq='seg',
                                       sampling=None,
                                       centre_sampling=None,
                                       target_shape=target_shape,
                                       target_space=target_space,
                                       norm=norm,
                                       do_undersample=None)

    # Visualize a set of input modalities and the corresponding label.
    if True:
        slice_ = 6
        save_slice_3d(t2s_recons[3], slice_, 'temp', f"DEBUG_{slice_}_t2_a{1/sampling}")
        save_slice_3d(dwis[3], slice_, 'temp', f"DEBUG_{slice_}_dwi_a{1/sampling}")
        save_slice_3d(adcs[3], slice_, 'temp', f"DEBUG_{slice_}_adc_a{1/sampling}")
        save_slice_3d(segs[3], slice_, 'temp', f"DEBUG_{slice_}_seg_a{1/sampling}")

    X = np.stack([t2s_recons, dwis, adcs], axis = 4)
    print_stats_np(X, "Train array")
    
    # Add output sequence dim. So that len(X.dims) == len(segs.dims)
    segs = np.expand_dims(segs, axis=4)

    return X, segs


################################  MODEL   ######################################


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
    l2_regularization = 0.0001):

    def conv_layer(x, kernel_size, out_filters, strides=(1,1,1)):
        x = Conv3D(out_filters, kernel_size, 
                strides             = strides,
                padding             = 'same',
                kernel_regularizer  = regularizers.l2(l2_regularization), 
                kernel_initializer  = 'he_normal',
                use_bias            = False
                )(x)
        return x

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
    conv2 = conv_block(conv1, 32, strides=(2,2,1), with_residual=True, with_se=True) #72x72x18
    conv3 = conv_block(conv2, 64, strides=(2,2,1), with_residual=True, with_se=True) #36x36x18
    conv4 = conv_block(conv3, 128, strides=(2,2,2), with_residual=True, with_se=True) #18x18x9
    conv5 = conv_block(conv4, 256, strides=(2,2,2), with_residual=True, with_se=True) #9x9x9
    
    # First upsampling sequence
    up1_1 = Conv3DTranspose(128, (3,3,3), strides=(2,2,2), padding='same')(conv5) #18x18x9
    up1_2 = Conv3DTranspose(128, (3,3,3), strides=(2,2,2), padding='same')(up1_1) #36x36x18
    up1_3 = Conv3DTranspose(128, (3,3,3), strides=(2,2,1), padding='same')(up1_2) #72x72x18
    bridge1 = concatenate([conv4, up1_1]) #18x18x9 (128+128=256)
    dec_conv_1 = conv_block(bridge1, 128, with_residual=True, with_se=True, activation='leaky') #18x18x9

    # Second upsampling sequence
    up2_1 = Conv3DTranspose(64, (3,3,3), strides=(2,2,2), padding='same')(dec_conv_1) # 36x36x18
    up2_2 = Conv3DTranspose(64, (3,3,3), strides=(2,2,1), padding='same')(up2_1) # 72x72x18
    bridge2 = concatenate([conv3, up1_2, up2_1]) # 36x36x18 (64+128+64=256)
    dec_conv_2 = conv_block(bridge2, 64, with_residual=True, with_se=True, activation='leaky')
    
    # Final upsampling sequence
    up3_1 = Conv3DTranspose(32, (3,3,3), strides=(2,2,1), padding='same')(dec_conv_2) # 72x72x18
    bridge3 = concatenate([conv2, up1_3, up2_2, up3_1]) # 72x72x18 (32+128+64+32=256)
    dec_conv_3 = conv_block(bridge3, 32, with_residual=True, with_se=True, activation='leaky')
    
    # Last upsampling to make heatmap
    up4_1 = Conv3DTranspose(16, (3,3,3), strides=(2,2,1), padding='same')(dec_conv_3) # 72x72x18
    dec_conv_4 = conv_block(up4_1, 16, with_residual=False, with_se=True, activation='leaky') #144x144x18 (16)

    # Reduce to a single output channel with a 1x1x1 convolution
    single_channel = Conv3D(1, (1, 1, 1))(dec_conv_4)  

    # Apply sigmoid activation to get binary prediction per voxel
    act  = Activation('sigmoid')(single_channel)

    # Model definition
    model = Model(inputs=inputs, outputs=act)
    return model


################################  PARAMS  ######################################

DESCRIPTION = """Trains a dual attention U-net on DWI, ADC andundersampled T2 images for prostate cancer lesion segmentation with label segmentation maps."""

# Path to file with train/val/test indexes
IDXS_PATH = "data/path_lists/pirads_4plus/train_val_test_idxs.yml"

ES_PATIENCE = 50    # Early stopping patience
LOSS = "BinaryFocalLoss"
FOCAL_LOSS_GAMMA = 2.0       # Emperically determined by Chris and Anindo
FOCAL_LOSS_POS_WEIGHT = 0.25 # Emperically determined by Chris and Anindo
LEARNING_RATE = 1e-4

# Number of CPUs for data loading undersampling and preprocessing
N_CPUS = 12
SEED = 3478+10+10
INP_SEQUENCES = ["u_t2", "dwi", "adc"]
OUT_SEQUENCES = ["seg"]
EPOCHS = 1000

LAST_INDEX = -1        # -1 = all indexes, other values should be less than 1535

if __name__ == '__main__':

    # Get all input parameters
    args = parse_input_args()
    print_p(f"\nAll input parameters: {args}\n")
    DEBUG = False if args.is_job else True
    
    if args.t2_recon_dir == None:
        print("---\n\nNO RECONSTRUCTION MODEL SELECTED. T2S ARE NOW SIMPLE/NOMRAL RECONSTRUCTION FROM THE IFFT (NORMAL RECONSTRUCTIONS)\n\n---")

    # See if the code is run on the login node or run as a job script.
    batch_size = 10 if not DEBUG else 1     # if job, then batch_size=10
    verbose = 2 if not DEBUG else 1         # if job, then less text in slurm
    perc_load = 1.0 if not DEBUG else 0.07  # decrease num imgs loaded if not a job
    N_CPUS = 12 if not DEBUG else 2         # decrease num cpus if not a job
    LAST_INDEX = -1 if not DEBUG else 40    # Use less data when it is not a job.

    # Set the GPU to the second GPU if not running the code via a job script.
    print(f"isDEBUG: {DEBUG}")
    if DEBUG:
        print("Setting GPU to GPU 1.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Determine the type of reconstruction parameters. Either from reconstruction model or given by argparse input arguments.
    if args.t2_recon_dir != None:
        recon_params = read_yaml_to_dict(f"train_output/{args.t2_recon_dir}/params.yml")
    else:
        recon_params = {
            "sampling": 1/args.acceleration,
            "acceleration": args.acceleration,
            "centre_sampling": args.centre_sampling,
            "target_shape": args.target_shape,
            "target_space": args.target_space
        }

    # Dump parameters to a file for documentation purposes.
    params = {
    "type_model": "diagnostic_model",
    "description": DESCRIPTION,
    "is_job": args.is_job,
    "train_output_dir": args.train_outdir,
    "models_outdir": args.train_outdir,
    "input_sequences": INP_SEQUENCES,
    "output_sequences": OUT_SEQUENCES,
    "batch_size": batch_size,
    "verbose": verbose,
    # "optimizer": OPTIMIZER,
    "seed": SEED,
    "number_folds": args.num_folds,
    "epochs": EPOCHS,
    # "last_activation": LAST_ACTIVATION,
    "t2_recon_dir": args.t2_recon_dir,
    "loss": LOSS,
    "sampling": (recon_params["sampling"]),
    "acceleration": recon_params["acceleration"],
    "centre_sampling": recon_params["centre_sampling"],
    "target_shape": recon_params["target_shape"],
    "target_space": recon_params["target_space"],
    "norm": args.norm,
    "datetime": date.today().strftime("%Y-%m-%d")}
    dump_dict_to_yaml(params, f"train_output/{args.train_outdir}", filename=f"params")
    
    # Create directories where train results and the best model will be stored.
    create_dirs_if_not_exists([f"models/{args.train_outdir}", f"train_output/{args.train_outdir}"])

    # Load input and label data.
    X, Y = load_data(norm=args.norm,
                     sampling=recon_params["sampling"],
                     centre_sampling=recon_params["centre_sampling"],
                     target_shape=recon_params["target_shape"],
                     target_space=recon_params["target_space"],
                     first_idx=0,
                     last_idx=LAST_INDEX,
                     t2_recon_dir=args.t2_recon_dir)

    # Get the number of observations
    num_obs = X.shape[0]
    print_p(f"\nNumber of observations: {num_obs}")
    print_p(f"\nFold number: {args.fold_num+1} of {args.num_folds}")

    # Randomly split the data into a train (80%) / validation (10%) / test (10%)
    kfold = KFold(args.num_folds, shuffle=True, random_state=SEED)
    
    if not DEBUG:
        idxs_dict = read_yaml_to_dict(IDXS_PATH)
        all_idxs = idxs_dict["train_set0"]
    else:
        all_idxs = list(range(num_obs))
    
    train_idxs, valid_idxs = list(kfold.split(all_idxs))[args.fold_num]
    train_idxs = list(train_idxs)
    valid_idxs = list(valid_idxs)
      
    print_p(f"\nDataset division:\n\t- Train: {len(train_idxs)} = {len(train_idxs)/len(all_idxs)*100}%")
    print_p(f"\t- Valid: {len(valid_idxs)} = {len(valid_idxs)/len(all_idxs)*100}%")
    print_p(f"\t- Test: {0}")
    print_p(f"Validation indexes: {valid_idxs}")

    crop_size = X[0].shape[:3]  # first observation, then first three dimensions
    print(f"crop size = {crop_size}")

    train_generator = get_generator(
        batch_size=batch_size,
        shape=crop_size,
        X=X,    # train data
        Y=Y,    # label data
        input_sequences=INP_SEQUENCES,
        output_sequences=OUT_SEQUENCES,
        indexes=train_idxs,
        shuffle=True)

    train_set = next(train_generator)
    print(f"\ntrain set: {train_set[0].shape}")

    validation_generator = get_generator(
        batch_size=None,
        shape=crop_size,
        X=X,    # train data
        Y=Y,    # label data
        input_sequences=INP_SEQUENCES,
        output_sequences=OUT_SEQUENCES,
        indexes=valid_idxs,
        shuffle=True,
        augmentation=False)

    validation_set = next(validation_generator)
    print(f"Validation set: {validation_set[0].shape}\n")

    # The following to stop exessive RAM consumption
    disable_eager_execution()

    # Load and compile the model.
    model = build_dual_attention_unet(input_shape=X[0].shape)
    if LOSS == "BinaryFocalLoss":
        loss = BinaryFocalLoss(gamma=FOCAL_LOSS_GAMMA, pos_weight=FOCAL_LOSS_POS_WEIGHT)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    metric = MeanIoU(num_classes=2)

    model.compile(
        optimizer=optimizer, 
        loss=loss,
        metrics=[metric])

    model.summary(line_length=200)

    # After every epoch store the model with the best validation performance
    model_cp_cb = ModelCheckpoint(f"models/{args.train_outdir}/best-direct-fold{args.fold_num}.h5",
        monitor = 'val_loss', 
        save_best_only=True, 
        mode='min',
        verbose=verbose)

    # This callback predicts on a number of images for the test set after each epoch
    # and shows a few slices in PNG files in the "train_output/" folder
    imgs_cb = IntermediateImagesDiag(validation_set, 
                                     prefix=f"diag_fold{args.fold_num}",
                                     train_outdir=args.train_outdir,
                                     num_images=10,
                                     input_sequences=INP_SEQUENCES,
                                     output_sequences=OUT_SEQUENCES)

    # This callback produces log file of training and val metrics at each epoch.
    csv_log_cb = CSVLogger(f"train_output/{args.train_outdir}/train_direct_log_fold{args.fold_num}.csv")

    # Early stopping callback
    es_cb = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ES_PATIENCE)

    if DEBUG:
        froc_cb = FROCCallback(validation_set=validation_set,
                               n_optimization=3,
                               save_plot=f"train_output/{args.train_outdir}/froc_callback.png")
    else:
        froc_cb = FROCCallback(validation_set=validation_set,
                               save_plot=f"train_output/{args.train_outdir}/froc_callback.png")
        
    # Train the model we created
    model.fit(train_generator,
              validation_data    = validation_set,
              steps_per_epoch    = len(train_idxs) // batch_size, 
              epochs             = EPOCHS,
              callbacks          = [model_cp_cb, imgs_cb, csv_log_cb, es_cb, froc_cb],
              verbose            = verbose)

    print_p("  -- DONE --  ")
