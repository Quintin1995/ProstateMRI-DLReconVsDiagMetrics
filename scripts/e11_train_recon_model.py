import argparse
import multiprocessing
import numpy as np
import os
import SimpleITK as sitk
import tensorflow as tf
import time
import uuid

from datetime import date
from functools import partial
from umcglib.plotting import save_slice_2d
from sklearn.model_selection import KFold
from typing import List

from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv3D, concatenate
from tensorflow.keras.layers import MaxPooling3D, UpSampling3D
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from fastMRI_PCa.utils import create_dirs_if_not_exists, read_yaml_to_dict
from fastMRI_PCa.data import get_generator, IntermediateImagesRecon
from fastMRI_PCa.data import resample, center_crop, normalize, undersample
from fastMRI_PCa.utils import print_stats_np, print_p, dump_dict_to_yaml
from fastMRI_PCa.utils import list_from_file
from fastMRI_PCa.models import ssim_loss, ssim_metric, psnr_metric
from fastMRI_PCa.utils.k_space_masks import get_rand_exp_decay_mask_ac_matrix


################################  README  ######################################
# OLD - This script will undersample T2 images and train a reconstruction model on it.
# Data will be loaded, undersampled and preprocessed in parallel on multiple
# CPUs and loaded into RAM.


################################ PARSER ########################################

def parse_input_args():
    parser = argparse.ArgumentParser(description='Parse arguments for training a Reconstruction model')

    parser.add_argument('fold_num',
                        type=int,
                        help='The current fold number to be run. Not the be confused with the total number of folds')

    parser.add_argument('num_folds',
                        type=int,
                        help='The number of folds in total.')

    parser.add_argument('--job',
                        dest='is_job',
                        action='store_true',
                        help="Whether the current code is run via a job script. Use --job for when it is not job.sh script ")

    parser.add_argument('--no_job',
                        dest='is_job',
                        action='store_false',
                        help="Whether the current code is run via a job script. Use --no_job for when it is not job.sh script")

    parser.add_argument('--acceleration',
            	        type=float,
                        default=1.0,
                        help='Acceleration of the simulated acquisition. R = 1/sampling')

    parser.add_argument('--centre_sampling',
                        type=float,
                        default=0.5,
                        help='Percentage of SAMPLED k-space that lies central in k-space.')

    parser.add_argument('--target_shape',
                        type=List[int],
                        default=[192, 192, 20],
                        help='Centre crop [x,y,z] that will be used for training')
    
    parser.add_argument('--target_space',
                        type=List[float],
                        default=[0.5, 0.5, 3.0],
                        help='Resampling spacing.')

    parser.add_argument('--norm',
                        type=str,
                        default="rescale_0_1",
                        help='Normalization method used.')
                        
    parser.add_argument('--outdir',
                        type=str,
                        help='Directory in train_output where some training results will be stored. A respective model folder will be created with the best performing model during training.')

    args = parser.parse_args()
    return args


################################ Functions #####################################

def parallel_loading_preprocess(
    nifti_paths: List[str],
    sampling: float,
    centre_sampling: float,
    target_shape: List[int],
    target_space: List[float],
    norm: str,
    do_undersample: bool,
    seed: int,
    dicom_db_path: str = r'sqliteDB/dicoms.db'
):

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
                                          do_undersample=do_undersample,
                                          seed=seed,
                                          dicom_db_path=dicom_db_path)

    # Apply the load and preprocess function for each file in the given paths
    data_list = pool.map(load_and_preprocess_partial, nifti_paths)
    pool.close()
    
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
    seed: int,
    do_undersample: bool,
    dicom_db_path: str,
):

    print_p(f"Processing T2 mri: {nifti_path}")
    
    # Read the T2 transversal file as SimpleITK object
    t2_tra_s = sitk.ReadImage(nifti_path, sitk.sitkFloat32)

    if do_undersample and sampling != 1.0:
        # Build patient mask
        # mask = get_rand_exp_decay_mask(width=t2_tra_s.GetSize()[0],
        #                                height=t2_tra_s.GetSize()[1],
        #                                sampling=sampling,
        #                                centre_sampling=centre_sampling,
        #                                seed=SEED,
        #                                verbatim=False)

        mask = get_rand_exp_decay_mask_ac_matrix(
            width           = t2_tra_s.GetSize()[0],
            height          = t2_tra_s.GetSize()[1],
            sampling        = sampling,
            centre_sampling = centre_sampling,
            seed            = seed,
            exp_scale       = 0.4,      # determined emperically
            nifti_path      = nifti_path,
            dicom_db_path   = dicom_db_path,
            tablename       = 'dicom_headers_v2',
            verbatim        = False,
        )

        # Perform undersampling in k-space with the given mask.
        t2_tra_s = undersample(t2_tra_s, mask)

        if mask.shape != (384, 384) and False:
            print(mask.shape)
            temp_str = get_unique_fname(5)
            temp_path = os.path.join('temp', f"mask{temp_str}.png")
            save_slice_2d(mask, temp_path)

    # Preprocess
    t2_tra_s = resample(t2_tra_s, 
                        min_shape=target_shape, 
                        method=sitk.sitkLinear, 
                        new_spacing=target_space)
    t2_tra_s = center_crop(t2_tra_s, target_shape)
    t2_tra_s = normalize(norm_str=norm, img_s=t2_tra_s)

    return sitk.GetArrayFromImage(t2_tra_s).T


def get_unique_fname(length):
    return str(uuid.uuid4().hex)[:length]


################################## CONSTANTS ###################################

DESCRIPTION = """Trains a standard U-net to reconstruct undersampled T2 images."""

# Path to file with train/val/test indexes
IDXS_PATH = "data/path_lists/pirads_4plus/train_val_test_idxs.yml"

# Number of CPUs used for loading and preprocessing
N_CPUS = 12
#Path to text file with all T2 paths in it.
t2_path_list_file = "data/path_lists/pirads_4plus/current_t2w.txt" 
OPTIMIZER = "adam"
LAST_ACTIVATION = 'sigmoid'
LOSS = "SSIM"

INP_SEQUENCES = ["t2"]
OUT_SEQUENCES = ["t2_label"]
EPOCHS = 1000
SEED   = 3478+10+10

KSPACE_MASK_SAVE_CHANCE = 0.007         # 0.007 for real experiments.

ES_PATIENCE = 50


if __name__ == '__main__':

    args = parse_input_args()

    KSPACE_MASKS_OUTDIR = f"train_output/{args.outdir}/kspace_mask_examples"
    
    batch_size = 10 if args.is_job else 2     # if job, then batch_size=10
    verbose = 2 if args.is_job else 1         # if job, then less text in slurm
    N_CPUS = 12 if args.is_job else 2         # decrease num cpus if not a job
    perc_load = 1.0 if args.is_job else 0.25  # decrease num imgs loaded if not a job
    
    if not args.is_job:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # select second GPU if not a job

    params = {
        "is_job": args.is_job,
        "type_model": "reconstruction_model",
        "description": DESCRIPTION,
        "train_output_dir": args.outdir,
        "models_outdir": args.outdir,
        "input_sequences": INP_SEQUENCES,
        "output_sequences": OUT_SEQUENCES,
        "batch_size": batch_size,
        "t2_path_list_file": t2_path_list_file,
        "verbose": verbose,
        "optimizer": OPTIMIZER,
        "seed": SEED,
        "number_folds": args.num_folds,
        "epochs": EPOCHS,
        "last_activation": LAST_ACTIVATION,
        "loss": LOSS,
        "sampling": (1/args.acceleration),
        "acceleration": args.acceleration,
        "centre_sampling": args.centre_sampling,
        "target_shape": args.target_shape,
        "target_space": args.target_space,
        "norm": args.norm,
        "datetime": date.today().strftime("%Y-%m-%d")
    }
    dump_dict_to_yaml(params, f"train_output/{args.outdir}", filename=f"params", verbose=False)
    
    # Create some dirs for training
    create_dirs_if_not_exists([f"models/{args.outdir}", f"train_output/{args.outdir}"])
    create_dirs_if_not_exists(f"temp/{args.outdir}")
    create_dirs_if_not_exists(KSPACE_MASKS_OUTDIR)

    # Load the T2 images in parallel. Load, undersample and preprocess
    t2_paths = list_from_file(t2_path_list_file)
    print("-"*20 + "\n")
    print_p(f"Amount MRIs loading {int(perc_load*len(t2_paths))}...")
    X = parallel_loading_preprocess(nifti_paths=t2_paths[:int(perc_load*len(t2_paths))],
                                    sampling=(1/args.acceleration),
                                    centre_sampling=args.centre_sampling,
                                    target_shape=args.target_shape,
                                    target_space=args.target_space,
                                    norm=args.norm,
                                    do_undersample=True,
                                    seed=SEED)

    # No undersampling for the label. Load and preprocess
    Y = parallel_loading_preprocess(nifti_paths=t2_paths[:int(perc_load*len(t2_paths))],
                                    sampling=(1/args.acceleration),
                                    centre_sampling=args.centre_sampling,
                                    target_shape=args.target_shape,
                                    target_space=args.target_space,
                                    norm=args.norm,
                                    do_undersample=False,
                                    seed=SEED)

    print_stats_np(X, "Train data")
    print_stats_np(Y, "Target data")

    if not args.is_job:
        for mri_idx in range(5):
            slice_idx = 10
            path1 = os.path.join('temp', args.outdir, f"mri{mri_idx}_slice{slice_idx}_a{args.acceleration}_X.png")
            path2 = os.path.join('temp', args.outdir, f"mri{mri_idx}_slice{slice_idx}_a{args.acceleration}_Y.png")
            save_slice_2d(np.squeeze(X[mri_idx, :, :, slice_idx, :]), path1)
            save_slice_2d(np.squeeze(Y[mri_idx, :, :, slice_idx, :]), path2)
            print_p(f">Wrote to: {path1}")
            print_p(f">Wrote to: {path2}")
            
    # Get the number of observations
    num_obs = X.shape[0]
    print_p(f"Number of observations: {num_obs}")
    print_p(f"Fold number: {args.fold_num+1} of {args.num_folds}")

    # Randomly split the data into a train (80%) / validation (10%) / test (10%)
    kfold = KFold(args.num_folds, shuffle=True, random_state=SEED)
    
    if args.is_job:
        idxs_dict = read_yaml_to_dict(IDXS_PATH)
        all_idxs = idxs_dict["train_set0"]
    else:
        all_idxs = list(range(num_obs))

    train_idxs, valid_idxs = list(kfold.split(all_idxs))[args.fold_num]
    train_idxs = list(train_idxs)
    valid_idxs = list(valid_idxs)
        
    print_p(f"Dataset division:\n\t- Train: {len(train_idxs)} = {len(train_idxs)/len(all_idxs)*100}%")
    print_p(f"\t- Valid: {len(valid_idxs)} = {len(valid_idxs)/len(all_idxs)*100}%")
    print_p(f"\t- Test: {0}")
    print_p(f"Validation indexes: {valid_idxs}")

    crop_size = X[0].shape[:3]
    print(f"Crop shape for generator: {crop_size}")

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
    print(f"Validation set: {validation_set[0].shape}")

    tf.compat.v1.disable_eager_execution()

    # Create NAMED input layers for each sequence
    ct_input  = Input(crop_size + (1,), name="mr_input")

    # Contraction path
    # he_normal defines initial weights - it is a truncated normal distribution (Gaussian dist.)
    # sets padding to same, meaning that input dimensions are the same as output dimensions
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(ct_input)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 1))(c1)

    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(p1)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(p2)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 1))(c3)

    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(p3)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(c4)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(p4)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(c5)

    # Upwards U part
    u6 = UpSampling3D((2, 2, 2))(c5)
    u6 = concatenate([u6, c4], axis=-1)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(u6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(c6)

    u7 = UpSampling3D((2, 2, 1))(c6)
    u7 = concatenate([u7, c3], axis=-1)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(u7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(c7)

    u8 = UpSampling3D((2, 2, 2))(c7)
    u8 = concatenate([u8, c2], axis=-1)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(u8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(c8)

    u9 = UpSampling3D((2, 2, 1))(c8)
    u9 = concatenate([u9, c1], axis=-1)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(u9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0001), padding='same')(c9)

    # Perform 1x1x1 convolution and reduce the feature maps of c9 to 
    # a single channel result.
    # Give it a sigmoid activation to map the outputs to [0-1]
    # 2) NOT YET TESTED, INPUTED NEW OUTPUT LAYER
    output_layer = Conv3D(1, (1, 1, 1), 
        padding='same', 
        name='reconstruction',
        activation=LAST_ACTIVATION
        )(c9)

    # Deal with losses and metrics
    # mse_metric = MeanSquaredError(name = "mse")
    if LOSS == "SSIM":
        loss = ssim_loss

    # Create the model and show summary
    dnn = Model(
        inputs=ct_input,
        outputs=output_layer
        )
    dnn.summary(line_length=200)
    dnn.compile(
        optimizer=OPTIMIZER,
        loss=loss,
        metrics=[ssim_metric, psnr_metric, 'mse']
    )

    # CALLBACKS
    # After every epoch store the model with the best validation performance on each metric
    model_cp_val_loss = ModelCheckpoint(f"models/{args.outdir}/best-direct-fold{args.fold_num}_val_loss.h5",
        monitor = 'val_loss', 
        save_best_only=True, 
        mode='min',
        verbose=verbose)

    model_cp_val_psnr = ModelCheckpoint(f"models/{args.outdir}/best-direct-fold{args.fold_num}_val_psnr.h5",
        monitor = 'val_psnr_metric', 
        save_best_only=True, 
        mode='max',
        verbose=verbose)

    model_cp_val_mse = ModelCheckpoint(f"models/{args.outdir}/best-direct-fold{args.fold_num}_val_mse.h5",
        monitor = 'val_mse', 
        save_best_only=True, 
        mode='min',
        verbose=verbose)

    # This callback predicts on a number of images for the test set after each epoch
    # and shows a few slices in PNG files in the "output/" folder
    images_callback = IntermediateImagesRecon(validation_set, 
                                              prefix=f"recon_fold{args.fold_num}",
                                              train_outdir=args.outdir,
                                              num_images=10,
                                              input_sequences=INP_SEQUENCES,
                                              output_sequences=OUT_SEQUENCES)

    # This callback produces a log file of the training and validation metrics at 
    # each epoch
    csv_log_callback = CSVLogger(f"train_output/{args.outdir}/train_direct_log_fold{args.fold_num}.csv")

    # Early stopping callback
    es_cb = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ES_PATIENCE)

    callbacks = [model_cp_val_loss,
                 model_cp_val_psnr,
                 model_cp_val_mse,
                 images_callback,
                 csv_log_callback,
                 es_cb]

    # Train the model we created
    dnn.fit(train_generator,
            validation_data    = validation_set,
            steps_per_epoch    = len(train_idxs) // batch_size, 
            epochs             = EPOCHS,
            callbacks          = callbacks,
            verbose            = verbose,
            )

    print("-- Done --")