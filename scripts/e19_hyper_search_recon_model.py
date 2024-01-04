import argparse
import optuna
import multiprocessing
import numpy as np
import os
import SimpleITK as sitk
import tensorflow as tf

from optuna.samplers import TPESampler
from functools import partial
from typing import List, Tuple

from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv3D, concatenate
from tensorflow.keras.layers import MaxPooling3D, UpSampling3D
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.models import load_model

from fastMRI_PCa.utils import create_dirs_if_not_exists, read_yaml_to_dict
from fastMRI_PCa.data import get_generator, IntermediateImagesRecon
from fastMRI_PCa.data import resample, center_crop, normalize, undersample
from fastMRI_PCa.utils import print_p
from fastMRI_PCa.utils import list_from_file, get_rand_exp_decay_mask
from fastMRI_PCa.models import ssim_loss, ssim_metric, psnr_metric

from fastMRI_PCa.utils import does_table_exist


################################  README  ######################################
# OLD - This script will perform a hyper parameter optimization for a
# reconstruction model with optuna. It will train solely on optimizing the 
# metric for T2 weighted images with an acceleration of 8x. The acceleration is 
# simulated in k-space.


def parse_input_args():
    parser = argparse.ArgumentParser(description='Parse arguments for hyper parameter optimization for a reconstruction model with 8 times acceleration.')

    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help="Use this if the code should be run in debug mode. So paramters will be scaled down so that it runs much faster.")

    parser.add_argument('-a',
                        '--acceleration',
            	        type=float,
                        default=8.0,
                        help='Acceleration of the simulated acquisition. R = 1/sampling. In this hyper paramter experiment we will focus on acceleration of 8x.')

    parser.add_argument('-cs',
                        '--centre_sampling',
                        type=float,
                        default=0.5,
                        help='Percentage of SAMPLED k-space that lies central in k-space.')

    parser.add_argument('-tsh',
                        '--target_shape',
                        nargs='+',
                        type=int,
                        default=[192, 192, 20],
                        help='Centre crop [x,y,z] that will be used for training.')
    
    parser.add_argument('-tsp',
                        '--target_space',
                        nargs='+',
                        type=float,
                        default=[0.5, 0.5, 3.0],
                        help='Resampling spacing.')

    parser.add_argument('-n',
                        '--norm',
                        type=str,
                        default="rescale_0_1",
                        help='Normalization method used.')
                        
    parser.add_argument('-to',
                        '--outdir',
                        type=str,
                        help='Directory in train_output where some training results will be stored. A respective model folder will be created with the best performing model during training.')

    parser.add_argument('-nt',
                        '--num_trials',
                        type=int,
                        default=5,
                        help='The number of hyper optimization trials that will be tried to be performed within one execution of the script. This script is meant to run for a long time and restarted without the --is_new_study flag if futher trials are desired.')

    args = parser.parse_args()

    print(f"Input Arguments:\n{args}")

    return args


def set_gpu(debug, gpu_idx):
    # Set the GPU to the second GPU if not running the code via a job script.
    if debug:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_idx}"
        print(f"Set GPU to GPU {gpu_idx}.")


def get_generators(
    train_data: np.ndarray,
    valid_data: np.ndarray,
    train_idxs: List[int],
    valid_idxs: List[int],
    batch_size: int,
    crop_size: Tuple[int, int, int]
):
        train_gen = get_generator(
            batch_size=batch_size,
            shape=crop_size,
            X=train_data,
            Y=valid_data,
            input_sequences=INP_SEQUENCES,
            output_sequences=OUT_SEQUENCES,
            indexes=train_idxs,
            shuffle=True
        )

        valid_gen = get_generator(
            batch_size=None,
            shape=crop_size,
            X=train_data,
            Y=valid_data,
            input_sequences=INP_SEQUENCES,
            output_sequences=OUT_SEQUENCES,
            indexes=valid_idxs,
            shuffle=True,
            augmentation=False
        )

        return train_gen, valid_gen


##################### DATA LOADING FUNCTIONS ###################################


def load_T2_data(
    acceleration: float = 8.0,
    centre_sampling: float = 0.5,
    target_shape: List[int] = [144, 144, 18],
    target_space: List[float] = [0.5, 0.5, 3.0],
    norm: str = "rescale_0_1"
):

    t2_paths = list_from_file(T2_PATH_LIST_FILE)

    n_obs = int(PERCENT_LOAD*len(t2_paths))

    X = parallel_loading_preprocess(nifti_paths=t2_paths[:n_obs],
                                    sampling=(1/acceleration),
                                    centre_sampling=centre_sampling,
                                    target_shape=target_shape,
                                    target_space=target_space,
                                    norm=norm,
                                    do_undersample=True)

    # No undersampling for the label. Load and preprocess
    Y = parallel_loading_preprocess(nifti_paths=t2_paths[:n_obs],
                                    sampling=None,
                                    centre_sampling=None,
                                    target_shape=target_shape,
                                    target_space=target_space,
                                    norm=norm,
                                    do_undersample=False)

    return X, Y


def parallel_loading_preprocess(
    nifti_paths: List[str],
    sampling: float,
    centre_sampling: float,
    target_shape: List[int],
    target_space: List[float],
    norm: str,
    do_undersample: bool
):

    # Start a pool of workers.
    pool = multiprocessing.Pool(processes=N_CPUS)

    # Define a partial function that can undergo a pool.map function.
    load_and_preprocess_partial = partial(load_and_preprocess_t2,
        sampling        = sampling,
        centre_sampling = centre_sampling,
        target_shape    = target_shape,
        target_space    = target_space,
        norm            = norm,
        do_undersample  = do_undersample
    )

    # Apply the load and preprocess function for each file in the given paths
    data_list = pool.map(load_and_preprocess_partial, nifti_paths)
    pool.close()
    
    # Aggregate the data in the first axis.
    data = np.stack(data_list, axis=0)
    data = np.expand_dims(data, 4)

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

    if do_undersample and sampling != 1.0:
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


def build_unet_model(
    crop_size: Tuple[int, int, int] = (192, 192, 20),
    optimizer: str = 'adam',
    last_act_func: str = 'sigmoid',
    loss: str = ssim_loss,
    conv_kernel_size: Tuple[int, int, int] = (3,3,3),
    kernel_regularizer='l2',
    strides: Tuple[int, int, int] = (1,1,1)
):
        # Create NAMED input layers for each sequence
        ct_input  = Input(crop_size + (1,), name="mr_input")

        # Contraction path
        # he_normal defines initial weights - it is a truncated normal distribution (Gaussian dist.)
        # sets padding to same, meaning that input dimensions are the same as output dimensions
        c1 = Conv3D(16, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(ct_input)
        c1 = Conv3D(16, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(c1)
        p1 = MaxPooling3D((2, 2, 1))(c1)

        c2 = Conv3D(32, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(p1)
        c2 = Conv3D(32, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(c2)
        p2 = MaxPooling3D((2, 2, 2))(c2)

        c3 = Conv3D(64, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(p2)
        c3 = Conv3D(64, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(c3)
        p3 = MaxPooling3D((2, 2, 1))(c3)

        c4 = Conv3D(128, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(p3)
        c4 = Conv3D(128, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(c4)
        p4 = MaxPooling3D((2, 2, 2))(c4)

        c5 = Conv3D(256, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(p4)
        c5 = Conv3D(256, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(c5)

        # Upwards U part
        u6 = UpSampling3D((2, 2, 2))(c5)
        u6 = concatenate([u6, c4], axis=-1)
        c6 = Conv3D(128, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(u6)
        c6 = Conv3D(128, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(c6)

        u7 = UpSampling3D((2, 2, 1))(c6)
        u7 = concatenate([u7, c3], axis=-1)
        c7 = Conv3D(64, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(u7)
        c7 = Conv3D(64, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(c7)

        u8 = UpSampling3D((2, 2, 2))(c7)
        u8 = concatenate([u8, c2], axis=-1)
        c8 = Conv3D(32, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(u8)
        c8 = Conv3D(32, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(c8)

        u9 = UpSampling3D((2, 2, 1))(c8)
        u9 = concatenate([u9, c1], axis=-1)
        c9 = Conv3D(16, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(u9)
        c9 = Conv3D(16, conv_kernel_size, strides, activation='relu', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer, padding='same')(c9)

        # Perform 1x1x1 convolution and reduce the feature maps of c9 to 
        # a single channel result.
        # Give it a sigmoid activation to map the outputs to [0-1]
        # 2) NOT YET TESTED, INPUTED NEW OUTPUT LAYER
        output_layer = Conv3D(1, (1, 1, 1), 
            padding='same', 
            name='reconstruction',
            activation=last_act_func
        )(c9)

        if loss == "SSIM":
            loss = ssim_loss

        recon_model = Model(
            inputs=ct_input,
            outputs=output_layer
        )

        recon_model.summary(line_length=200)

        recon_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[ssim_metric, psnr_metric]
        )
        
        return recon_model


def get_train_val_idxs():
    
    idxs_dict = read_yaml_to_dict(TRAIN_VAL_TEST_IDXS_PATH)
    
    if DEBUG:
        train_idxs = list(range(int(PERCENT_LOAD * len(idxs_dict["train_set0"]))))
        valid_idxs = list(range(int(PERCENT_LOAD * len(idxs_dict["val_set0"]))))
    else:
        train_idxs = idxs_dict["train_set0"]
        valid_idxs = idxs_dict["val_set0"]

    print(f"\nDataset division:\nTrain: {len(train_idxs)}")
    print(f"Valid: {len(valid_idxs)}\n")
    
    return train_idxs, valid_idxs


def load_or_create_study(
    is_new_study: bool,
    study_dir: str,
):
    # Create an optuna if it does not exist.
    storage = f"sqlite:///train_output/{study_dir}/{study_dir}.db"
    if is_new_study:
        print(f"Creating a NEW study. With name: {study_dir}")
        study = optuna.create_study(storage=storage,
                                    study_name=study_dir,
                                    direction='maximize',
                                    sampler=TPESampler(n_startup_trials=N_STARTUP_TRIALS))
    else:
        print(f"LOADING study '{study_dir}' from database file.")
        study = optuna.load_study(storage=storage,
                                  study_name=study_dir)

    return study


def get_new_trial_params(trial):
    params = {
        'optimizer':          'adam',
        'last_act_func':      trial.suggest_categorical('last_act_func', ['sigmoid', 'relu', 'tanh']),
        'kernel_regularizer': trial.suggest_categorical('kernel_regularizer', ['l1', 'l2', 'l1_l2']),
        'conv_kernel_size':   trial.suggest_int('conv_kernel_size', 2, 4),
        'strides':            trial.suggest_int('strides', 1, 2)
    }

    # Kernel regularizer and its learning rate
    if params['kernel_regularizer'] == 'l1':
        params['kernel_learning_rate'] = trial.suggest_float('kernel_learning_rate', 1e-5, 1e-3, log=True)
        params['kernel_regularizer'] = regularizers.l1(params['kernel_learning_rate'])
    if params['kernel_regularizer'] == 'l2':
        params['kernel_learning_rate'] = trial.suggest_float('kernel_learning_rate', 1e-5, 1e-3, log=True)
        params['kernel_regularizer'] = regularizers.l2(params['kernel_learning_rate'])
    if params['kernel_regularizer'] == 'l1_l2':
        params['kernel_regularizer'] = regularizers.L1L2()

    # Create optimizer
    if params['optimizer'] == 'adam':
        params['optimizer'] = tf.keras.optimizers.Adam(
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            name="Adam"
        )
    
    # Kernel Size (make into tuple)
    params['conv_kernel_size'] = (params['conv_kernel_size'], params['conv_kernel_size'], params['conv_kernel_size'])
    
    # Strides (make into tuple)
    params['strides']        = (params['strides'], params['strides'], params['strides']) 

    # - Data augmentation params
    # - Instance normalization
    # - More conv layers per block?
    
    print()
    for key in params:
        print(f"{key}: {params[key]}")
    print()
    return params


def ssim_optuna_objective(
    trial,
    train_gen,
    valid_set,
    target_shape: List[int],
    outdir: str,
    acceleration: float,
    train_idxs: List[int],
):
    print("\nStarting new trial.\n")

    train_trial_dir = f"train_output/{outdir}/trial{trial.number}"
    model_trial_dir = f"models/{outdir}/trial{trial.number}"
    create_dirs_if_not_exists([train_trial_dir, model_trial_dir])

    params = get_new_trial_params(trial)
    recon_model = build_unet_model(**params, crop_size=target_shape)

    best_model_filepath = f"{model_trial_dir}/best_val_loss_R{int(acceleration)}.h5"

    model_cp_val_loss_cb = ModelCheckpoint(
        filepath       = best_model_filepath,
        monitor        = 'val_loss', 
        save_best_only = True, 
        mode           = 'min',
        verbose        = VERBOSE
    )

    # This callback predicts on a number of images for the test set after each epoch
    # and shows a few slices in PNG files in the "output/" folder
    imgs_cb = IntermediateImagesRecon(
        validation_set   = valid_set, 
        prefix           = f"recon_R{int(acceleration)}",
        train_outdir     = f"{outdir}/trial{trial.number}",
        num_images       = 6,
        input_sequences  = INP_SEQUENCES,
        output_sequences = OUT_SEQUENCES
    )

    # This produces a log file of the train and validation metrics at each epoch
    csv_log_cb = CSVLogger(f"{train_trial_dir}/train_log_R{int(acceleration)}_trial{trial.number}.csv")
    es_cb = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ES_PATIENCE, min_delta=ES_MIN_DELTA)

    callbacks = [           # The order of the callbacks is really IMPORTANT!
        model_cp_val_loss_cb,
        imgs_cb,
        es_cb,
        csv_log_cb
    ]

    # Train the model we created
    recon_model.fit(train_gen,
        validation_data    = valid_set,
        steps_per_epoch    = len(train_idxs) // BATCH_SIZE, 
        epochs             = EPOCHS,
        callbacks          = callbacks,
        verbose            = VERBOSE,
    )

    recon_model = load_model(
        filepath = best_model_filepath,
        custom_objects={'ssim_loss': ssim_loss, 'ssim_metric': ssim_metric, 'psnr_metric': psnr_metric}
    )

    scores = recon_model.evaluate(x=valid_set[0], y=valid_set[1])
    
    print(f"\nValidation Scores: {scores}")
    print('Validation loss:', scores[0]) 
    print('Validation SSIM:', scores[1])
    print('Validation PSNR:', scores[2])

    return scores[1]


################################### CONSTANTS ##################################
DEBUG        = False
N_CPUS       = 12             # Num CPUs used for data preprocessing and loading
BATCH_SIZE   = 15
VERBOSE      = 2              # 2=Supress a lot of printing
PERCENT_LOAD = 1.0            # 1.0 = 100% Data loading
SEED         = 4321

T2_PATH_LIST_FILE        = "data/path_lists/pirads_4plus/current_t2w.txt"
TRAIN_VAL_TEST_IDXS_PATH = "data/path_lists/pirads_4plus/train_val_test_idxs.yml"

INP_SEQUENCES = ["t2"]
OUT_SEQUENCES = ["t2_label"]

ES_PATIENCE      = 10
ES_MIN_DELTA     = 0.005
EPOCHS           = 500
N_STARTUP_TRIALS = 5

if __name__ == '__main__':

    tf.compat.v1.disable_eager_execution()
    print("start")
    args = parse_input_args()

    if args.debug:
        DEBUG        = True
        BATCH_SIZE   = 2      # Smaller batchsize for less resource consumption
        VERBOSE      = 1      # More printing
        N_CPUS       = 4      # Less CPUs for less resource consumption
        PERCENT_LOAD = 0.05   # Less percentage data loading if in debug
        EPOCHS       = 2      # In debug mode, just a couple of epochs is fine
        set_gpu(DEBUG, gpu_idx=1)

    create_dirs_if_not_exists([f"models/{args.outdir}", f"train_output/{args.outdir}"])

    # Load the data before running hyper parameter optimization with optuna
    X, Y = load_T2_data(
        acceleration=args.acceleration,
        centre_sampling=args.centre_sampling,
        target_shape=args.target_shape,
        target_space=args.target_space,
        norm=args.norm
    )

    train_idxs, valid_idxs = get_train_val_idxs()

    train_gen, valid_gen = get_generators(
        train_data=X,
        valid_data=Y,
        train_idxs=train_idxs,
        valid_idxs=valid_idxs,
        batch_size=BATCH_SIZE,
        crop_size=tuple(args.target_shape)
    )

    train_set = next(train_gen)
    valid_set = next(valid_gen)

    # START OPTUNA FROM HERE ON
    table_exists = does_table_exist('trials', f"train_output/{args.outdir}/{args.outdir}.db")
    study = load_or_create_study(is_new_study=not table_exists, study_dir=args.outdir)
    
    opt_func = lambda trail: ssim_optuna_objective(trail,
                                                   train_gen    = train_gen,
                                                   valid_set    = valid_set,
                                                   target_shape = tuple(args.target_shape),
                                                   outdir       = args.outdir,
                                                   acceleration = args.acceleration,
                                                   train_idxs   = train_idxs
    )
    study.optimize(opt_func, n_trials=args.num_trials)

    print("Optimization for this script complete for now.")