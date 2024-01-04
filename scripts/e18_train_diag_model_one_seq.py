import argparse
import numpy as np
import time
import multiprocessing
import SimpleITK as sitk
import os
import random
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

from datetime import date
from focal_loss import BinaryFocalLoss
from sklearn.model_selection import KFold
from typing import List
from functools import partial

from tensorflow.keras.models import load_model
from fastMRI_PCa.visualization.visualize import save_array2d_to_image
from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from fastMRI_PCa.visualization import save_slice_3d, write_array2nifti_t2
from fastMRI_PCa.models import ssim_loss, ssim_metric, psnr_metric, weighted_binary_cross_entropy
from fastMRI_PCa.utils import print_p, print_stats_np, dump_dict_to_yaml, get_unique_fname
from fastMRI_PCa.utils import list_from_file, get_rand_exp_decay_mask, create_dirs_if_not_exists
from fastMRI_PCa.data import get_generator, IntermediateImagesDiag
from fastMRI_PCa.utils import read_yaml_to_dict

from umcglib.callbacks import FROCCallback
from umcglib.models.unet import build_dual_attention_unet
import umcglib.images as im


################################  README  ######################################
# OLD -This script will load ONE of the following sequences: (undersampled)T2W,
# DWI or ADC. The data will be used to train a diagnostic network where lesions
# are segmented for the patient. Model = Dual Attention U-net from Anindo.
# Goal: Train a diagnostic U-Net on ONE input sequence type.


def parse_input_args():

    help = """This script will load the following data: DWI images, ADC maps and
              undersampled T2w images. The data will be used to train a
              diagnostic network where lesions are segmented for the patient.
              Goal: Train a diagnostic U-Net. Trains a model on ONE input sequence."""

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

    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help="Use this if the code should be run in debug mode. So paramters will be scaled down so that it runs much faster.")

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

    parser.add_argument('-i',
                        '--input_modality',
                        type=str,
                        help='Choose which input modality/sequence is used for reconstruction. Options = ("normal_t2", "normal_recon_t2", "deep_learn_recon_t2", "dwi", "adc")')

    args = parser.parse_args()

    if args.input_modality != None:
        assert args.input_modality in ["normal_t2", "normal_recon_t2", "deep_learn_recon_t2", "dwi", "adc"], 'Please choose a valid input modality'
    assert args.norm in ["znorm", "rescale_0_1"], 'Please choose a valid normalization method.'

    if args.input_modality == "deep_learn_recon_t2":
        assert args.t2_recon_dir != None, "Please select a T2 Reconstruction model directory if the input modality is deep_learn_recon_t2"

    print(f"\nAll input parameters: {args}\n")
    return args


################################################################################


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
        
        if random.random() < 0.001:
            save_array2d_to_image(mask, "K-Space Mask", "temp/kspace_masks", f"mask_{get_unique_fname()}")

        # Perform undersampling in k-space with the given mask.
        t2_tra_s = im.undersample_kspace(t2_tra_s, mask)

    # Preprocess
    t2_tra_s = im.resample(
        image       = t2_tra_s, 
        min_shape   = target_shape, 
        method      = sitk.sitkNearestNeighbor, 
        new_spacing = target_space
    )
    t2_tra_s = im.center_crop(image=t2_tra_s, shape=target_shape)
    t2_tra_s = im.normalize_s(norm_str=norm, img_s=t2_tra_s)

    return sitk.GetArrayFromImage(t2_tra_s).T


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
    mri_tra_s = im.resample(
        image=mri_tra_s, 
        min_shape=target_shape, 
        method=sitk.sitkNearestNeighbor, 
        new_spacing=target_space
    )
    mri_tra_s = im.center_crop(mri_tra_s, target_shape)
    if seq != 'seg':
        mri_tra_s = im.normalize_s(norm_str=norm, img_s=mri_tra_s)
    else:
        mri_tra_s = im.binarize_s(mri_tra_s)

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
                            centre_sampling=centre_sampling,
                            target_shape=target_shape,
                            target_space=target_space,
                            norm="rescale_0_1",
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


def load_reconstructed_t2s(
    nifti_paths: List[str],
    seq: str,
    sampling: float,
    centre_sampling: float,
    target_shape: List[int],
    target_space: List[float],
    norm: str,
    t2_recon_dir: str):

    if sampling == None:        # We just want the regular T2 images without undersampling
        # Load the T2 images (reconstructions) in parallel as numpy array
        recons_ifft = parallel_loading_preprocess(nifti_paths=nifti_paths,
                                                  seq=seq,
                                                  sampling=sampling,
                                                  centre_sampling=centre_sampling,
                                                  target_shape=target_shape,
                                                  target_space=target_space,
                                                  norm=norm,
                                                  do_undersample=False)
    else:
        # Load the T2 images (reconstructions) in parallel as numpy array
        recons_ifft = parallel_loading_preprocess(nifti_paths=nifti_paths,
                                                  seq=seq,
                                                  sampling=sampling,
                                                  centre_sampling=centre_sampling,
                                                  target_shape=target_shape,
                                                  target_space=target_space,
                                                  norm=norm,
                                                  do_undersample=True)

    # if DEBUG:
    #     slice_ = 6
    #     save_slice_3d(recons_ifft[1], slice_, f"temp", f"{t2_recon_dir}_ifft_slice{slice_}_{seq}_a{1/sampling}")
    #     write_array2nifti_t2(recons_ifft[1], f"temp", f"{t2_recon_dir}_ifft_{seq}_a{1/sampling}")

    # if t2_recon_dir not given as input argument we will simply load the IFFT reconstruction for the diagnostic model.
    if t2_recon_dir == None:
        print("Normal Reconstructions loaded. (not Deep Learning reconstructions but simple IFFT reconstructions.)")
        return np.squeeze(recons_ifft)
    else:
        # Load T2 reconstruction model.
        t2_model_path = f"models/{t2_recon_dir}/best-direct-fold0_val_loss.h5"
        recon_model = load_model(t2_model_path, custom_objects={'ssim_loss': ssim_loss, 'ssim_metric': ssim_metric, 'psnr_metric': psnr_metric})
        recon_model.summary()

        if recons_ifft.ndim == 4:
            recons_ifft = np.expand_dims(recons_ifft, 4)

        if DEBUG:
            recons_dl = np.squeeze(recon_model.predict(recons_ifft, batch_size=1))
        else:
            recons_dl = np.squeeze(recon_model.predict(recons_ifft, batch_size=10))
    
        print_stats_np(recons_dl, "Recons_dl")
        print("Deep Learning Reconstructions loaded.")
        return recons_dl


def load_data(
    norm: str,
    sampling: float,
    centre_sampling: float,
    target_shape: List[int],
    target_space: List[float],
    t2_recon_dir: str,
    inp_seq: str,
    train_dir: str
):
    
    if inp_seq in ["normal_recon_t2", "deep_learn_recon_t2", "normal_t2"]:
        t2_paths = list_from_file(T2_LIST_PATH)
        last_idx = LAST_INDEX if DEBUG else len(t2_paths)
        X = load_reconstructed_t2s(t2_paths[0:last_idx],
                                   seq='T2',
                                   sampling=sampling,
                                   centre_sampling=centre_sampling,
                                   target_shape=target_shape,
                                   target_space=target_space,
                                   norm=norm,
                                   t2_recon_dir=t2_recon_dir)
    
    if inp_seq == "dwi":
        dwi_paths = list_from_file(DWI_LIST_PATH)
        last_idx = LAST_INDEX if DEBUG else len(dwi_paths)
        X = parallel_loading_preprocess(dwi_paths[0:last_idx],
                                        seq='dwi',
                                        sampling=None,
                                        centre_sampling=None,
                                        target_shape=target_shape,
                                        target_space=target_space,
                                        norm=norm,
                                        do_undersample=None)

    if inp_seq == "adc":
        adc_paths = list_from_file(ADC_LIST_PATH)
        last_idx = LAST_INDEX if DEBUG else len(adc_paths)
        X = parallel_loading_preprocess(adc_paths[0:last_idx],
                                        seq='ADC',
                                        sampling=None,
                                        centre_sampling=None,
                                        target_shape=target_shape,
                                        target_space=target_space,
                                        norm=norm,
                                        do_undersample=None)

    # LABEL Segmentation images: Load - preprocess.
    seg_paths = list_from_file(SEG_LIST_PATH)
    last_idx = LAST_INDEX if DEBUG else len(seg_paths)
    Y = parallel_loading_preprocess(seg_paths[0:last_idx],
                                    seq='seg',
                                    sampling=None,
                                    centre_sampling=None,
                                    target_shape=target_shape,
                                    target_space=target_space,
                                    norm=norm,
                                    do_undersample=None)
   
    if DEBUG:
        slice_ = 6
        if sampling != None:
            save_slice_3d(X[1], slice_, f"temp/{train_dir}", f"slice{slice_}_{inp_seq}_a{1/sampling}")
            write_array2nifti_t2(X[1], f"temp/{train_dir}", f"{inp_seq}_a{1/sampling}")
        else:
            save_slice_3d(X[1], slice_, f"temp/{train_dir}", f"slice{slice_}_{inp_seq}")
            write_array2nifti_t2(X[1], f"temp/{train_dir}", f"{inp_seq}")

    # # Add output sequence dim. So that len(X.dims) == len(segs.dims)
    X = np.expand_dims(X, axis=4)
    Y = np.expand_dims(Y, axis=4)
    
    print_stats_np(X, f"Train array with input seq {inp_seq}")
    print_stats_np(Y, f"Label array (segmentations)")

    return X, Y


def set_gpu(debug, gpu_idx):
    # Set the GPU to the second GPU if not running the code via a job script.
    if debug:
        print(f"Setting GPU to GPU {gpu_idx}.")
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_idx}"


def get_params(args, do_dump_to_file=True):
    
    # Set all basic parameters
    params = {
        "type_model": f"one_seq_diagnostic_model_{args.input_modality}",
        "description": DESCRIPTION,
        "debug": args.debug,
        "train_output_dir": args.train_outdir,
        "models_outdir": args.train_outdir,
        "input_modality": args.input_modality,
        "output_sequences": OUT_SEQUENCES,
        "batch_size": BATCH_SIZE,
        "verbose": VERBOSE,
        "optimizer": OPTIMIZER,
        "seed": SEED,
        "fold_num": args.fold_num,
        "num_folds": args.num_folds,
        "epochs": EPOCHS,
        # "last_activation": LAST_ACTIVATION,
        "loss": LOSS,
        "norm": args.norm,
        "train_setkey": SETKEY,
        "es_patience": ES_PATIENCE,
        "es_min_delta": ES_MIN_DELTA,
        "datetime": date.today().strftime("%Y-%m-%d")
        }

    # Determine the type of reconstruction parameters. Either from reconstruction model or given by argparse input arguments.
    if args.input_modality in ["deep_learn_recon_t2"]:
        recon_params = read_yaml_to_dict(f"train_output/{args.t2_recon_dir}/params.yml")

    # if normal reconstruction is chosen, then undersampling parameters need to be given as input parameters.
    if args.input_modality in ["normal_recon_t2"]:
        recon_params = {
            "sampling": 1/args.acceleration,
            "acceleration": args.acceleration,
            "centre_sampling": args.centre_sampling,
            "target_shape": args.target_shape,
            "target_space": args.target_space
        }

    # Set some param items equal to that of recon params
    if args.input_modality in ["normal_recon_t2", "deep_learn_recon_t2"]:
        params["t2_recon_dir"]    =  args.t2_recon_dir
        params["sampling"]        =  recon_params["sampling"]
        params["acceleration"]    =  recon_params["acceleration"]
        params["centre_sampling"] =  recon_params["centre_sampling"]
        params["target_shape"]    =  recon_params["target_shape"]
        params["target_space"]    =  recon_params["target_space"]

    if args.input_modality in ["normal_t2", "dwi", "adc"]:
        params["t2_recon_dir"]    = None
        params["sampling"]        = None
        params["centre_sampling"] = None
        params["target_shape"]    = args.target_shape
        params["target_space"]    = args.target_space
        
    # In most cases we need to dump input params to file for later use
    if do_dump_to_file:
        dump_dict_to_yaml(params, f"train_output/{args.train_outdir}", filename=f"params")

    return params


def get_idxs_from_kfold(
    num_obs: int,
    num_folds: int,
    fold_num: int,
    setkey: str):

    # Get the number of observations
    print_p(f"\nNumber of observations (loaded): {num_obs}")
    print_p(f"\nFold number: {params['fold_num']+1} of {params['num_folds']}")

    # Split the data according to the number of folds selected in the input arguments.
    kfold = KFold(num_folds, shuffle=True, random_state=SEED)
    
    if not DEBUG:
        idxs_dict = read_yaml_to_dict(IDXS_PATH)
        all_idxs = idxs_dict[setkey]
    else:
        all_idxs = list(range(num_obs))
    
    train_idxs, valid_idxs = list(kfold.split(all_idxs))[fold_num]

    print(f"\nDataset division:\n\t- Train: {len(train_idxs)} = {len(train_idxs)/len(all_idxs)*100}%")
    print(f"\t- Valid: {len(valid_idxs)} = {len(valid_idxs)/len(all_idxs)*100}%")
    print(f"\t- Test: {0}")
    print(f"Validation indexes: {valid_idxs}")
    
    return list(train_idxs), list(valid_idxs)

################################### ######### ##################################


if __name__ == '__main__':

    DESCRIPTION = """Trains a dual attention U-net on either DWI only, ADC only or undersampled T2 images only for prostate cancer lesion segmentation with label segmentation maps."""

    T2_LIST_PATH  = "data/path_lists/pirads_4plus/current_t2w.txt"
    DWI_LIST_PATH = "data/path_lists/pirads_4plus/current_dwi.txt"
    ADC_LIST_PATH = "data/path_lists/pirads_4plus/current_adc.txt"
    SEG_LIST_PATH = "data/path_lists/pirads_4plus/seg_new.txt"

    # Path to file with train/val/test indexes
    IDXS_PATH = "data/path_lists/pirads_4plus/train_val_test_idxs.yml"
    SETKEY    = "train_set0"       # This set will be chosen for kfold. From the yaml in the line above.

    ES_PATIENCE  = 100           # Early stopping patience
    ES_MIN_DELTA = 0.005        # Performance should at least increase by this much.

    # LOSS = "BinaryFocalLoss"
    LOSS = "weighted_binary_cross_entropy"

    FOCAL_LOSS_GAMMA = 2.0       # Emperically determined by Chris and Anindo
    FOCAL_LOSS_POS_WEIGHT = 0.25 # Emperically determined by Chris and Anindo

    LEARNING_RATE = 1e-4
    OPTIMIZER = "adam"

    # Number of CPUs for data loading undersampling and preprocessing
    SEED = 3478 + 10 + 10
    OUT_SEQUENCES = ["seg"]
    EPOCHS = 1000

    # Get all input parameters
    args = parse_input_args()

    # See if the code is run on the login node or run as a job script.
    DEBUG = True if args.debug else False

    # The following to stop exessive RAM consumption
    disable_eager_execution()

    BATCH_SIZE = 10 if not DEBUG else 2     # if job, then batch_size=10
    VERBOSE = 2 if not DEBUG else 1         # if job, then less text in slurm
    N_CPUS = 12 if not DEBUG else 2         # decrease num cpus if not a job
    LAST_INDEX = -1 if not DEBUG else 60    # Use less data when it is not a job.
    set_gpu(DEBUG, gpu_idx=1)               # Choose different gpu if in DEBUG mode

    # Params is enriched with params from file if chosen in args.
    params = get_params(args, do_dump_to_file=True)
    
    # Create directories where train results and the best model will be stored.
    folders = [
        f"models/{params['train_output_dir']}",
        f"train_output/{params['train_output_dir']}",
        f"temp/{params['train_output_dir']}"
    ]
    create_dirs_if_not_exists(folders)

    # Real start of code
    
    # Load input and label data.
    X, Y = load_data(
        norm            = params['norm'],
        sampling        = params["sampling"],
        centre_sampling = params["centre_sampling"],
        target_shape    = params["target_shape"],
        target_space    = params["target_space"],
        t2_recon_dir    = params['t2_recon_dir'],
        inp_seq         = params['input_modality'],
        train_dir       = params['train_output_dir']
    )

    train_idxs, valid_idxs = get_idxs_from_kfold(
        num_obs=X.shape[0],
        fold_num=params['fold_num'],
        num_folds=params['num_folds'],
        setkey=SETKEY
    )

    train_generator = get_generator(
        X                = X,
        Y                = Y,
        batch_size       = BATCH_SIZE,
        shape            = tuple(params['target_shape']),
        input_sequences  = [params['input_modality']],
        output_sequences = OUT_SEQUENCES,
        indexes          = train_idxs,
        shuffle          = True
    )

    train_set = next(train_generator)
    print(f"\ntrain set: {train_set[0].shape}")

    validation_generator = get_generator(
        X                = X,
        Y                = Y,
        batch_size       = None,
        shape            = tuple(params['target_shape']),
        input_sequences  = [params['input_modality']],
        output_sequences = OUT_SEQUENCES,
        indexes          = valid_idxs,
        shuffle          = True,
        augmentation     = False
    )

    validation_set = next(validation_generator)
    print(f"Validation set: {validation_set[0].shape}\n")

    # Load and compile the model.
    model = build_dual_attention_unet(input_shape=X[0].shape)

    if LOSS == "BinaryFocalLoss":
        loss = BinaryFocalLoss(gamma=FOCAL_LOSS_GAMMA, pos_weight=FOCAL_LOSS_POS_WEIGHT)
    if LOSS == "weighted_binary_cross_entropy":
        weight_for_0 = 0.05
        weight_for_1 = 0.95
        loss = weighted_binary_cross_entropy({0: weight_for_0, 1: weight_for_1})
    if OPTIMIZER == "adam":
        optimizer = Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer, 
        loss=loss,
        metrics=["AUC"]
    )

    model.summary(line_length=200)

    if DEBUG:
        froc_cb = FROCCallback(
            validation_set=validation_set,
            n_optimization=3,
            delay_epochs=2,
            optimization_interval=2,
            save_froc_plot=f"train_output/{params['train_output_dir']}/{params['train_output_dir']}_froc_callback.png",
            save_roc_plot =f"train_output/{params['train_output_dir']}/{params['train_output_dir']}_roc_callback.png"
        )
    else:
        froc_cb = FROCCallback(
            validation_set=validation_set,
            save_froc_plot=f"train_output/{params['train_output_dir']}/{params['train_output_dir']}_froc_callback.png",
            save_roc_plot =f"train_output/{params['train_output_dir']}/{params['train_output_dir']}_roc_callback.png"
        )

    # After every epoch store the model with the best validation performance
    model_cp_val_loss_cb = ModelCheckpoint(
        filepath       = f"models/{params['train_output_dir']}/best-direct-fold{params['fold_num']}_val_loss.h5",
        monitor        = 'val_loss', 
        save_best_only = True, 
        mode           = 'min',
        verbose        = VERBOSE
    )

    model_cp_val_froc_cb = ModelCheckpoint(
        filepath       = f"models/{params['train_output_dir']}/best-direct-fold{params['fold_num']}_p_auc_val_froc.h5",
        monitor        = 'val_froc', 
        save_best_only = True, 
        mode           = 'max',
        verbose        = VERBOSE
    )

    # This callback predicts on some validation images after each epoch
    imgs_cb = IntermediateImagesDiag(
        validation_set   = validation_set, 
        prefix           = f"diag_fold{params['fold_num']}",
        train_outdir     = params['train_output_dir'],
        num_images       = 30,
        input_sequences  = [params['input_modality']],
        output_sequences = OUT_SEQUENCES
    )

    csv_log_cb = CSVLogger(f"train_output/{params['train_output_dir']}/train_direct_log_fold{params['fold_num']}.csv")
    
    es_cb_val_loss = EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=ES_PATIENCE,
        min_delta=ES_MIN_DELTA
    )

    # Early stopping on the partial AUC for the FROC curve. That is this value.
    # es_cb_val_froc = EarlyStopping(
    #     monitor   = 'val_froc',
    #     mode      = 'max',
    #     verbose   = 1,
    #     patience  = ES_PATIENCE,
    #     min_delta = ES_MIN_DELTA
    # )

    callbacks = [
        model_cp_val_loss_cb,
        froc_cb,
        model_cp_val_froc_cb,
        imgs_cb,
        es_cb_val_loss,
        csv_log_cb
    ]

    # Train the model we created
    model.fit(
        x                  = train_generator,
        validation_data    = validation_set,
        steps_per_epoch    = len(train_idxs) // BATCH_SIZE, 
        epochs             = EPOCHS,
        callbacks          = callbacks,
        verbose            = VERBOSE
    )

    print_p("  -- DONE --  ")
