import argparse
import SimpleITK as sitk
import numpy as np
from typing import Iterator, List, Optional, Tuple
import random
import os
from shutil import copyfile

from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.compat.v1 import disable_eager_execution

import umcglib.images as im
from umcglib.augment import augment, random_crop
from umcglib.callbacks import IntermediateImages, ConditionalStopping
from umcglib.models.unet import build_dual_attention_unet, build_unet
from umcglib.losses import weighted_binary_cross_entropy
from umcglib.utils import print_, apply_parallel, read_yaml_to_dict, set_gpu
from umcglib.utils import get_random_str
from umcglib.plotting import save_array2d_to_image

from fastMRI_PCa.utils.k_space_masks import get_rand_exp_decay_mask


################################  README  ######################################
# NEW -This script will load ONE of the following sequences: (undersampled) T2W,
# The data will be used to train a diagnostic network where lesions
# are segmented for the patient. Model = Dual Attention U-net from Anindo.
# Goal: Train a diagnostic U-Net on ONE input sequence type.


def parse_input_args():

    help = """This script will load the following data: undersampled T2w images.
              The data will be used to train a diagnostic network where lesions
              are segmented for the patient. Goal: Train a diagnostic U-Net.
              Trains a model on ONE input sequence. """

    parser = argparse.ArgumentParser(description=help)

    parser.add_argument(
        '-c',
        '--config_fpath',
        type=str,
        required=True,
        help='Path to config file with hyper parameters for a reconstruction training session.',
    )

    args = parser.parse_args()

    print(f"\nAll input parameters: {args}\n")
    return args


def get_generator(
    input_images: List[np.ndarray], 
    output_images: List[np.ndarray],
    batch_size: Optional[int] = 5, 
    shuffle: bool = False, 
    augmentation = True,
    shape = None,
    crop_method = None,
    normalization = None,
    rotation_freq = 0.1,
    tilt_freq = 0.1,
    noise_freq = 0.3, 
    noise_mult = 1e-3,
    mirror_freq = 0.5
    ) -> Iterator[Tuple[dict, dict]]:
    """
    Returns a (training) generator for use with model.fit().

    Parameters:
    input_modalities: List of modalty names to include.
    output_modalities: Names of the target modalities.
    batch_size: Number of images per batch (default: all).
    indexes: Only use the specified image indexes.
    shuffle: Shuffle the lists of indexes once at the beginning.
    augmentation: Apply augmentation or not (bool).
    """

    num_rows = len(input_images)
    indexes = list(range(num_rows))

    if batch_size == None:
        batch_size = len(indexes)  

    idx = 0

    # Prepare empty batch placeholder with named inputs and outputs
    input_batch = np.zeros((batch_size,) + shape + (1,))
    output_batch = np.zeros((batch_size,) + shape + (1,))

    # Loop infinitely to keep generating batches
    while True:
        # Prepare each observation in a batch
        for batch_idx in range(batch_size):
            
            # Shuffle the order of images if all indexes have been seen
            if idx == 0 and shuffle:
                np.random.shuffle(indexes)

            current_index = indexes[idx]

            img_crop = input_images[current_index]
            seg_crop = output_images[current_index]
            
            if crop_method == "random":
                # Crop img and seg simultaneously so we get the same crop
                img_crop, seg_crop = random_crop(
                    img   = img_crop.squeeze(), 
                    label = seg_crop,
                    shape = shape
                )
            
            if crop_method == "center":
                img_crop = im.center_crop_n(img_crop, shape)
                seg_crop = im.center_crop_n(seg_crop, shape)
            
            if normalization == "znorm":
                for c in range(img_crop.shape[-1]):
                    img_crop[..., c] -= np.mean(img_crop[..., c])
                    img_crop[..., c] /= np.std(img_crop[..., c])

            if augmentation:
                img_crop, seg_crop = augment(img_crop, seg_crop,
                noise_chance = noise_freq,
                noise_mult_max = noise_mult,
                rotate_chance = rotation_freq,
                tilt_chance = tilt_freq,
                mirror_chance = mirror_freq)
            
            input_batch[batch_idx] = img_crop
            output_batch[batch_idx] = seg_crop

            # Increase the current index and modulo by the number of rows
            #  so that we stay within bounds
            idx = (idx + 1) % len(indexes)
                
        yield input_batch, output_batch


def preprocess(
    t2,
    window_size,
    spacing         = (0.5, 0.5, 3.),
    crop            = True, 
    to_numpy        = True,
    seg             = None,
    norm            = "znorm",
    acceleration    = None,
    centre_sampling = None,
    seed            = 12345
):
    save_chance = 0.10 if not DEBUG else 0.25
    cur_chance = random.random()
    uniq_str = get_random_str(n_chars=7)

    if cur_chance < save_chance and DEBUG:
        acc = int(acceleration)
        fname_debug = f"{UNDERSAMPLE_DIR}{uniq_str}_r{acc}n_before_u.nii.gz"
        sitk.WriteImage(t2, fname_debug)
        print_(f">Wrote to: {fname_debug}")

    if acceleration != None: 
        mask = get_rand_exp_decay_mask(
            width           = t2.GetSize()[0],
            height          = t2.GetSize()[1],
            sampling        = 1.0/acceleration,
            centre_sampling = centre_sampling,
            seed            = seed,
            verbatim        = False
        )
        t2 = im.undersample_kspace(t2, mask)

        if cur_chance < save_chance and DEBUG:
            save_array2d_to_image(mask, "mask", UNDERSAMPLE_DIR, f"{uniq_str}_mask.png")
            fname_debug = f"{UNDERSAMPLE_DIR}{uniq_str}_r{acc}n_before_prep_u.nii.gz"
            sitk.WriteImage(t2, fname_debug)
            print_(f">Wrote to: {fname_debug}")

    # Preprocess the ADC image, then resample the rest to it
    t2 = im.resample(
        image       = t2, 
        min_shape   = window_size, 
        method      = sitk.sitkNearestNeighbor, 
        new_spacing = spacing,
    )

    if crop:
        t2 = im.center_crop(t2, window_size)

    if seg is not None:
        seg = im.resample_to_reference(seg, t2, sitk.sitkNearestNeighbor)

    if cur_chance < save_chance:
        t2_s = t2
        seg_s = seg

    # Return the SITK preprocessed images if requested
    if not to_numpy: 
        return t2, seg

    # Extract numpy arrays
    t2  = sitk.GetArrayFromImage(t2).T

    # Stack the inputs, add new axis to seg
    img_n = t2[..., np.newaxis]
    
    # Z-Normalize at crop level
    if norm == "znorm":
        img_n -= np.mean(img_n)
        img_n /= np.std(img_n)
    
    if norm == "rescale_0_1":
        img_n = (1.0*(img_n - np.min(img_n))/np.ptp(img_n))
    
    # Extract batch for the segmentation if provided
    if seg is not None: 
        seg = sitk.GetArrayFromImage(seg).T
        seg = (seg[..., None] > 0.5) * 1.

    if cur_chance < save_chance and DEBUG:
        fname_debug = f"{UNDERSAMPLE_DIR}{uniq_str}_r{acc}n_seg.nii.gz"
        temp_seg = sitk.GetImageFromArray(np.squeeze(seg).T)
        temp_seg.CopyInformation(seg_s)
        temp_seg.SetSpacing(spacing)
        sitk.WriteImage(temp_seg, fname_debug)
        print_(f">Wrote to: {fname_debug}")

        fname_debug = f"{UNDERSAMPLE_DIR}{uniq_str}_r{acc}n_after_prep.nii.gz"
        temp_t2 = sitk.GetImageFromArray(np.squeeze(img_n).T)
        temp_t2.CopyInformation(t2_s)
        temp_t2.SetSpacing(spacing)
        sitk.WriteImage(temp_t2, fname_debug)
        print_(f">Wrote to: {fname_debug}")

    return img_n, seg


def read_and_preprocess(
    paths, 
    window_size     = None, 
    spacing         = (0.5, 0.5, 3.), 
    crop            = True,
    to_numpy        = True,
    norm            = "znorm",
    acceleration    = None,
    centre_sampling = None,
    seed            = 12345,
):
    # Explode the inputs
    if len(paths) == 1:
        t2_path = paths[0]
        seg_path = None
        seg = None
    if len(paths) == 2:
        t2_path, seg_path = paths

    # Read each image
    t2 = sitk.ReadImage(t2_path, sitk.sitkFloat32)
    if seg_path is not None:
        seg = sitk.ReadImage(seg_path, sitk.sitkFloat32)

    return preprocess(t2,
                      seg=seg,
                      window_size=window_size,
                      spacing=spacing,
                      crop=crop,
                      to_numpy=to_numpy,
                      norm=norm,
                      acceleration=acceleration,
                      centre_sampling=centre_sampling,
                      seed=seed)


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


################################################################################


def train(
    train_path_list: str,
    label_path_list: str,
    t2_recon_dir: str,
    seed: int = None,
    fold_num: int = 0,
    num_folds: int = 5,
    target_space: List = [0.5, 0.5, None],
    indexes_path: str = "data/path_lists/pirads_4plus/train_val_test_idxs.yml",
    train_set_key: str = "train_set0",
    window_size: tuple = (160, 160, 20),
    batch_size: int = 13,
    max_epochs: int = 5000,
    n_workers: int = 10,
    do_early_stop: bool = True,
    early_stop_pat: int = 5,
    early_stop_var: str = "val_loss",
    early_stop_mode: str = "min",
    conditional_stopping_threshold=None,
    conditional_stopping_epochs=None,
    conditional_stopping_var=None,
    conditional_stopping_mode=None,
    crop_method: str = "center",
    normalization: str = "znorm",
    learning_rate: float = 0.0004,
    loss: str = "binary_crossentropy",
    optimizer: str = "adam",
    rotation_freq: float = 0.1,
    tilt_freq: float = 0.0,
    noise_freq: float = 0.3,
    noise_mult: float = 1e-3,
    mirror_freq: float = 0.5,
    validation_interval: int = 1,
    unet_type: str = "dual_attention",
    l2_regularization: float = 0.0001,
    instance_norm: bool = False,
    verbose: int = 2,
    num_validation: int = 50,
    acceleration: float = None,
    centre_sampling: float = None,
    sampling_window: tuple = None,
    **kwargs,
):

    # Match images and segmentations
    t2_files = [l.strip() for l in open(train_path_list)]
    seg_files = [l.strip() for l in open(label_path_list)]
    all_idxs = read_yaml_to_dict(indexes_path)[train_set_key]

    if DEBUG:
        percent_load        = 0.04
        all_idxs            = all_idxs[:int(percent_load * len(all_idxs))]
        validation_interval = 2
        n_workers           = 4
        batch_size          = 2
        verbose             = 1
        max_epochs          = 100
        set_gpu(gpu_idx=1)

    # # If reconstruction is done with a Reconstruction model, then we need to
    # # gather its acceleration and central sampling rate.
    # if configs['t2_recon_dir'] != None:
    #     print("if you want t2 reconstruction with a unet. implement it.")
    #     quit()
    #     configs['acceleration'] = read_yaml_to_dict(os.path.join(configs['t2_recon_dir'], configs['foldnum']))
    #     acceleration            = read_yaml_to_dict(f"train_output/{T2_RECON_DIR}/params.yml")['acceleration']
    #     centre_sampling         = read_yaml_to_dict(f"train_output/{T2_RECON_DIR}/params.yml")['centre_sampling']
    #     sampling_window         = read_yaml_to_dict(f"train_output/{T2_RECON_DIR}/params.yml")['target_shape']
    #     normalization           = read_yaml_to_dict(f"train_output/{T2_RECON_DIR}/params.yml")['norm']

    train_idxs, valid_idxs = get_train_val_idxs_from_kfold(all_idxs, fold_num, num_folds, seed)

    # Zip the list and read + preprocess in parallel
    print_(f"> Reading images.. {len(t2_files)}")
    combined_prepped = []

    if DEBUG:
        t2_files = list(np.asarray(t2_files)[train_idxs]) + list(np.asarray(t2_files)[valid_idxs])
        seg_files = list(np.asarray(seg_files)[train_idxs]) + list(np.asarray(seg_files)[valid_idxs])

    num_read_splits = 10 if not DEBUG else 5
    num_read_at_once = len(seg_files) // num_read_splits + 1
    combined_list = list(zip(t2_files, seg_files))
    combined_prepped = []
    for i in range(num_read_splits):
        _from = i * num_read_at_once
        _to = (i + 1) * num_read_at_once
        print_("From idx:", _from, " To idx:", _to)
        cur_paths = combined_list[_from:_to]
        combined_prepped += apply_parallel(
            cur_paths,
            read_and_preprocess,
            n_workers,
            window_size     = window_size,
            crop            = True,
            spacing         = target_space,
            norm            = normalization,
            acceleration    = acceleration,
            centre_sampling = centre_sampling,
            seed            = seed
        )
        
    prepped_imgs, prepped_segs = [], []
    for i, s in combined_prepped:
        prepped_imgs.append(i)
        prepped_segs.append(s)

    print(f">> lenght prepped imgs = {len(prepped_imgs)}")
    print(f">> lenght prepped segs = {len(prepped_segs)}")

    # Predict with the reconstruction model if chosen.
    if t2_recon_dir != None:
        for i in range(20):
            fname_debug = f"{UNDERSAMPLE_DIR}{i}_r{int(acceleration)}n_before_DLrecon.nii.gz"
            dl_pred_recon_s = sitk.GetImageFromArray(np.squeeze(prepped_imgs[i]).T)
            sitk.WriteImage(dl_pred_recon_s, fname_debug)
            print_(f"Wrote to: {fname_debug}")

        prepped_imgs = recon_model_predict(prepped_imgs, t2_recon_dir, fold_num)
        
        for i in range(20):
            fname_debug = f"{UNDERSAMPLE_DIR}{i}_r{int(acceleration)}n_after_DLrecon.nii.gz"
            dl_pred_recon_s = sitk.GetImageFromArray(np.squeeze(prepped_imgs[i]).T)
            sitk.WriteImage(dl_pred_recon_s, fname_debug)
            print_(f"Wrote to: {fname_debug}")
    
    # Split data
    # all_indexes = list(range(len(prepped_imgs)))
    # if not DEBUG:   # A real job gets indexes predefined form a file 80%tr/10%va/10%te
    #     all_indexes = read_yaml_to_dict(IDXS_PATH)[TRAINSETKEY]

    # Perform K-fold over the training data
    # kfold = KFold(num_folds, shuffle=True, random_state=seed)
    # train_idxs, valid_idxs = list(kfold.split(all_idxs))[fold_num]

    if DEBUG:
        train_idxs = list(range(len(train_idxs)))
        valid_idxs = list(range(len(valid_idxs)))

    train_imgs = [prepped_imgs[i] for i in train_idxs]
    train_segs = [prepped_segs[i] for i in train_idxs]
    valid_imgs = [prepped_imgs[i] for i in valid_idxs]
    valid_segs = [prepped_segs[i] for i in valid_idxs]

    num_train = len(train_imgs)
    num_valid = len(valid_imgs)
    print("TRN: ", num_train)
    print("VAL: ", num_valid)

    if loss == "weighted_binary_cross_entropy":
        weight_for_0 = 0.05
        weight_for_1 = 0.95
        loss = weighted_binary_cross_entropy({0: weight_for_0, 1: weight_for_1})

    if optimizer == "adam":
        optimizer = Adam(learning_rate)

    # Get the number of observations
    num_train_obs = len(train_imgs)

    # Peek first image to get shape
    num_inputs = train_imgs[0].shape[-1]

    # Create a data generator that the model can train on
    train_generator = get_generator(
        batch_size    = batch_size,
        shape         = window_size,
        input_images  = train_imgs,
        output_images = train_segs,
        shuffle       = True,
        crop_method   = crop_method,
        normalization = normalization,
        augmentation  = True,
        rotation_freq = rotation_freq,
        tilt_freq     = tilt_freq,
        noise_freq    = noise_freq,
        noise_mult    = noise_mult,
        mirror_freq   = mirror_freq,
    )

    sample_X, sample_Y = next(train_generator)
    print("> Exporting training samples.")
    for i in range(batch_size):
        X, Y = sample_X[i], sample_Y[i]
        for in_channel in range(X.shape[-1]):
            img_n = X[..., in_channel]
            im.to_sitk(
                img_n,
                save_as=os.path.join(TRAIN_DIR, "samples", f"in_{i}_{in_channel}.nii.gz"),
            )
        for out_channel in range(Y.shape[-1]):
            img_n = Y[..., out_channel]
            im.to_sitk(
                img_n,
                save_as=os.path.join(TRAIN_DIR, "samples", f"out_{i}_{in_channel}.nii.gz"),
            )

    # Create a validation set
    validation_generator = get_generator(
        shape         = window_size,
        input_images  = valid_imgs,
        output_images = valid_segs,
        shuffle       = False,
        crop_method   = "center",
        normalization = normalization,
        augmentation  = False,
        batch_size    = len(valid_imgs) if not DEBUG else 2,
    )
    validation_set = next(validation_generator)

    # Create the model and show summary
    if unet_type == "simple":
        dnn = build_unet(
            window_size       = window_size + (num_inputs,),
            num_classes       = 1,
            l2_regularization = l2_regularization,
            instance_norm     = instance_norm,
            final_activation  = "sigmoid",
        )
    else:
        dnn = build_dual_attention_unet(
            input_shape       = window_size + (num_inputs,),
            num_classes       = 1,
            l2_regularization = l2_regularization,
            instance_norm     = instance_norm,
            final_activation  = "sigmoid",
        )

    dnn.summary(line_length=160)
    dnn.compile(optimizer=optimizer, loss=loss, metrics=[AUC()])
    callbacks = []

    # This callback predicts on a number of images for the test set after each epoch
    # and shows a few slices in PNG files in the "output/" folder
    callbacks += [
        IntermediateImages(
            validation_set   = validation_set,
            prefix           = os.path.join(OUTPUT_DIR, f"sample_"),
            num_images       = num_validation,
            binary_threshold = 0.6,
            input_names      = ["t2"],
        )
    ]

    # This callback produces a log file of the training and validation metrics at
    # each epoch
    callbacks += [CSVLogger(os.path.join(LOG_DIR, f"train.csv"))]

    # After every epoch store the model with the best validation performance for
    # each metric that we record
    for metric_name in ["val_loss", "val_auc"]:
        callbacks += [
            ModelCheckpoint(
                os.path.join(MODEL_DIR, f"best_{metric_name}.h5"),
                monitor        = f"{metric_name}",
                save_best_only = True,
                mode           = "min" if "loss" in metric_name else "max",
                verbose        = 1,
            )
        ]

    # Stop training after X epochs without improvement
    if do_early_stop:
        callbacks += [
            EarlyStopping(
                patience = early_stop_pat,
                monitor  = early_stop_var,
                mode     = early_stop_mode,
                verbose  = 1,
            )
        ]

    # Stop training if threshold value is not reached after X epochs
    # Used to prune unsuccessful trials early
    if conditional_stopping_epochs:
        callbacks += [
            ConditionalStopping(
                monitor      = conditional_stopping_var,
                threshold    = conditional_stopping_threshold,
                after_epochs = conditional_stopping_epochs,
                mode         = conditional_stopping_mode,
            )
        ]

    # Train the model we created
    dnn.fit(
        train_generator,
        validation_data = validation_set,
        steps_per_epoch = num_train_obs // batch_size * validation_interval,
        epochs          = max_epochs,
        callbacks       = callbacks,
        verbose         = verbose,
    )
    print_("[I] Completed.")


def recon_model_predict(
    prepped_imgs: List,
    t2_recon_dir: str,
    fold_num: int
):
        print("Predicting with the reconstruction model on understampled T2w images. (validation loss model)")
        t2_model_path = f"models/{t2_recon_dir}/best-direct-fold{fold_num}_val_loss.h5"
        recon_model = load_model(t2_model_path, compile=False)
        recons = np.squeeze(recon_model.predict(np.stack(prepped_imgs, 0), batch_size=BATCH_SIZE))
        prepped_imgs = [recons[mri_idx] for mri_idx in range(recons.shape[0])]
        prepped_imgs = np.expand_dims(prepped_imgs, axis=4)
        recon_model.summary()
        return prepped_imgs


def read_config(config_path, verbatim=True):
    
    configs = read_yaml_to_dict(config_path)
    
    # fix some params
    configs['window_size'] = tuple(configs['window_size'])
    configs['target_shape'] = tuple(configs['target_shape'])
    
    configs['seed'] = None if not configs['use_seed'] else configs['seed']
    
    if verbatim:
        print_("\n>Configs")
        for key in configs:
            value = configs[key]
            print_(f"\t>{key} ({type(value)}):\t\t{value}")
        print_()

    return configs


################################################################################
DEBUG = True
BATCH_SIZE = 1 if DEBUG else 12

if __name__ == "__main__":

    # Read the progrma input arguments
    args = parse_input_args()

    # Read the configurations of the model that needs to be trained
    configs = read_config(args.config_fpath)
    
    # Set train dir
    DEBUG = configs['is_debug']
    TRAIN_DIR = os.path.join(configs['train_dir'], f"fold{configs['fold_num']}")
    if DEBUG:
        TRAIN_DIR = os.path.join(TRAIN_DIR, 'debug')

    # Create experiment folder structure
    print_("> Creating folder structure")
    SEGMENTATION_DIR = os.path.join(TRAIN_DIR, "segmentations/")
    OUTPUT_DIR = os.path.join(TRAIN_DIR, "output/")
    SAMPLE_DIR = os.path.join(TRAIN_DIR, "samples/")
    LOG_DIR = os.path.join(TRAIN_DIR, "logs/")
    MODEL_DIR = os.path.join(TRAIN_DIR, "models/")
    UNDERSAMPLE_DIR = os.path.join(TRAIN_DIR, "undersamples/")
    TEMP_DIR = os.path.join(TRAIN_DIR, "temp/")
    FIGS_DIR = os.path.join(TRAIN_DIR, "figs/")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(SEGMENTATION_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(UNDERSAMPLE_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # copy the config that started the program to the experiment folder.
    dst = os.path.join(TRAIN_DIR, args.config_fpath.split('/')[-1])
    copyfile(args.config_fpath, dst)

    # Disable do not go OOM when training.
    disable_eager_execution()

    train(**configs)

    print(f"-- Done --")