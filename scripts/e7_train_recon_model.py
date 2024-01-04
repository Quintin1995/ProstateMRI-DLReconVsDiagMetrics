import argparse
from datetime import date
import h5py
import glob
import numpy as np
import SimpleITK as sitk
import sys
from sklearn.model_selection import KFold

from fastMRI_PCa.utils import create_dirs_if_not_exists
from fastMRI_PCa.utils import print_p, dump_dict_to_yaml
from fastMRI_PCa.data import get_generator, IntermediateImagesRecon
from fastMRI_PCa.models import ssim_loss, ssim_metric, psnr_metric


################################  README  ######################################
# OLD - This script will undersample T2 images and train a reconstruction model on it.
# First the data needs to be made with a make_data script. This script will read
# a storage file (.h5). This storage file contains undersampled T2 data.


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

    args = parser.parse_args()
    return args


################################################################################


STORAGE = "data/interim/3_rand_exp_dist/storage_t2_samp25.h5"
TRAIN_OUTDIR = "9_improved_mask_ssim_long2x"
MODELS_OUTDIR = TRAIN_OUTDIR         # Same name, I want to reference them seperately for clarity.
OPTIMIZER = "adam"
LAST_ACTIVATION = 'sigmoid'
LOSS = "SSIM"
SEED = 3478
EPOCHS = 1000

INP_SEQUENCES = ["t2"]
OUT_SEQUENCES = ["t2_label"]

args = parse_input_args()

batch_size = 10 if args.is_job else 1  # if job, then batch_size=10
verbose = 2 if args.is_job else 1      # if job, then less text in slurm

create_dirs_if_not_exists([f"models/{MODELS_OUTDIR}", f"train_output/{TRAIN_OUTDIR}"])

print_p(f"Fold number: {args.fold_num+1} of {args.num_folds}")

params = {
    "is_job": args.is_job,
    "storage": STORAGE,
    "train_output_dir": TRAIN_OUTDIR,
    "models_outdir": MODELS_OUTDIR,
    "input_sequences": INP_SEQUENCES,
    "output_sequences": OUT_SEQUENCES,
    "batch_size": batch_size,
    "verbose": verbose,
    "optimizer": OPTIMIZER,
    "seed": SEED,
    "number_folds": args.num_folds,
    "epochs": EPOCHS,
    "last_activation": LAST_ACTIVATION,
    "loss": LOSS,
    "datetime": date.today().strftime("%Y-%m-%d")}
dump_dict_to_yaml(params, f"train_output/{TRAIN_OUTDIR}", filename=f"params")

storage = h5py.File(STORAGE, 'r')

# Get the number of observations in the storage file
num_obs = len(storage['t2'])
print_p(f"Number of observations: {num_obs}")

# Randomly split the data into a train (80%) / validation (10%) / test (10%)
all_idxs = list(range(num_obs))

kfold = KFold(args.num_folds, shuffle=True, random_state=SEED)
train_idxs, valid_idxs = list(kfold.split(all_idxs))[args.fold_num]
train_idxs = list(train_idxs)
valid_idxs = list(valid_idxs)

print_p(f"Dataset division:\n\t- Train: {len(train_idxs)} = {len(train_idxs)/len(all_idxs)*100}%")
print_p(f"\t- Valid: {len(valid_idxs)} = {len(valid_idxs)/len(all_idxs)*100}%")
print_p(f"\t- Test: {0}")
print_p(f"Validation indexes: {valid_idxs}")

crop_size = storage['t2'][0].shape

train_generator = get_generator(
    batch_size=batch_size,
    shape=crop_size,
    storage=storage,
    input_sequences=INP_SEQUENCES,
    output_sequences=OUT_SEQUENCES,
    indexes=train_idxs,
    shuffle=True)

train_set = next(train_generator)
print(f"train set: {train_set[0].shape}")

validation_generator = get_generator(
    batch_size=None,
    shape=crop_size,
    storage=storage,
    input_sequences=INP_SEQUENCES,
    output_sequences=OUT_SEQUENCES,
    indexes=valid_idxs,
    shuffle=True,
    augmentation=False)
    
validation_set = next(validation_generator)
print(f"Validation set: {validation_set[0].shape}")


from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv3D, concatenate
from tensorflow.keras.layers import MaxPooling3D, UpSampling3D
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow as tf


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

# After every epoch store the model with the best validation performance
model_cp = ModelCheckpoint(f"models/{MODELS_OUTDIR}/best-direct-fold{args.fold_num}.h5",
    monitor = 'val_loss', 
    save_best_only=True, 
    mode='min',
    verbose=verbose)

# This callback predicts on a number of images for the test set after each epoch
# and shows a few slices in PNG files in the "output/" folder
images_callback = IntermediateImagesRecon(validation_set, 
                                          prefix=f"fold{args.fold_num}",
                                          train_outdir=TRAIN_OUTDIR,
                                          num_images=10,
                                          input_sequences=INP_SEQUENCES,
                                          output_sequences=OUT_SEQUENCES)

# This callback produces a log file of the training and validation metrics at 
# each epoch
csv_log_callback = CSVLogger(f"train_output/{TRAIN_OUTDIR}/train_direct_log_fold{args.fold_num}.csv")

# Train the model we created
dnn.fit(train_generator,
        validation_data    = validation_set,
        steps_per_epoch    = len(train_idxs) // batch_size, 
        epochs             = EPOCHS,
        callbacks          = [model_cp, images_callback, csv_log_callback],
        verbose            = verbose,
        )

print("-- Done --")