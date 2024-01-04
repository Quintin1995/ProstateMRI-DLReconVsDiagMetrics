from tensorflow.keras.callbacks import Callback
import os
import SimpleITK as sitk
from typing import List
import numpy as np

from fastMRI_PCa.utils import print_p


#Tensorflow Callback for exporting nifti predictions after each epoch.
class IntermediateImagesRecon(Callback):

    def __init__(self,
        validation_set,
        prefix: str,
        train_outdir: str,
        input_sequences: List[str],
        output_sequences: List[str],
        num_images=10):

        self.prefix = prefix
        self.num_images = num_images
        self.validation_set = (
            validation_set[0][:num_images, ...],        # input
            validation_set[1][:num_images, ...]         # label
            )
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences

        self.out_dir = f"train_output/{train_outdir}"

        if self.num_images > validation_set[0].shape[0]:
            self.num_images = validation_set[0].shape[0]

        print_p(f"> IntermediateImages: Exporting images and targets to {self.out_dir}")
        for i in range(self.num_images):

            # Write the input to file
            for seq_idx, seq in enumerate(self.input_sequences):
                img_s = sitk.GetImageFromArray(self.validation_set[0][i, ..., seq_idx].T)
                img_s.SetSpacing([0.5, 0.5, 3.0])
                sitk.WriteImage(img_s, f"{self.out_dir}/{prefix}_{i:03d}_{seq}.nii.gz")

            # Write the label to file
            for seq_idx, seq in enumerate(self.output_sequences):
                seg_s = sitk.GetImageFromArray(self.validation_set[1][i, ..., seq_idx].T)
                seg_s.SetSpacing([0.5, 0.5, 3.0])
                sitk.WriteImage(seg_s, f"{self.out_dir}/{prefix}_{i:03d}_{seq}.nii.gz")


    def on_epoch_end(self, epoch, logs={}):
        print(f"Writing predictions to {self.out_dir}")
        
        # Predict on the validation_set
        self.predictions = self.model.predict(self.validation_set, batch_size=1)
        
        error_maps = self.predictions - self.validation_set[1]

        # Do predictions for each output sequence.
        for i in range(self.num_images):
            for seq in self.input_sequences:
                pred_s = sitk.GetImageFromArray(self.predictions[i, ..., 0].T)
                pred_s.SetSpacing([0.5, 0.5, 3.0])
                sitk.WriteImage(pred_s, f"{self.out_dir}/{self.prefix}_{i:03d}_{seq}_pred.nii.gz")
            
            # Calculate the error map for each sequence.
            for seq in self.input_sequences:
                error_map_s = sitk.GetImageFromArray(error_maps[i, ..., 0].T)
                error_map_s.SetSpacing([0.5, 0.5, 3.0])
                sitk.WriteImage(error_map_s, f"{self.out_dir}/{self.prefix}_{i:03d}_{seq}_error_map.nii.gz")


# Tensorflow Callback for exporting nifti predictions after each epoch for a
# diagnostic lesion segmentations model.
class IntermediateImagesDiag(Callback):

    def __init__(self,
        validation_set,
        prefix: str,
        train_outdir: str,
        input_sequences: List[str],
        output_sequences: List[str],
        num_images=10):

        self.prefix = prefix
        self.num_images = num_images
        self.validation_set = (
            validation_set[0][:num_images, ...],
            validation_set[1][:num_images, ...])
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences

        self.out_dir = f"train_output/{train_outdir}"

        if self.num_images > validation_set[0].shape[0]:
            self.num_images = validation_set[0].shape[0]

        print_p(f"> IntermediateImages: Exporting images and targets to {self.out_dir}")
        for i in range(self.num_images):

            # Write the input to file
            for seq_idx, seq in enumerate(self.input_sequences):
                img_s = sitk.GetImageFromArray(self.validation_set[0][i, ..., seq_idx].T)
                img_s.SetSpacing([0.5, 0.5, 3.0])
                sitk.WriteImage(img_s, f"{self.out_dir}/{prefix}_{i:03d}_inp_{seq}.nii.gz")

            # Write the label to file
            for seq_idx, seq in enumerate(self.output_sequences):
                seg_s = sitk.GetImageFromArray(self.validation_set[1][i, ..., seq_idx].T)
                seg_s.SetSpacing([0.5, 0.5, 3.0])
                sitk.WriteImage(seg_s, f"{self.out_dir}/{prefix}_{i:03d}_out_{seq}.nii.gz")


    def on_epoch_end(self, epoch, logs={}):
        print(f"Writing predictions to {self.out_dir}")
        
        # Predict on the validation_set
        self.predictions = self.model.predict(self.validation_set, batch_size=1)

        # Do predictions for each output sequence.
        for i in range(self.num_images):
            for seq_idx, seq in enumerate(self.output_sequences):
                pred_s = sitk.GetImageFromArray(np.squeeze(self.predictions[i, ...]).T)
                pred_s.SetSpacing([0.5, 0.5, 3.0])
                sitk.WriteImage(pred_s, f"{self.out_dir}/{self.prefix}_{i:03d}_pred_{seq}.nii.gz")