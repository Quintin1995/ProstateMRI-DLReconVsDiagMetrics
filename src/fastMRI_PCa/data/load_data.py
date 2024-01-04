import SimpleITK as sitk
import numpy as np
from typing import Tuple, List, Optional, Iterator

from fastMRI_PCa.utils import print_p
from scipy import ndimage


def load_nifti(path: str) -> np.ndarray:
    """ Load a single nifti image to numpy format """
    print("Loading nifti...    file: {0}".format(path))
    img_s = sitk.ReadImage(path, sitk.sitkFloat32)
    img_n = sitk.GetArrayFromImage(img_s)
    return img_n.transpose([2,1,0])


def get_generator(
    shape: Tuple,
    X: np.ndarray,
    Y: np.ndarray,
    input_sequences: List[str],
    output_sequences: List[str],
    batch_size: Optional[int] = 5, 
    indexes: Optional[List[int]] = None, 
    shuffle: bool = False, 
    augmentation = True
    ) -> Iterator[Tuple[dict, dict]]:
    """
    Returns a (training) generator for use with model.fit().
    augmentation: Apply augmentation or not (bool).
    """

    num_rows = X.shape[0]

    if indexes == None:
        indexes = list(range(num_rows))

    if type(indexes) == int:
        indexes = list(range(indexes))

    if batch_size == None:
        batch_size = len(indexes)  

    idx = 0

    # Prepare empty batch placeholder with named inputs and outputs
    input_batch = np.zeros((batch_size,) + shape + (len(input_sequences),))   # T2, adc, dwi
    output_batch = np.zeros((batch_size,) + shape + (len(output_sequences),)) # seg

    # Loop infinitely to keep generating batches
    while True:
        # Prepare each observation in a batch
        for img_idx in range(batch_size):
            # Shuffle the order of images if all indexes have been seen
            if idx == 0 and shuffle:
                np.random.shuffle(indexes)

            current_index = indexes[idx]

            # Insert the augmented images into the input batch
            img_crop, lab_crop = (
                np.zeros(shape + (len(input_sequences),)),
                np.zeros(shape + (len(output_sequences),))
                )
            
            for seq_idx, seq in enumerate(input_sequences):
                img_crop[:, :, :, seq_idx] = X[current_index, :, :, :, seq_idx]

            for seq_idx, seq in enumerate(output_sequences):
                lab_crop[:, :, :, seq_idx] = Y[current_index, :, :, :, seq_idx]
            
            if augmentation:
                img_crop, lab_crop = augment_XY(img_crop, lab_crop)

            input_batch[img_idx]  = img_crop
            output_batch[img_idx] = lab_crop

            # Increase the current index and modulo by the number of rows
            # so that we stay within bounds
            idx = (idx + 1) % len(indexes)
                
        yield input_batch, output_batch


def augment_noise(img, multiplier):
    noise = np.random.standard_normal(img.shape) * multiplier
    return img + noise


def augment_rotate(img, lab, angle):
    img = ndimage.rotate(img, angle, reshape=False)
    lab = ndimage.rotate(lab, angle, reshape=False, order=0)
    return img, lab


def augment_flip(img, lab):
    """Flip along X axis"""
    img = np.flip(img, axis=1)
    lab = np.flip(lab, axis=1)
    return img, lab


def augment_XY(img, lab, 
    noise_chance = 0.3,
    noise_mult_max = 0.001,
    rotate_chance = 0.2,
    rotate_max_angle = 30,
    flip_chance = 0.5
    ):
    if np.random.uniform() < noise_chance:
        img = augment_noise(img, np.random.uniform(0., noise_mult_max))
        
    if np.random.uniform() < rotate_chance:
        img, lab = augment_rotate(img, lab, np.random.uniform(0-rotate_max_angle, rotate_max_angle))
    if np.random.uniform() < flip_chance:
        img, lab = augment_flip(img, lab)
    return img, lab