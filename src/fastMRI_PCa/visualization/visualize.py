import matplotlib.pyplot as plt
import numpy as np
# from scipy import fftpack
# import SimpleITK as sitk
from fastMRI_PCa.utils import print_p
# from typing import List
# import os


FONTSIZE = 8


# def save_kspace_mask(numpy_2d: np.ndarray,
#     figdir: str,
#     filename: str) -> None:
#     """ Saves binary mask for k space to png file
    
#     Parameters:
#     `numpy_2d (np.ndarray)`: 2d numpy array (width, height) binary file. zeros and ones.
#     `figdir (str)`: figure dicectory where the file will be saved.
#     `filename (str)`: name of the saved slice in the figdir.
#     """

#     fig = plt.figure()
#     plt.imshow(numpy_2d, cmap="gray")
#     plt.title(f'kspace_mask', size=FONTSIZE)
#     plt.axis('on')
#     path = f"{figdir}/{filename}.png"
#     fig.savefig(path, dpi=300, bbox_inches='tight')
#     print_p(f">-Wrote kspace to file: {path}")
#     plt.close()


# def save_slice_3d(numpy_3d: np.ndarray,
#                   slice_num: int,
#                   figdir: str,
#                   filename: str) -> None:
#     """ Saves the a slice of the give volume to a file (.png).
    
#     Parameters:
#     `numpy_3d (np.ndarray)`: 3d numpy array (width, height, depth)
#     `slice_num (int)`: slice number to be saved to file
#     `figdir (str)`: figure dicectory where the file will be saved.
#     `filename (str)`: name of the saved slice in the figdir.
#     """

#     fig = plt.figure()
#     plt.imshow(numpy_3d[:, :, slice_num].T, cmap="gray")
#     plt.title(f'Slice {slice_num}', size=FONTSIZE)
#     plt.axis('off')
#     path = f"{figdir}{filename}"
#     fig.savefig(path, dpi=300, bbox_inches='tight')
#     print_p(f">-Wrote slice {slice_num} to file: {path}")
#     plt.close()


# # def save_slice_2d(
# #     numpy_2d: np.ndarray,
# #     figdir: str,
# #     filename: str
# # ) -> None:
# #     """ Saves the a slice of the give volume to a file (.png).
    
# #     Parameters:
# #     `numpy_2d (np.ndarray)`: 2d numpy array (width, height)
# #     `figdir (str)`: figure dicectory where the file will be saved.
# #     `filename (str)`: name of the saved slice in the figdir.
# #     """

# #     fig = plt.figure()
# #     plt.imshow(numpy_2d, cmap="gray")
# #     plt.title(f'Slice', size=FONTSIZE)
# #     plt.axis('on')
# #     path = f"{figdir}/{filename}.png"
# #     fig.savefig(path, dpi=300, bbox_inches='tight')
# #     print_p(f">-Wrote slice to file: {path}")
# #     plt.close()


def save_array2d_to_image(
    numpy_2d: np.ndarray,
    title: str,
    figdir: str,
    filename: str,
    show = False) -> None:
    """ Saves the a array to a file in greyscale (.png).
    
    Parameters:
    `numpy_2d (np.ndarray)`: 2d numpy array (width, height)
    `title (str)`: title of the image
    `figdir (str)`: figure dicectory where the file will be saved.
    `filename (str)`: name of the saved array in the figdir.
    `show (bool)`: Prompts the user directly with the image in a window
    """

    fig = plt.figure()
    plt.imshow(numpy_2d, cmap="gray")
    plt.title(title, size=FONTSIZE)
    plt.axis('off')
    path = f"{figdir}{filename}"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print_p(f">-Wrote array({numpy_2d.shape[0]}, {numpy_2d.shape[1]}) to file: {path}")

    if show:
        plt.show()

    plt.close()


# def save_kspace(
#     kspace: np.ndarray,
#     figdir: str,
#     filename: str,
#     do_shift=False) -> None:
#     """ Saves a k-space to file.
#         K-space will undergo a log transformation for better visualization.

#     Parameters:
#     `kspace (np.ndarray): Kspace slice
#     `figdir (str)`: Figure dicectory where the file will be saved.
#     `filename (str)`: Name of the saved slice in the figdir.
#     `do_shift (bool)`: Whether to calculate the magnitude and shift the zero
#         frequency
#     """

#     if np.iscomplex(kspace).any():
#         kspace = np.abs(kspace)

#     fig = plt.figure()
#     M, N = kspace.shape

#     if do_shift:
#         # Apply Transformation to k-space
#         kspace_mag = np.abs(kspace)
#         kspace_mag = fftpack.fftshift(kspace_mag)
#         plt.imshow(np.log(1 + kspace_mag), cmap='gray',
#                 extent=(-N // 2, N // 2, -M // 2, M // 2))
#     else:
#         plt.imshow(kspace, cmap='gray',
#                 extent=(-N // 2, N // 2, -M // 2, M // 2))
    
#     plt.title('Spectrum Magnitude', size=FONTSIZE)
#     plt.axis('on')
#     path = f"{figdir}/{filename}.png"
#     fig.savefig(path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print_p(f"Wrote kspace to file: {path}")
    

# def write_array2nifti_t2(
#     mri_vol: np.ndarray,
#     figdir: str,
#     filename: str,
#     target_space: List[float] = [0.5, 0.5, 3.0]
# ) -> None:
#     """ Writes a nifti (T2) to file based on a 3D numpy array. Target spacing is
#     set if it is not given.
    
#     `mri_vol (np.ndarray)`: Mri volume. 3D numpy array
#     `figdir (str)`: Figure dicectory where the file will be saved.
#     `filename (str)`: Name of the saved nifti in the figdir.
#     `target_space (List[int])`: Voxel size of mri volume
#     """
#     mri_sitk = sitk.GetImageFromArray(mri_vol.squeeze().T)
#     mri_sitk.SetSpacing(target_space)
#     path = os.path.join(figdir, filename)
#     sitk.WriteImage(mri_sitk, path)
#     print_p(f">-Wrote nifti to path: {path}")


# def write_sitk2nifti_t2(
#     mri_sitk: sitk,
#     figdir: str,
#     filename: str,
#     target_space = None) -> None:
#     """ Writes a nifti (T2) to file based on a 3D numpy array. Target spacing is
#     set if it is not given.
    
#     `mri_sitk (sitk.Image)`: Mri volume.
#     `figdir (str)`: Figure dicectory where the file will be saved.
#     `filename (str)`: Name of the saved nifti in the figdir.
#     `target_space (List[int])`: Voxel size of mri volume
#     """

#     if target_space == None:s
#         mri_sitk.SetSpacing([0.5, 0.5, 3.0])
#     else:
#         mri_sitk.SetSpacing(target_space)
#     path = f"{figdir}{filename}.nii.gz"
#     sitk.WriteImage(mri_sitk, path)
#     print_p(f">-Wrote nifti to path: {path}")


# def write_array2nifti_dwi(
#     mri_vol: np.ndarray,
#     figdir: str,
#     filename: str,
#     target_space = None) -> None:
#     """ Writes a nifti (DWI) to file based on a 3D numpy array. Target spacing is
#     set if it is not given.
    
#     `mri_vol (np.ndarray)`: Mri volume.
#     `figdir (str)`: Figure dicectory where the file will be saved.
#     `filename (str)`: Name of the saved nifti in the figdir.
#     `target_space (List[int])`: Voxel size of mri volume
#     """

#     mri_sitk = sitk.GetImageFromArray(mri_vol.T)

#     if target_space == None:
#         mri_sitk.SetSpacing([1.37, 1.37, 3.0])
#     else:
#         mri_sitk.SetSpacing(target_space)
#     path = f"{figdir}{filename}.nii.gz"
#     sitk.WriteImage(mri_sitk, path)
#     print_p(f">-Wrote mri to path: {path}")



# def write_np2nifti(img_np, filename):
#     sitk.WriteImage(sitk.GetImageFromArray(img_np.squeeze().T), filename)
#     print(f">-Wrote image to: {filename}")
