import numpy as np
from typing import List
import SimpleITK as sitk
from SimpleITK import Image
from scipy import fftpack


def undersample(img_s: sitk.Image, mask: np.ndarray):
    """ Undersamples the given sitk image in kspace with the provided mask.
    
    Parameters:
    `img_s (sitk.Image)`: 3D (width, height, depth) image.
    `mask (np.ndarray)`: 2D binary mask to be applied to each slice of img_s.
    `return (sitk.Image)`: Undersampled sitk.Image.
    """

    # Convert to Numpy
    img_n = sitk.GetArrayFromImage(img_s).T

    # Reconstruction has same dimenions as original nifti/MR image
    recon_n = np.zeros(img_n.shape, dtype=np.float32)

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    for slice_idx in range(img_n.shape[2]):

        # Process per slice
        slice = img_n[:, :, slice_idx]

        # Transform to k-space (FFT)
        kspace = fftpack.fftn(slice)

        # Shift kspace (complex part is needed)
        kspace_shift = fftpack.fftshift(kspace)

        # Undersample/Mask kspace
        m_kspace = kspace_shift * mask  # masked k-space

        # Shift kspace back (with complex part)
        m_kspace_t = fftpack.ifftshift(m_kspace)
        
        # Transform back to image space (IFFT)
        slice_recon = np.abs(fftpack.ifft2(m_kspace_t))

        # Assign slice to 3D volume.
        recon_n[:, : , slice_idx] = slice_recon

        # if slice_idx == 10:
        #     save_slice_2d(slice, "temp/", "1_inp")
        #     save_kspace(kspace, "temp/", "2_kspace", True)
        #     save_kspace(kspace_shift, "temp/", "3_kspace_shift", True)
        #     save_kspace(m_kspace, "temp/", "4_kspace_shift_mask", True)
        #     save_kspace(m_kspace_t, "temp/", "5_masked_kspace", True)
        #     save_slice_2d(slice_recon, "temp/", "6_out")

    # Transform to sitk and copy original img metadata to reconstruction.
    recon_s = sitk.GetImageFromArray(recon_n.T)
    recon_s.CopyInformation(img_s)
    
    return recon_s


def binarize_s(img_s: sitk.Image) -> sitk.Image:
    """ Performs binary threshold on the sitk input image.

    Parameters:
    `img_s (sitk.Image)`: Image that needs to be binarized.
    return: Binarized image.
    """

    filter = sitk.BinaryThresholdImageFilter()
    filter.SetLowerThreshold(1.0)
    img = filter.Execute(img_s)
    return sitk.Cast(img, sitk.sitkFloat32)



def znorm_n(arr: np.ndarray) -> np.ndarray:
    """ Performs standard Z-normalization

    Parameters:
    `arr (np.ndarray)`: Numpy array of an image that needs to be z-normalized
    return: Z-normalized np.ndarray.
    """

    return np.divide(arr - np.mean(arr), np.std(arr))


def normalize(norm_str: str, img_s: sitk.Image) -> sitk.Image:
    """ Normalizes the given sitk.Image according to the supplied normalize
        method.

    Parameters:
    `norm_str (str)`: Normalize method as string.
        Options = ("znorm", "rescale_0_1")
    Returns:
    `(sitk.Image)`: normalized according to given method 
    """
    if norm_str == "znorm":
        return znorm_s(img_s)
    if norm_str == "rescale_0_1":
        return norm_rescale_intensity_s(img_s)
    raise ValueError(f"The given '{norm_str}' normalization method does not exist")


def znorm_s(img_s: sitk.Image):
    """ Performs standard Z-normalization

    Parameters:
    `img_s (sitk.Image)`: Image to be normalized. It is an sitk.Image object
    
    Returns:
    `(sitk.Image)`: Z-normalized
    """
    filter = sitk.NormalizeImageFilter()
    return filter.Execute(img_s)


def norm_rescale_intensity_s(img_s: sitk.Image):
    """ Performs normalization of the 3D input volume to values between 0 and 1.

    Parameters:
    `img_s (sitk.Image)`: 3D input volume. An MR Image
    Returns:
    `(sitk.Image)`: Z-normalized np.ndarray.
    """
    filter = sitk.RescaleIntensityImageFilter()
    filter.SetOutputMaximum(1)
    filter.SetOutputMinimum(0)
    return filter.Execute(img_s)


def center_crop(
    image: Image,
    shape: List[int],
    offset: List[int] = None
    ) -> sitk.Image:
    """Extracts a region of interest from the center of an SITK image.

    Parameters:
    `image (sitk.Image)`: Input image (SITK).
    `shape (List[int])`: The shape of the crop
    `offset (List[int])`: x, y, z offsets as list
    
    Returns:
    `cropped_image (sitk.Image)`: Cropped image
    """

    size = image.GetSize()

    # Determine the centroid
    centroid = [sz / 2 for sz in size]
    
    # Determine the origin for the bounding box by subtracting half the 
    # shape of the bounding box from each dimension of the centroid.
    box_start = [int(c - sh / 2) for c, sh in zip(centroid, shape)]

    if offset:
        box_start = [b - o for b, o in zip(box_start, offset)]

    # Extract the region of provided shape starting from the previously
    # determined starting pixel.
    region_extractor = sitk.RegionOfInterestImageFilter()
    region_extractor.SetSize(shape)
    region_extractor.SetIndex(box_start)
    cropped_image = region_extractor.Execute(image)

    return cropped_image


def resample( 
    image: Image,
    min_shape: List[int],
    method=sitk.sitkLinear, 
    new_spacing: List[float]=[0.5, 0.5, 3.0]
    ) -> Image:
    """Resamples an image to given target spacing and shape.
    
    Parameters:
    `image (sitk.Image)`: Input image.
    `min_shape (List[str])`: Minimum output shape for the underlying array.
    `method (sitk.Method)`: SimpleITK interpolator to use for resampling. 
        (e.g. sitk.sitkNearestNeighbor, sitk.sitkLinear)
    `new_spacing (List[float])`: The new spacing to resample to.
    
    Returns:
    `resampled_img (sitk.Image)`: Resampled image
   """

    # Extract size and spacing from the image
    size = image.GetSize()
    spacing = image.GetSpacing()

    # Determine how much larger the image will become with the new spacing
    factor = [sp / new_sp for sp, new_sp in zip(spacing, new_spacing)]

    # Determine the outcome size of the image for each dimension
    get_size = lambda size, factor, min_shape: max(int(size * factor), min_shape)
    new_size = [get_size(sz, f, sh) for sz, f, sh in zip(size, factor, min_shape)]

    # Resample the image 
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(method)
    resampled_image = resampler.Execute(image)

    return resampled_image