import os
import SimpleITK as sitk
import umcglib.images as im

dirpath = os.path.join('figs_workspace', "pat240_false_positive")

target_shape = (180, 180, 16)
has_origin = "240_5segs_pred_crop_R1.0.nii.gz"

to_crop = [
    '240_1unaccelerated_inp_R1.0.nii.gz',
    '240_1unaccelerated_inp_R4.0.nii.gz',
    '240_2usampled_R1.0.nii.gz',
    '240_2usampled_R4.0.nii.gz',
    '240_5segs_pred_crop_R4.0.nii.gz',
    '240_5segs_pred_crop_R1.0.nii.gz',
    '240_3recons_R1.0.nii.gz',
    '240_3recons_R4.0.nii.gz'
    ]

has_origin_img = sitk.ReadImage(os.path.join(dirpath, has_origin), sitk.sitkFloat32)

for fname in to_crop:
    fpath = os.path.join(dirpath, fname)

    t2 = sitk.ReadImage(fpath, sitk.sitkFloat32)
    # t2_n = sitk.GetArrayFromImage(t2).T

    print(t2.GetSize()[0],)
    print(t2.GetSize()[1],)
    print(t2.GetSize()[2],)

    t2 = im.center_crop(t2, target_shape)

    print(t2.GetSize()[0],)
    print(t2.GetSize()[1],)
    print(t2.GetSize()[2],)

    t2.CopyInformation(has_origin_img)
    path = os.path.join(dirpath, "better"+ fname)
    sitk.WriteImage(t2, path)

    print(f"wrote to {path}\n")