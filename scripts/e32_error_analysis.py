import os
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
import csv


if __name__ == '__main__':

    workspace = os.path.join('figs_workspace', 'error_analysis')
    detect_dir = os.path.join('train_output', '91_diag_t2_r1_norm')

    n_folds = 5     # number of folds
    n_imgs  = 306    # number of images to be analysed in one go

    baseline_model_dir = "R1_"
    compare_model_dir  = "R4_80_recon_r4"


    csv_data = {
        "idx": [],
        "fold_num": [],
        "acceleration": [],
        "metric": [],
        "metric_val": []
    }

    for fold_num in range(n_folds):

        print(f"fold: {fold_num}")
        baseline_dir = os.path.join(detect_dir, f"fold_{fold_num}", 'test_preds', baseline_model_dir)
        compare_dir = os.path.join(detect_dir, f"fold_{fold_num}", 'test_preds', compare_model_dir)

        for img_num in range(n_imgs):

            print(f"\tImage num: {img_num}")

            # Read the image
            seg_fname      = os.path.join(baseline_dir, f"{img_num}_4segs_crop_R1.0.nii.gz")
            baseline_fname = os.path.join(baseline_dir, f"{img_num}_5segs_pred_crop_R1.0.nii.gz")
            compare_fname  = os.path.join(compare_dir, f"{img_num}_5segs_pred_crop_R4.0.nii.gz")
            baseline_img_s = sitk.ReadImage(baseline_fname, sitk.sitkFloat32)
            compare_img_s  = sitk.ReadImage(compare_fname, sitk.sitkFloat32)
            seg_s          = sitk.ReadImage(seg_fname, sitk.sitkFloat32)

            # Write to one folder for a nice overview
            sitk.WriteImage(baseline_img_s, os.path.join(workspace, f"{img_num}_r1_pred_fold{fold_num}.nii.gz"))
            sitk.WriteImage(compare_img_s, os.path.join(workspace, f"{img_num}_r4DL_pred_fold{fold_num}.nii.gz"))
            sitk.WriteImage(seg_s, os.path.join(workspace, f"{img_num}_seg.nii.gz"))

            # Calculate MSE between MSE1: (R1 and seg) vs MSE2: (R4DL and seg).
            # Find the cases where the difference in MSE1 and MSE2 is large
            baseline_img_n = sitk.GetArrayFromImage(baseline_img_s).T
            compare_img_n  = sitk.GetArrayFromImage(compare_img_s).T
            seg_n          = sitk.GetArrayFromImage(seg_s).T

            mse_base_vs_seg = (np.square(np.subtract(baseline_img_n, seg_n))).mean()
            mse_comp_vs_seg = (np.square(np.subtract(compare_img_n, seg_n))).mean()

            csv_data["idx"].append(img_num)
            csv_data["fold_num"].append(fold_num)
            csv_data["acceleration"].append("R1")
            csv_data["metric"].append('mse')
            csv_data["metric_val"].append(mse_base_vs_seg)

            csv_data["idx"].append(img_num)
            csv_data["fold_num"].append(fold_num)
            csv_data["acceleration"].append("R4DL")
            csv_data["metric"].append('mse')
            csv_data["metric_val"].append(mse_comp_vs_seg)


            print(f"\t\tMSE base vs seg  = {mse_base_vs_seg}")
            print(f"\t\tMSE base vs comp = {mse_comp_vs_seg}")


    df = pd.DataFrame(data=csv_data)

    csv_path = os.path.join(workspace, 'error_data.csv')
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    print(f"> Wrote csv file to: {csv_path}")

    print("done")