from pathlib import Path
import SimpleITK as sitk
import numpy as np

if __name__ == "__main__":
    
    # Set the root for RUMC and UMCG
    root_rumc = Path('/projects/hb-pca-rad/datasets/000_tmp_backup_till_01-01-2024/radboud_lesions_2022/')
    root_umcg = Path('/projects/hb-pca-rad/datasets/000_tmp_backup_till_01-01-2024/umcg_lesions_2022/')

    # Read all the segs used in the project
    seg_path_list = Path('/scratch/p290820/fastMRI_PCa/data/path_lists/pirads_4plus/seg_new.txt').read_text().splitlines()

    # Get all the .ni.gz files in the projects directory
    all_seg_files = list(root_rumc.glob('**/*.gz'))
    all_seg_files += list(root_umcg.glob('**/*.gz'))
    print(f'Number of files all the files: {len(all_seg_files)}')

    # loop over all the files that we want to find so that we can make the path to the file, so that we can read it later.
    new_paths = []
    for seg_path in seg_path_list:

        if seg_path.split('/')[4] == "radboud_lesions_2022":
            seg_path_new = root_rumc / seg_path.split('/')[-1]
        if seg_path.split('/')[4] == "umcg_lesions_2022":
            seg_path_new = root_umcg / seg_path.split('/')[-1]
        new_paths.append(seg_path_new)

    print(f'Number of new paths: {len(new_paths)}')

    positives, negatives = 0, 0
    for seg_path in new_paths:
        seg = sitk.ReadImage(str(seg_path))
        seg = sitk.GetArrayFromImage(seg)
        if np.sum(seg) >= 1:
            positives += 1
        elif np.sum(seg) == 0:
            negatives += 1
        else:
            print(f'{seg_path} is not correct')

    print(f'Number of positives: {positives}')
    print(f'Number of negatives: {negatives}')
