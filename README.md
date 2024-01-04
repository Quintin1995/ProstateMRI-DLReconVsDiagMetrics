# ProstateMRI-DLReconVsDiagMetrics
Code for evaluating prostate MRI DL reconstructions. Focuses on T2-weighted scans, undersampling simulation, DL model training, and diagnostic metric analysis.


# Data Preparation on DICOM data

## Sqlite Database filling
This (`e6_fill_dicom_db.py`) script automates the process of filling a DICOM database with metadata extracted from DICOM files. It parses DICOM headers to extract relevant information like patient ID, study dates, and image paths. This data is then inserted into an SQLite database for easy access and management.

## Train-Validation-Test Indexes
The script `e12_make_train_val_test_indexes.py` is designed to create training, validation, and testing datasets for machine learning models. It efficiently segregates data by generating indices based on predefined criteria, ensuring a balanced and representative sample distribution across datasets.

# Model Training
## Training fo the reconstruction model (DLRecon)
`e24_train_t2w_recon_model.py` focuses on training reconstruction models for T2-weighted MRI images. It includes functionalities for data preparation, model configuration, training, and validation. This script is key in developing AI models for enhanced MRI image reconstruction.

## Training of the diagnostic model (DLDetect)
`e27_train_diag_t2_model.py` is designed for training diagnostic models specific to T2-weighted MRI. It encompasses the entire pipeline from data loading, model training, to validation. This script plays a vital role in leveraging AI for diagnostic purposes in MRI imaging.


# Evaluation
## Reconstruction Eval (DLRecon)
`e25_eval_recon_model.py` is dedicated to evaluating reconstruction models for T2-weighted MRI images. It processes test data, applies the trained model, and assesses the model's performance using various metrics. This script is crucial for understanding the effectiveness of the reconstruction models.

## Visual comparison of DLRecon outputs
`e23_comp_t2_accs_to_r1.py` is a Python script designed for comparing T2-weighted MRI images across various acceleration factors with a reference image (R1). It includes functions for loading and processing image data, performing comparisons, and saving results. The script is integral in assessing the quality of MRI reconstructions at different acceleration levels.

## Diagnostic Eval (DLDetect Model)
`e28_eval_t2w_diag_model.py` focuses on evaluating T2-weighted MRI diagnostic models. It involves applying the trained model on test datasets and analyzing the results using diagnostic performance metrics. This script is essential for assessing the accuracy and reliability of AI-driven diagnostic models in MRI.


# Advanced Analysis
`e29_predict_with_recon_model.py` Predicts with the reconstruction model to obtain the reconstructions on various acceleration rates.

`e30_perm_test_diag_models_and_froc_plot.py` Performs permutation tests and FROC curve analysis, highlighting the diagnostic models' performance.

`e31_predict_all_imgs_in_pipeline.py` Processes a series of images using trained models, focusing on patch-based processing. Here we refer to the input images, the intermediate aliased images, the reconstructed images, the csPCa likelihood maps and FROC binarization per lesion.

`e35_patient_level_error_analysis.py` Prediction, statistical analysis, and error analysis in the context of DL reconstruction and diagnostic evaluation.