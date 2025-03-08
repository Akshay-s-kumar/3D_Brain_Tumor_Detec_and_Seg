# 3D_Brain_Tumor_Detec_and_Seg

Brain Tumor Segmentation using 3D CNN with Frequency-Aware Learning
This repository contains the implementation of a 3D Convolutional Neural Network (3D CNN) with frequency-aware learning for brain tumor segmentation using the BraTS 2021 dataset.

Table of Contents
Introduction

Dataset

Setup

Code Overview

Results

Future Work

References

=============================================================================
1️⃣ Preprocessing

Convert the uploaded MRI scan to match BraTS format (e.g., rescale, normalize).
Ensure the scan includes the same modalities (T1, T2, FLAIR, T1-Gd).

::
1. Problem Definition
Objective 1: Classify brain tumors as Benign or Malignant.
Objective 2: Classify glioma subtypes (e.g., GBM, LGG).
Input: Multi-modal MRI scans (T1, T1c, T2, FLAIR).
Output: Classification labels (Benign/Malignant, Glioma Subtypes).

2. Data Preparation
Dataset Structure
The dataset contains NIfTI files for each patient:
T1: Native T1-weighted MRI.
T1c: Post-contrast T1-weighted MRI.
T2: T2-weighted MRI.
FLAIR: T2-FLAIR MRI.
Seg: Ground truth segmentation masks (labels: 1 = NCR, 2 = ED, 4 = ET).

Steps
1: Load NIfTI Files:
Use a library like nibabel to load .nii.gz files into numpy arrays.
Each MRI modality will be a 3D volume (e.g., 240x240x155).
2: Preprocessing:
Normalization: Normalize each modality to have zero mean and unit variance.
Resampling: Ensure all modalities are aligned and have the same resolution (1 mm³).
Skull Stripping: Use the provided skull-stripped data.
Cropping: Crop the volumes to remove unnecessary background regions.
Data Augmentation: Apply augmentations like rotation, flipping, and scaling to increase dataset diversity.
3: Label Preparation:
Benign vs. Malignant:
Use the segmentation masks (seg.nii.gz) to determine tumor type.
For example, if the tumor has a significant enhancing region (label 4), classify it as malignant.
Glioma Subtypes:
Use metadata or external labels (if available) to classify gliomas into subtypes like GBM or LGG.
4: Dataset Splitting:
Split the dataset into training, validation, and test sets (e.g., 70% training, 15% validation, 15% testing).

2️⃣ Model Inference

The trained segmentation model predicts tumor presence.
Generates a segmentation mask with labels (1, 2, 4).

3️⃣ Overlay Visualization

The output mask is overlaid on the MRI scan for easy interpretation.
Different colors highlight ET, ED, and NCR regions.
=====================================================================

Deployment Possibilities

Web App (Flask/Django + React/Streamlit) → Users upload scans & see predictions.
Cloud API (FastAPI/TensorFlow Serving) → Process MRI scans via REST API.
Mobile App (TensorFlow Lite) → Scan MRI reports via phone.
