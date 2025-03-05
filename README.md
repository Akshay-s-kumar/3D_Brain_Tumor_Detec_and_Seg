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

Introduction
The goal of this project is to develop a deep learning model for segmenting brain tumors from 3D MRI scans. The model uses a 3D CNN architecture with frequency-aware learning to capture both spatial and frequency-domain features for improved segmentation accuracy.

Dataset
The dataset used is the BraTS 2021 dataset, which contains multi-modal MRI scans (T1, T1ce, T2, FLAIR) and corresponding segmentation masks. The segmentation masks include the following labels:

0: Background (normal tissue)

1: Necrotic and non-enhancing tumor core (NCR/NET)

2: Peritumoral edema (ED)

4: Enhancing tumor (ET)

Each MRI scan is a 3D volume of size (240, 240, 155).

