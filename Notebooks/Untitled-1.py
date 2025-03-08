# %%
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchio as tio  # For preprocessing

# Function to load NIfTI files
def load_nifti(file_path):
    nifti_img = nib.load(file_path)  
    return np.array(nifti_img.get_fdata(), dtype=np.float32)  # Convert to NumPy array

# Example Paths (Modify these paths)
t1_path = r"D:\Downloads\BraTS2021_Training_Data\BraTS2021_00000\BraTS2021_00000_t1.nii.gz"
t1ce_path = r"D:\Downloads\BraTS2021_Training_Data\BraTS2021_00000\BraTS2021_00000_t1ce.nii.gz"
t2_path = r"D:\Downloads\BraTS2021_Training_Data\BraTS2021_00000\BraTS2021_00000_t2.nii.gz"
flair_path = r"D:\Downloads\BraTS2021_Training_Data\BraTS2021_00000\BraTS2021_00000_flair.nii.gz"
seg_path = r"D:\Downloads\BraTS2021_Training_Data\BraTS2021_00000\BraTS2021_00000_seg.nii.gz"

# Load each modality
t1 = load_nifti(t1_path)
t1ce = load_nifti(t1ce_path)
t2 = load_nifti(t2_path)
flair = load_nifti(flair_path)
seg = load_nifti(seg_path)  # Segmentation mask

# Stack modalities as channels (C, H, W, D) -> (4, 240, 240, 155)
input_data = np.stack([t1, t1ce, t2, flair], axis=0)  # Input with 4 channels

# Convert segmentation mask (C, H, W, D) -> (1, 240, 240, 155)
seg = np.expand_dims(seg, axis=0)

# Convert to PyTorch tensor
input_tensor = torch.tensor(input_data, dtype=torch.float32)
seg_tensor = torch.tensor(seg, dtype=torch.long)  # Long type for segmentation labels

print("Input shape:", input_tensor.shape)  # Should be (4, 240, 240, 155)
print("Segmentation mask shape:", seg_tensor.shape)  # Should be (1, 240, 240, 155)


# %%
import nibabel as nib
import numpy as np
import torch
import os


# Function to load NIfTI files
def load_nifti(file_path):
    nifti_img = nib.load(file_path)  
    return np.array(nifti_img.get_fdata(), dtype=np.float32)  # Convert to NumPy array

# Example Paths (Modify these paths)
t1_path = r"D:\Downloads\BraTS2021_Training_Data\BraTS2021_00000\BraTS2021_00000_t1.nii.gz"
t1ce_path = r"D:\Downloads\BraTS2021_Training_Data\BraTS2021_00000\BraTS2021_00000_t1ce.nii.gz"
t2_path = r"D:\Downloads\BraTS2021_Training_Data\BraTS2021_00000\BraTS2021_00000_t2.nii.gz"
flair_path = r"D:\Downloads\BraTS2021_Training_Data\BraTS2021_00000\BraTS2021_00000_flair.nii.gz"
seg_path = r"D:\Downloads\BraTS2021_Training_Data\BraTS2021_00000\BraTS2021_00000_seg.nii.gz"

# Load each modality
t1 = load_nifti(t1_path)
t1ce = load_nifti(t1ce_path)
t2 = load_nifti(t2_path)
flair = load_nifti(flair_path)
seg = load_nifti(seg_path)  # Segmentation mask

# Stack modalities as channels (C, H, W, D) -> (4, 240, 240, 155)
input_data = np.stack([t1, t1ce, t2, flair], axis=0)  # Input with 4 channels

# Convert segmentation mask (C, H, W, D) -> (1, 240, 240, 155)
seg = np.expand_dims(seg, axis=0)

# Get unique labels from segmentation mask (including background)
unique_labels = np.unique(seg)
print("Unique labels in segmentation mask:", unique_labels)

# Convert to PyTorch tensor
input_tensor = torch.tensor(input_data, dtype=torch.float32)
seg_tensor = torch.tensor(seg, dtype=torch.long)  # Long type for segmentation labels

print("Input shape:", input_tensor.shape)  # Should be (4, 240, 240, 155)
print("Segmentation mask shape:", seg_tensor.shape)  # Should be (1, 240, 240, 155)


# Define the output directory
output_dir = r"C:\Users\LENOVO\3D_Brain_Tumor_Detec_and_Seg\Dataset\Processed"

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the processed MRI as a NIfTI file
output_mri_path = os.path.join(output_dir, "processed_mri.nii.gz")
output_nifti = nib.Nifti1Image(input_data, affine=np.eye(4))  # Identity affine matrix
nib.save(output_nifti, output_mri_path)
print(f"3D MRI saved at: {output_mri_path}")

# Save the segmentation mask as a NIfTI file
output_seg_path = os.path.join(output_dir, "processed_segmentation.nii.gz")
output_seg_nifti = nib.Nifti1Image(seg, affine=np.eye(4))
nib.save(output_seg_nifti, output_seg_path)
print(f"3D Segmentation mask saved at: {output_seg_path}")



# %%
print("Input Data Shape:", input_data.shape)
print("Input Data Min:", np.min(input_data))
print("Input Data Max:", np.max(input_data))
print("Input Data Mean:", np.mean(input_data))


# %% [markdown]
# 🔧 Fix: Normalize Input Data
# Before saving the MRI file, normalize the pixel values to 0-1 or 0-255 for better visualization.
# 

# %%
# Normalize to [0,1]
input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))

# Scale to [0,255] for better visualization
input_data = (input_data * 255).astype(np.uint8)


# %%
output_mri_path = r"C:\Users\LENOVO\3D_Brain_Tumor_Detec_and_Seg\Dataset\Processed\processed_mri.nii.gz"
output_nifti = nib.Nifti1Image(input_data, affine=np.eye(4))
nib.save(output_nifti, output_mri_path)

print(f"✅ 3D MRI saved at: {output_mri_path}")


# %%
import nibabel as nib
import numpy as np

mri_path = r"C:\Users\LENOVO\3D_Brain_Tumor_Detec_and_Seg\Dataset\Processed\processed_mri.nii.gz"
mri_img = nib.load(mri_path)
mri_data = np.array(mri_img.get_fdata())

print(f"Min Intensity: {mri_data.min()}")
print(f"Max Intensity: {mri_data.max()}")
print(f"Mean Intensity: {mri_data.mean()}")


# %% [markdown]
# Your MRI data is not completely empty (Min: 0.0, Max: 255.0), but the mean intensity is very low (3.71), which suggests that most of the voxel values are close to zero. This is why your scan appears black in ITK-SNAP.

# %% [markdown]
# 2️⃣ Affine Matrix Issue
# If the affine matrix is incorrect, ITK-SNAP might not display the image.
# Try using the affine from the original MRI scan instead of np.eye(4).

# %%
original_mri = nib.load(t1_path)  # Use any original MRI
correct_affine = original_mri.affine  # Get the correct affine

output_nifti = nib.Nifti1Image(input_data, affine=correct_affine)
nib.save(output_nifti, mri_path)

print("✅ MRI saved with the correct affine matrix!")


# %% [markdown]
#  Wrong Data Format (Channel First vs. Channel Last)
# Your input_data has shape (4, 240, 240, 155) (4 channels), but nibabel expects (H, W, D, C) or (H, W, D).
# Save only one channel at a time, or use .transpose().

# %%
# Extract the first channel (T1 modality) and save
single_channel_mri = input_data[0]  # Use only the first modality
output_nifti = nib.Nifti1Image(single_channel_mri, affine=correct_affine)
nib.save(output_nifti, mri_path)

print("✅ Saved first channel MRI. Try loading it in ITK-SNAP.")


# %% [markdown]
# ✅ Fix 1: Normalize Intensity for Better Visibility
# Since the intensity range is 0-255, ITK-SNAP might not display it correctly if the majority of voxels are near 0. Try scaling the intensity:

# %%
# Load MRI
import nibabel as nib
import numpy as np

mri_path = r"C:\Users\LENOVO\3D_Brain_Tumor_Detec_and_Seg\Dataset\Processed\processed_mri.nii.gz"
mri_img = nib.load(mri_path)
mri_data = np.array(mri_img.get_fdata())

print(f"Min Intensity: {mri_data.min()}")
print(f"Max Intensity: {mri_data.max()}")
print(f"Mean Intensity: {mri_data.mean()}")

# Normalize intensities to [0, 255]
mri_data = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min()) * 255
mri_data = mri_data.astype(np.uint8)  # Convert to 8-bit integer

print(f"Min Intensity: {mri_data.min()}")
print(f"Max Intensity: {mri_data.max()}")
print(f"Mean Intensity: {mri_data.mean()}")

# Save normalized MRI
normalized_nifti = nib.Nifti1Image(mri_data, affine=mri_img.affine)
nib.save(normalized_nifti, r"C:\Users\LENOVO\3D_Brain_Tumor_Detec_and_Seg\Dataset\Processed\processed_mri.nii.gz")

print("✅ Saved normalized MRI. Try loading 'processed_mri.nii.gz' in ITK-SNAP!")


# %%



