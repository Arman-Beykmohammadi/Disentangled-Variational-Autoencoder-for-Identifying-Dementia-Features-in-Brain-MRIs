import os
import torch
import nibabel as nib
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms


class MRIDataset(Dataset):
    def __init__(self, csv_path, axis="coronal", max_visits=None, transform=None):
        super().__init__()

        # Loading metadata
        self.metadata = pd.read_csv(csv_path)
        self.axis = axis
        self.transform = transform

        self.metadata = self.metadata[self.metadata['Group'].isin(['CN', 'AD'])]
        self.metadata['Group'] = self.metadata['Group'].map({'CN': 0, 'AD': 1})
        self.metadata['Sex'] = self.metadata['Sex'].map({'M': 0, 'F': 1})

        self.paths = self.metadata['path'].to_numpy()
        self.labels = self.metadata[['Group', 'Sex', 'Age']].to_numpy(dtype=int)
        self.max_visits = max_visits or len(self.paths)


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]  # Corrected attribute name
        assert Path(path).exists(), f"File not found: {path}"
        scan = nib.load(path).get_fdata()

        # Extract a 2D slice
        if self.axis == "coronal":
            slice_idx = scan.shape[1] // 2
            img_slice = scan[:, slice_idx, :]
        elif self.axis == "sagittal":
            slice_idx = scan.shape[0] // 2
            img_slice = scan[slice_idx, :, :]
        elif self.axis == "axial":
            slice_idx = scan.shape[2] // 2
            img_slice = scan[:, :, slice_idx]
        else:
            raise ValueError(f"Invalid axis: {self.axis}")
        
        # Normalize and transform
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-5)
        img_tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        label = torch.tensor(self.labels[idx], dtype=torch.int)
        return img_tensor, label
    
csv_path = "../data/dementia_df.csv"
dataset = MRIDataset(csv_path=csv_path, axis="coronal")
for i in range(5):
    img, label = dataset[i]
    print(f"Sample {i}: Image shape = {img.shape}, Label = {label}")