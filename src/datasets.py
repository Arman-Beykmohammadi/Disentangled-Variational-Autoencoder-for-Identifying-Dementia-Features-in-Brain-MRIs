import torch
import torch.utils.data as torch_data
import nibabel as nib
import pandas as pd
from pathlib import Path
import numpy as np
import torch.nn.functional as F


class MRIDataset(torch_data.Dataset):
    def __init__(self, path_to_csv):
        super().__init__()

        # Loading desired columns of metadata, path for image,
        # subject, group, sex, and age for use in prior encoder
        self.metadata = pd.read_csv(
            path_to_csv)[['path', 'Subject', 'Group', 'Sex', 'Age']]

        # Selecting subjects with CN and AD
        self.metadata = self.metadata[self.metadata['Group'].isin([
                                                                  'CN', 'AD'])]
        # Maping the values in Group and Sex to 0 and 1
        self.metadata['Group'] = self.metadata['Group'].map({'CN': 0, 'AD': 1})
        self.metadata['Sex'] = self.metadata['Sex'].map({'M': 0, 'F': 1})

        self.path = self.metadata['path'].to_numpy()
        self.Subjects = self.metadata['Subject'].to_numpy()
        self.metadata = self.metadata.to_numpy()[:, 2:].astype(np.float32)
        self.metadata[:, 2] /= self.metadata[:, 2].max()

        self.n_samples = self.path.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Loading the scan
        path_to_scan = self.path[idx]
        assert Path(path_to_scan).exists(), f"File not found: {path_to_scan}"
        scan = nib.load(path_to_scan).get_fdata()

        # Selecting the middle slice of the scan
        scan = scan[:, scan.shape[1] // 2]
        scan = (scan - scan.min()) / scan.max() * 2 - 1
        scan_tensor = torch.tensor(scan, dtype=torch.float32).unsqueeze(0)
        scan_tensor = F.interpolate(scan_tensor.unsqueeze(
            0), size=(176, 176), mode='bilinear', align_corners=False)
        scan_tensor = scan_tensor.squeeze(0)
        labels_tensor = torch.tensor(self.metadata[idx], dtype=torch.float32)

        return scan_tensor, labels_tensor
