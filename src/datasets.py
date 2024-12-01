import torch
import torch.utils.data as torch_data
import nibabel as nib
import pandas as pd
from pathlib import Path


class MRIDataset(torch_data.Dataset):
    def __init__(self, path_to_csv):
        super().__init__()

        # Loading metadata
        self.metadata = pd.read_csv(
            path_to_csv)[['path', 'Subject', 'Group', 'Sex', 'Age']]

        # Selecting subjects with CN and AD
        self.metadata = self.metadata[self.metadata['Group'].isin([
                                                                  'CN', 'AD'])].iloc[: 100]
        # Maping the values in Group and Sex to 0 and 1
        self.metadata['Group'] = self.metadata['Group'].map({'CN': 0, 'AD': 1})
        self.metadata['Sex'] = self.metadata['Sex'].map({'M': 0, 'F': 1})

        self.path = self.metadata['path'].to_numpy()
        self.Subjects = self.metadata['Subject'].to_numpy()
        self.metadata = self.metadata.to_numpy()[:, 2:].astype(int)

        self.n_samples = self.path.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        path_subject = self.path[idx]
        assert Path(path_subject).exists(), f"File not found: {path_subject}"
        scan = nib.load(path_subject).get_fdata()
        scan = scan[:, scan.shape[1] // 2]
        scan = (scan - scan.min()) / scan.max() * 2 - 1
        scan_tensor = torch.tensor(scan, dtype=torch.float64)
        labels_tensor = torch.tensor(self.metadata[idx], dtype=torch.int)

        return scan_tensor.unsqueeze(0), labels_tensor
