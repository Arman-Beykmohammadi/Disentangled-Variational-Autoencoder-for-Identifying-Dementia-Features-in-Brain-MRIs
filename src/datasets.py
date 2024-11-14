import torch
import torch.utils.data as torch_data


class MRIDataset(torch_data.Dataset):
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return torch.randn(64), torch.randn(3)
