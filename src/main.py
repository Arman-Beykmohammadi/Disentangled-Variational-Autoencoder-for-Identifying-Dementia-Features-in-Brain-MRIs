import os
from datetime import datetime
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import SubsetRandomSampler, DataLoader

from sklearn.model_selection import StratifiedGroupKFold

from datasets import MRIDataset
from models import IVAE
from config import CONFIG

def cli_main():
    dataset = MRIDataset(csv_path=CONFIG['data_csv'], axis="coronal")
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    model = IVAE(CONFIG)

    wandb_logger = WandbLogger(project="IVAE", name="experiment-1")
    trainer = pl.Trainer(
        max_epochs=CONFIG['num_epochs'],
        logger=wandb_logger,
        accelerator='gpu', devices=-1
    )
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    cli_main()
