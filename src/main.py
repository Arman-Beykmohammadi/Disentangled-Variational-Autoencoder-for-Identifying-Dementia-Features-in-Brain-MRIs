import optuna
import os
from datetime import datetime
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import SubsetRandomSampler, DataLoader

from sklearn.model_selection import StratifiedGroupKFold

from datasets import MRIDataset
from models import IVAE, ImagePredictionLogger


def cli_main():
    dataset = MRIDataset(
        "/dhc/home/arman.beykmohammadi/AMLS/ml-seminar-brain-mri-dementia-ivae/df.csv")
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True)

    # For creating wand project based on the date and time of the run
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Start cross-validation on the training set
    for i, (train_idx, val_idx) in enumerate(sgkf.split(np.zeros(dataset.n_samples), dataset.metadata[:, 0], dataset.Subjects)):
        print('--------------------')
        print(f'Starting fold {i+1}...')
        print(
            f"  Train: Shape={train_idx.shape}, Number of Groups={np.unique(dataset.Subjects[train_idx]).shape}, dementia status statistics={np.unique(dataset.metadata[train_idx,0], return_counts=True)}")

        print(
            f"  Validation: Shape={val_idx.shape}, Number of Groups={np.unique(dataset.Subjects[val_idx]).shape}, dementia status statistics={np.unique(dataset.metadata[val_idx,0], return_counts=True)}")

        # Initialize train loaders
        train_loader = DataLoader(dataset, batch_size=32, sampler=SubsetRandomSampler(
            train_idx), num_workers=min(os.cpu_count(), 4))

        # Initialize val loaders
        val_loader = DataLoader(dataset, batch_size=len(val_idx), sampler=SubsetRandomSampler(
            val_idx), num_workers=min(os.cpu_count(), 4))

        val_samples = next(iter(DataLoader(dataset, batch_size=min(10, len(val_idx)), sampler=SubsetRandomSampler(
            val_idx), num_workers=min(os.cpu_count(), 4))))

        # Initialize the model
        model = IVAE(
            batch_size=32,
            lr=1e-3,
            coeff_kl=1,
            dim_latent_space=8,
            dim_labels=3,
            input_height=176
        )

        wandb_logger = WandbLogger(
            project=f'IVAE - {time}', name=f'fold_{i+1}')
        trainer = pl.Trainer(max_epochs=200,
                             callbacks=[ImagePredictionLogger(val_samples)],
                             accelerator='gpu', devices=-1, logger=wandb_logger)

        print('-------------------')
        print(f'Starting training on fold {i+1}...')
        start_time = datetime.now()
        trainer.fit(model, train_dataloaders=train_loader,
                    val_dataloaders=val_loader)
        end_time = datetime.now()
        print(
            f'Training for fold {i+1} completed. Total time to train: {end_time - start_time}')

        print('-------------------')
        print(f'Starting validating on fold {i+1}...')
        start_time = datetime.now()
        trainer.validate(model, dataloaders=val_loader)
        end_time = datetime.now()
        print(
            f'Validating for fold {i+1} completed. Total time to validate {end_time - start_time}')

        # Create directory if it doesn't exist
        checkpoint_dir = f'./checkpoints/fold_{i+1}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model
        trainer.save_checkpoint(checkpoint_dir + '/final_model.ckpt')

        wandb_logger.experiment.finish()


if __name__ == '__main__':
    cli_main()
