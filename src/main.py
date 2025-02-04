import os
from datetime import datetime
import random
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import SubsetRandomSampler, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import StratifiedGroupKFold

from datasets import MRIDataset
from models import IVAE, ImagePredictionLogger

import wandb


def cli_main():
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

    # Initialize the wandb run first
    wandb.init()

    # Get hyperparameters from sweep
    configs = wandb.config

    # Name the run based on the configuration
    run_name = "_".join(f"{value}" for _, value in configs.items())
    wandb.run.name = run_name

    # Initialize the wandb logger to log everything
    wandb_logger = WandbLogger()
    wandb_logger.experiment.config.update(configs)

    dataset = MRIDataset(
        "/dhc/home/arman.beykmohammadi/AMLS/ml-seminar-brain-mri-dementia-ivae/df.csv")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    mean_val_loss = float(0)
    # Subject out Stratified Cross Validation (5 folds)
    for i, (train_idx, val_idx) in enumerate(sgkf.split(X=np.zeros(dataset.n_samples), y=dataset.metadata[:, 0], groups=dataset.Subjects)):
        print('--------------------')
        print(f'Starting fold {i+1}...')
        print(
            f"  Train: Shape={train_idx.shape}, Number of Groups={np.unique(dataset.Subjects[train_idx]).shape}, dementia status statistics={np.unique(dataset.metadata[train_idx,0], return_counts=True)}")

        print(
            f"  Validation: Shape={val_idx.shape}, Number of Groups={np.unique(dataset.Subjects[val_idx]).shape}, dementia status statistics={np.unique(dataset.metadata[val_idx,0], return_counts=True)}")

        # Initialize train loaders
        train_loader = DataLoader(dataset, batch_size=configs['batch_size'], sampler=SubsetRandomSampler(
            train_idx), num_workers=min(4, os.cpu_count()))

        # Initialize val loaders
        val_loader = DataLoader(dataset, batch_size=configs['batch_size'], sampler=SubsetRandomSampler(
            val_idx), num_workers=min(4, os.cpu_count()))

        val_samples = next(iter(DataLoader(dataset, batch_size=min(10, len(val_idx)), sampler=SubsetRandomSampler(
            val_idx), num_workers=min(4, os.cpu_count()))))

        # Initialize the model
        model = IVAE(
            batch_size=configs['batch_size'],
            lr=configs['lr'],
            coeff_kl=configs['coeff_kl'],
            dim_latent_space=configs['dim_latent_space'],
            input_height=176,
            fold_idx=i+1
        )

        # Create directory if it doesn't exist
        checkpoint_dir = f'./checkpoints/IVAE/{run_name}/fold_{i+1}'
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set up usefull callbacks
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch:03d}',
            save_top_k=1,
            verbose=True,
            monitor=f'val/Fold {i+1}-loss',
            save_last=True,
            mode='min'
        )
        earlystopping = EarlyStopping(
            patience=20,
            monitor=f'val/Fold {i+1}-loss',
            mode='min',
            check_finite=True,
            verbose=True
        )

        trainer = pl.Trainer(max_epochs=100,
                             callbacks=[earlystopping, checkpoint,
                                        ImagePredictionLogger(val_samples, fold_idx=i+1)],
                             accelerator='gpu',
                             devices=-1,
                             logger=wandb_logger)

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
        val_loss = trainer.validate(model, dataloaders=val_loader)
        end_time = datetime.now()
        print(
            f'Validating for fold {i+1} completed. Total time to validate {end_time - start_time}')

        mean_val_loss += val_loss[0][f'val/Fold {i+1}-loss']

        # Save the model
        trainer.save_checkpoint(checkpoint_dir + '/final_model.ckpt')

    mean_val_loss /= 5

    wandb_logger.experiment.log({'mean_val_loss': mean_val_loss})
    wandb.finish()


if __name__ == '__main__':
    print("started")
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'mean_val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'batch_size': {
                'values': [32, 64, 128]
            },
            'lr': {
                'values': [1e-3, 1e-4]
            },
            'coeff_kl': {
                'values': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            'dim_latent_space': {
                'values': [8, 16, 32, 64, 128]
            }
        }
    }
    # Initialize sweep
    # sweep_id = wandb.sweep(sweep_config, project="IVAE - Sweep")
    sweep_id = "arman-beykmohammadi/IVAE - Sweep/6hlmzx3b"

    # Start sweep agent
    wandb.agent(sweep_id, function=cli_main, count=1)
