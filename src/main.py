import torch
import pytorch_lightning.loggers
from pytorch_lightning.cli import LightningCLI

from models import DummyIVAE


def cli_main():
    logger = pytorch_lightning.loggers.WandbLogger(project='your-project-name', save_dir='logs')

    cli = LightningCLI(
        save_config_callback=None,
        run=False,
        model_class=DummyIVAE,
        trainer_defaults={
            'logger': logger,
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'deterministic': True,
            'log_every_n_steps': 50,
        }
    )

    cli.trainer.fit(cli.model)


if __name__ == '__main__':
    cli_main()
