import torch
import pytorch_lightning as pl
import torch.nn as nn
from utils import resnet18_encoder, resnet18_decoder, PriorEncoder
import wandb


class IVAE(pl.LightningModule):
    def __init__(
            self,
            batch_size=64,
            lr=1e-3,
            coeff_kl=1,
            dim_latent_space=8,
            input_height=176,
            dim_labels=3,
            first_conv=False,
            maxpool1=False,
            fold_idx=None
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoder using ResNet18
        self.encoder = nn.Sequential(resnet18_encoder(
            first_conv=first_conv, maxpool1=maxpool1,),
            nn.Linear(512, 2 * self.hparams.dim_latent_space),
        )

        # Decoder using ResNet18
        self.decoder = resnet18_decoder(
            latent_dim=self.hparams.dim_latent_space,
            input_height=self.hparams.input_height,
            first_conv=first_conv,
            maxpool1=maxpool1,
        )

        # Prior Encoder using fc layers
        self.prior_encoder = PriorEncoder(
            dim_labels=self.hparams.dim_labels,
            latent_dim=2 * self.hparams.dim_latent_space,
            hidden_layers=[256, 128, 64],
        )

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx=None):
        return self._step(batch=batch, batch_idx=batch_idx, step_name='train')

    def validation_step(self, batch, batch_idx=None):
        return self._step(batch=batch, batch_idx=batch_idx, step_name='val')

    def compute_kl(
            self,
            mu,
            logvar,
            mu_prior,
            logvar_prior,
    ):
        # formula for KL divergence between two diagonal gaussians
        return (-.5 + 0.5*(logvar_prior - logvar) + ((mu - mu_prior).pow(2) + logvar.exp()) / (2 * logvar_prior.exp())).mean()

    def _step(self, batch, step_name, batch_idx=None):
        x, u = batch

        x_hat, mu, logvar = self(x)
        loss_rec = ((x - x_hat) ** 2).mean()

        mu_logvar_prior = self.prior_encoder(u)
        mu_prior = mu_logvar_prior[:, self.hparams.dim_latent_space:]
        logvar_prior = mu_logvar_prior[:, :self.hparams.dim_latent_space]

        loss_kl = self.compute_kl(
            mu=mu,
            logvar=logvar,
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
        )

        loss = loss_rec + self.hparams.coeff_kl * loss_kl
        logs = {
            'loss': loss,
            'loss_rec': loss_rec,
            'loss_kl': loss_kl,
        }
        batch_size = x.size(0)
        for metric_name, metric_val in logs.items():
            self.log(f"{step_name}/Fold {self.hparams.fold_idx}-{metric_name}",
                     metric_val, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)

        return loss

    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar[:, self.hparams.dim_latent_space:], mu_logvar[:,
                                                                             :self.hparams.dim_latent_space]
        # reparameterization trick
        z = mu + torch.exp(logvar / 2) * torch.randn_like(mu)
        x_hat = self.decoder(z)

        return x_hat, mu, logvar


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, fold_idx):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.fold_idx = fold_idx

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        x_hat, _mu, _logvar = pl_module(val_imgs)

        val_imgs = (val_imgs - val_imgs.min()) / \
            (val_imgs.max() - val_imgs.min())
        x_hat = (x_hat - x_hat.min()) / (x_hat.max() - x_hat.min())

        trainer.logger.experiment.log({
            f"Fold {self.fold_idx}-input": [wandb.Image(x.cpu().numpy().reshape((176, 176, 1)), caption=f"Label:{y}")
                                            for x, y in zip(val_imgs, self.val_labels)],
            f"Fold {self.fold_idx}-reconstructed": [wandb.Image(x.cpu().numpy().reshape((176, 176, 1)), caption=f"Label:{y}")
                                                    for x, y in zip(x_hat, self.val_labels)],
        })
