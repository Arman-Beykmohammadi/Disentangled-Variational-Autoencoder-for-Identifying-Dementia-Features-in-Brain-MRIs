import torch
import torch.utils.data as torch_data
import pytorch_lightning as pl
import torch.nn as nn

from datasets import DummyDataset


class DummyIVAE(pl.LightningModule):
    def __init__(
            self,
            batch_size=128,
            lr=1e-3,
            coeff_kl=1,
            dim_latent_space=8,
            dim_labels=3,
            dim_input=64,
    ):
        super().__init__()
        self.save_hyperparameters()


        self.dataset = DummyDataset()
        # actually do subsetting here
        self.dataset_val = self.dataset
        self.dataset_train = self.dataset

        self.encoder = nn.Sequential(
            nn.Linear(dim_input, 16),
            nn.ReLU(),
            # 2 times the latent space dimensionality, since we will encode the mean and the variance of each dimension
            nn.Linear(16, 2 * self.hparams.dim_latent_space),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.dim_latent_space, 16),
            nn.ReLU(),
            nn.Linear(16, dim_input),
        )
        self.prior_encoder = nn.Sequential(
            nn.Linear(self.hparams.dim_labels, 16),
            nn.ReLU(),
            nn.Linear(16, 2 * self.hparams.dim_latent_space),
        )


    def train_dataloader(self):
        return torch_data.DataLoader(
            dataset=self.dataset_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch_data.DataLoader(
            dataset=self.dataset_val,
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            lr=self.hparams.lr,
            params=self.parameters(),
        )
        return optim

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
        return (-.5 + logvar_prior - logvar + ((mu - mu_prior).pow(2) + logvar.exp()) / (2 * logvar_prior.exp())).mean()


    def _step(self, batch, step_name, batch_idx=None):
        x, u = batch

        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar[:, self.hparams.dim_latent_space:], mu_logvar[:, :self.hparams.dim_latent_space]
        # reparameterization trick
        z = mu + torch.exp(logvar / 2) * torch.randn_like(mu)
        x_hat = self.decoder(z)
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
        for metric_name, metric_val in logs.items():
            self.log(f"{step_name}/{metric_name}", metric_val, prog_bar=True)

        return loss
