import torch
import pytorch_lightning as pl
import torch.nn as nn
from utils import resnet18_encoder, resnet18_decoder, PriorEncoder


class IVAE(pl.LightningModule):
    def __init__(
            self,
            batch_size=128,
            lr=1e-3,
            coeff_kl=1,
            dim_latent_space=8,
            dim_labels=3,
            first_conv=False,
            maxpool1=False,
            input_height=64,

    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoder using ResNet18
        self.encoder = resnet18_encoder(
            first_conv=first_conv, maxpool1=maxpool1,)
        self.fc_mu = nn.Linear(512, self.hparams.dim_latent_space)
        self.fc_logvar = nn.Linear(512, self.hparams.dim_latent_space)

        # Decoder using ResNet18
        self.decoder = resnet18_decoder(
            latent_dim=self.hparams.dim_latent_space,
            input_height=input_height,
            first_conv=first_conv,
            maxpool1=maxpool1,
        )

        # Prior Encoder using fc layers
        self.prior_encoder = PriorEncoder(
            dim_labels=self.hparams.dim_labels,
            latent_dim=self.hparams.dim_latent_space,
            hidden_dim=128,
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
        return (-.5 + 0.5*(logvar_prior - logvar) + ((mu - mu_prior).pow(2) + logvar.exp()) / (2 * logvar_prior.exp())).mean()

    def _step(self, batch, step_name, batch_idx=None):
        x, u = batch
        print(f"x.shape: {x.shape}, u.shape: {u.shape}")
        print(f"type(x): {type(x)}, type(u): {type(u)}")

        # mu_logvar = self.encoder(x)
        # mu = mu_logvar[:, self.hparams.dim_latent_space:]
        # logvar = mu_logvar[:, :self.hparams.dim_latent_space]
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)

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
