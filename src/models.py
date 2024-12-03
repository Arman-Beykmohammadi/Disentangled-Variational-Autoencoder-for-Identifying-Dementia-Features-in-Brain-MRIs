import torch
import torch.nn as nn
import pytorch_lightning as pl
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
            prior_hidden_dim=128,

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
            dim_labels=dim_labels,
            latent_dim=dim_latent_space,
            hidden_dim=prior_hidden_dim,
        )

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) using N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def compute_kl(self, mu, logvar, mu_prior, logvar_prior):
        """KL divergence between two diagonal Gaussians."""
        kl_divergence = -0.5 * torch.sum(
            1 + logvar - logvar_prior - ((mu - mu_prior).pow(2) + logvar.exp()) / logvar_prior.exp(),
            dim=-1
        )
        return kl_divergence.mean()

    def compute_loss(self, x, x_hat, mu, logvar, mu_prior, logvar_prior):
        """Compute the combined reconstruction and KL divergence loss."""
        loss_rec = nn.MSELoss()(x, x_hat)
        loss_kl = self.compute_kl(mu, logvar, mu_prior, logvar_prior)
        total_loss = loss_rec + self.hparams.coeff_kl * loss_kl
        return total_loss, loss_rec, loss_kl

    def forward(self, x, u):
        """Forward pass."""
        # Encoder
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)

        # Reparameterization
        z = self.reparameterize(mu, logvar)

        # Decoder
        x_hat = self.decoder(z)

        # Prior Encoder
        mu_prior, logvar_prior = self.prior_encoder(u)

        return x_hat, mu, logvar, mu_prior, logvar_prior

    def _step(self, batch, step_name):
        """Common step logic for training/validation."""
        x, u = batch

        # Forward pass
        x_hat, mu, logvar, mu_prior, logvar_prior = self.forward(x, u)

        # Compute loss
        total_loss, loss_rec, loss_kl = self.compute_loss(x, x_hat, mu, logvar, mu_prior, logvar_prior)

        # Logging
        logs = {
            f"{step_name}_loss": total_loss,
            f"{step_name}_loss_rec": loss_rec,
            f"{step_name}_loss_kl": loss_kl,
        }
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, step_name="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, step_name="val")