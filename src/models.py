import os
import torch
import torch.utils.data as torch_data
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms

from datasets import MRIDataset



# Define Interpolate and convolutional helper functions
class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""
    def __init__(self, size=None, scale_factor=None) -> None:
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode='nearest')

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def resize_conv3x3(in_planes, out_planes, scale=1):
    """Upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv3x3(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv3x3(in_planes, out_planes))

def resize_conv1x1(in_planes, out_planes, scale=1):
    """Upsample + 1x1 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv1x1(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv1x1(in_planes, out_planes))

# Define Encoder and Decoder Blocks
class EncoderBlock(nn.Module):
    """ResNet block."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class DecoderBlock(nn.Module):
    """ResNet block for Decoder with resize convolutions."""
    expansion = 1

    def __init__(self, inplanes, planes, scale=1, upsample=None) -> None:
        super().__init__()
        self.conv1 = resize_conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        return self.relu(out)

# Define ResNet Encoder and Decoder
class ResNetEncoder(nn.Module):
    def __init__(self, block, layers, in_channels=3, first_conv=False, maxpool1=False) -> None:
        super().__init__()

        self.inplanes = 64
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)           # Initial Conv
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # Max Pooling

        x = self.layer1(x)          # ResNet Layers
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)         # Global Average Pooling
        return torch.flatten(x, 1)  # Flatten to (batch_size, 512 * expansion)

class ResNetDecoder(nn.Module):
    """ResNet Decoder."""
    def __init__(self, block, layers, latent_dim, input_height, first_conv=False, maxpool1=False, out_channels=3) -> None:
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 512 * self.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 256, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 64, layers[2], scale=2)
        self.layer4 = self._make_layer(block, 64, layers[3], scale=2) if self.maxpool1 else self._make_layer(block, 64, layers[3], scale=2)

        self.conv1 = nn.Conv2d(64 * block.expansion, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()  # Assuming output is normalized between 0 and 1

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = x.view(x.size(0), self.inplanes, 4, 4)  # Reshape to (batch_size, inplanes, 4, 4)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv1(x)
        x = self.sigmoid(x)
        return x

# Factory functions for ResNet18
def resnet18_encoder(in_channels=3, first_conv=False, maxpool1=False):
    return ResNetEncoder(EncoderBlock, [2, 2, 2, 2], in_channels=in_channels, first_conv=first_conv, maxpool1=maxpool1)

def resnet18_decoder(latent_dim, input_height, first_conv=False, maxpool1=False, out_channels=3):
    return ResNetDecoder(DecoderBlock, [2, 2, 2, 2], latent_dim, input_height, first_conv=first_conv, maxpool1=maxpool1, out_channels=out_channels)


class IVAE(pl.LightningModule):
    def __init__(
            self,
            batch_size=128,
            lr=1e-3,
            coeff_kl=1,
            dim_latent_space=8,
            dim_labels=3,
            dim_input=64,
            img_channels=1
    ):
        super().__init__()
        self.save_hyperparameters()
        # Define transformations if needed
        transform = transforms.Compose([
            transforms.Resize((dim_input, dim_input)),
            transforms.ToTensor(),
            # Add normalization if necessary
        ])

        # Initialize datasets
        full_dataset = MRIDataset(transform=transform)  # Ensure MRIDataset accepts transforms

        # actually do subsetting here
        # implement the split method
        self.dataset_val = self.full_dataset
        self.dataset_train = self.full_dataset

        # Initialize ResNet-based Encoder and Decoder
        self.encoder = resnet18_encoder(in_channels=img_channels, first_conv=False, maxpool1=False)
        encoder_output_dim = 512  # For ResNet18, layer4 outputs 512 * expansion

        # Linear layer to map encoder output to mu and logvar
        self.fc_mu = nn.Linear(encoder_output_dim, self.hparams.dim_latent_space)
        self.fc_logvar = nn.Linear(encoder_output_dim, self.hparams.dim_latent_space)

        # Linear layer to map latent space to decoder input
        self.fc_dec = nn.Linear(self.hparams.dim_latent_space, encoder_output_dim * 4 * 4)

        self.decoder = resnet18_decoder(
            latent_dim=self.hparams.dim_latent_space,
            input_height=dim_input,
            first_conv=False,
            maxpool1=False,
            out_channels=img_channels,
        )

        # # Encoder: Convolutional layers to extract features
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),  # 32 x 32 x 32
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),             # 64 x 16 x 16
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),            # 128 x 8 x 8
        #     nn.ReLU(),
        #     nn.Flatten(),                                                       # 128*8*8 = 8192
        #     nn.Linear(128 * (dim_input // 8) * (dim_input // 8), 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 2 * self.hparams.dim_latent_space),                # Output mean and logvar
        # )

        # self.encoder = nn.Sequential(
        #     nn.Linear(dim_input, 16),
        #     nn.ReLU(),
        #     # 2 times the latent space dimensionality, since we will encode the mean and the variance of each dimension
        #     nn.Linear(16, 2 * self.hparams.dim_latent_space),
        # )

        # Decoder: Transposed Convolutional layers to reconstruct images
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.dim_latent_space, 512),
            nn.ReLU(),
            nn.Linear(512, 128 * (dim_input // 8) * (dim_input // 8)),
            nn.ReLU(),
            nn.Unflatten(1, (128, dim_input // 8, dim_input // 8)),             # 128 x 8 x 8
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 64 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 32 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),  # img_channels x 64 x 64
            nn.Sigmoid(),  # Assuming input images are normalized between 0 and 1
        )

        # self.decoder = nn.Sequential(
        #     nn.Linear(self.hparams.dim_latent_space, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, dim_input),
        # )

        # Prior network: Maps labels to the parameters of the prior distribution
        self.prior_encoder = nn.Sequential(
            nn.Linear(self.hparams.dim_labels, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.hparams.dim_latent_space),
        )

        # self.prior_encoder = nn.Sequential(
        #     nn.Linear(self.hparams.dim_labels, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 2 * self.hparams.dim_latent_space),
        # )

    def train_dataloader(self):
        return torch_data.DataLoader(
            dataset=self.dataset_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=min(os.cpu_count(), 4),
            pin_memory=True,
        )

    # def train_dataloader(self):
    #     return torch_data.DataLoader(
    #         dataset=self.dataset_train,
    #         batch_size=self.hparams.batch_size,
    #         shuffle=True,
    #     )

    def val_dataloader(self):
        return torch_data.DataLoader(
            dataset=self.dataset_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    # def val_dataloader(self):
    #     return torch_data.DataLoader(
    #         dataset=self.dataset_val,
    #         batch_size=self.hparams.batch_size,
    #         shuffle=True,
    #     )

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
        mu, logvar = mu_logvar[:, self.hparams.dim_latent_space:], mu_logvar[:,
                                                                             :self.hparams.dim_latent_space]
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
