# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Orion (https://github.com/xiaomi-mlab/Orion)
# Copyright (c) Xiaomi-mlab. All rights reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn

from .layers import Bottleneck


class DistributionModule(nn.Module):
    """
    A convolutional net that parametrises a diagonal Gaussian distribution.
    """

    def __init__(
        self, in_channels, latent_dim, min_log_sigma, max_log_sigma):
        super().__init__()
        self.compress_dim = in_channels // 2
        self.latent_dim = latent_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma


        self.encoder = DistributionEncoder1DV2(
            in_channels,
            self.compress_dim,
        )

        self.last_conv = nn.Sequential(
            nn.Conv1d(self.compress_dim, out_channels=2 * self.latent_dim, kernel_size=1)
        )


    def forward(self, s_t):
        encoding = self.encoder(s_t.permute(0, 2, 1).float())
        mu_log_sigma = self.last_conv(encoding).permute(0, 2, 1)
        mu = mu_log_sigma[:, :, :self.latent_dim]
        log_sigma = mu_log_sigma[:, :, self.latent_dim:]

        # clip the log_sigma value for numerical stability
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        return mu, log_sigma

class DistributionEncoder2D(nn.Module):
    """Encodes s_t or (s_t, y_{t+1}, ..., y_{t+H}).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            Bottleneck(in_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
        )

    def forward(self, s_t):
        return self.model(s_t)

class DistributionEncoder1D(nn.Module):
    """Encodes s_t or (s_t, y_{t+1}, ..., y_{t+H}).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=in_channels*2, kernel_size=1, stride=1),
            nn.Conv1d(in_channels*2, out_channels=in_channels*2, kernel_size=1, stride=1),
            nn.Conv1d(in_channels*2, out_channels=in_channels, kernel_size=1, stride=1),
            nn.Conv1d(in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        )

    def forward(self, s_t):
        return self.model(s_t)

class DistributionEncoder1DV2(nn.Module):
    """Encodes s_t or (s_t, y_{t+1}, ..., y_{t+H}).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels=in_channels * 2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels * 2, out_channels=in_channels * 2, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(in_channels * 2, out_channels=out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, s_t):
        s_t = self.relu(self.conv1(s_t))
        s_t = self.relu(self.conv2(s_t))
        s_t = self.conv3(s_t)

        return s_t

class DistributionDecoder1DV2(nn.Module):
    """Decodes sample to future states.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels=in_channels * 8, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels * 8, out_channels=in_channels * 8, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(in_channels * 8, out_channels=out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_t):
        f_t = self.relu(self.conv1(f_t))
        f_t = self.relu(self.conv2(f_t))
        f_t = self.conv3(f_t)

        return f_t

class PredictModel(nn.Module):
    """predict future states with rnn.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_channels, num_layers=num_layers)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels*2)
        self.linear2 = nn.Linear(hidden_channels*2, hidden_channels*4)
        self.linear3 = nn.Linear(hidden_channels*4, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x , h):
        x, h = self.gru(x, h)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ProbabilisticLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, output, valid_mask):
        present_mu = output['present_mu']
        present_log_sigma = output['present_log_sigma']
        future_mu = output['future_mu']
        future_log_sigma = output['future_log_sigma']

        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
                present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                    2 * var_present)
        )
        kl_div = kl_div * valid_mask.any(dim=-1).unsqueeze(-1).unsqueeze(-1)
        kl_loss = torch.mean(torch.sum(kl_div, dim=-1)) * self.loss_weight

        return kl_loss

class VAEPEDecoder(nn.Module):
    """VAE to replace MLP for input coordinates to 3d positional encoding.
    """
    def __init__(self, llm_hidden_dim, latent_dim, with_cur=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.present_distribution_in_channels = llm_hidden_dim
        self.future_distribution_in_channels = llm_hidden_dim+2
        self.MIN_LOG_SIGMA = -5.0
        self.MAX_LOG_SIGMA = 5.0
        
        self.present_distribution = DistributionModule(
                    self.present_distribution_in_channels,
                    self.latent_dim,
                    min_log_sigma=self.MIN_LOG_SIGMA,
                    max_log_sigma=self.MAX_LOG_SIGMA,
                )

        self.future_distribution = DistributionModule(
            self.future_distribution_in_channels,
            self.latent_dim,
            min_log_sigma=self.MIN_LOG_SIGMA,
            max_log_sigma=self.MAX_LOG_SIGMA,
        )
        
        self.with_cur = with_cur
        # a mlp decoder to decode the sampled latent vector to self.ego_fut_mode*2 coordinates 
        if with_cur:
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim + llm_hidden_dim , self.latent_dim*2),
                nn.ReLU(inplace=True),
                nn.Linear(self.latent_dim*2, self.latent_dim*4),
                nn.ReLU(inplace=True),
                nn.Linear(self.latent_dim*4, 4), 
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim*2),
                nn.ReLU(inplace=True),
                nn.Linear(self.latent_dim*2, self.latent_dim*4),
                nn.ReLU(inplace=True),
                nn.Linear(self.latent_dim*4, 2),
            )
        
        self.loss_vae_gen = ProbabilisticLoss(loss_weight=1.0)



    def forward(self, pe_embedding, gt_coords=None, noise=None):
        """
        pe_embedding: (B, num_pos, llm_hidden) the 3d positional encoding from input coordinates
        gt_coords: (B,num_pos, 2) the ground truth coordinates for the next step
        noise: (B,num_pos, latent_dim) the noise for sampling, if None, sample from standard normal distribution
        """
        distribution_comp = {}
        training = False

        if gt_coords is not None:
            training = True

        present_features = pe_embedding # (B, num_pos, llm_hidden)

        if training:
            future_distribution_inputs = gt_coords
             # (B, num_pos, 2   )

            sample, output_distribution = self.distribution_forward(
                            present_features, future_distribution_inputs, noise
                        ) # sample has shape (B, num_pos,, latent_dim)
            distribution_comp = {**distribution_comp, **output_distribution}
            if self.with_cur:
                # concatenate the pe_embedding with the sample
                sample = torch.cat([sample, pe_embedding], dim=-1) # (B, num_pos, latent_dim + llm_hidden)
            decoded_coords = self.decoder(sample)[:,:,:2] # (B, num_sample,  2)
            loss_vae_gen = self.loss_vae_gen(distribution_comp, valid_mask=torch.ones_like(gt_coords[:, :1]))

            return decoded_coords, loss_vae_gen
        else:
            sample, output_distribution = self.distribution_forward(
                                present_features, None, noise
                            )

            if self.with_cur:
                # concatenate the pe_embedding with the sample
                sample = torch.cat([sample, pe_embedding], dim=-1) # (B, num_pos, latent_dim + llm_hidden)
            decoded_coords = self.decoder(sample)[:,:,:2] # (B, num,  2)
            return decoded_coords

    
    def distribution_forward(self, present_features, future_distribution_inputs=None, noise=None):

        b = present_features.shape[0]

        present_mu, present_log_sigma = self.present_distribution(present_features)

        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            future_features = torch.cat([present_features, future_distribution_inputs], dim=2)
            future_mu, future_log_sigma = self.future_distribution(future_features)

        if noise is None:
            if self.training:
                noise = torch.randn_like(present_mu)
            else:
                noise = torch.randn_like(present_mu)
        if self.training:
            mu = future_mu
            sigma = torch.exp(future_log_sigma)
        else:
            mu = present_mu
            sigma = torch.exp(present_log_sigma)
        sample = mu + sigma * noise # shape (6, B, latent_dim)
        sample = sample # (B, 6, latent_dim)

        output_distribution = {
            'present_mu': present_mu,
            'present_log_sigma': present_log_sigma,
            'future_mu': future_mu,
            'future_log_sigma': future_log_sigma,
        }

        return sample, output_distribution





