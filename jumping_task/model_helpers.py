import random
import warnings
# warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RandomizingConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            bias=False)

        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        nn.init.xavier_normal_(self.conv.weight)
        x = self.conv(x)
        return x
    

class RandConv(nn.Module):
    def __init__(self, channels=1, kernel_size=3, alpha=0.1):
        super().__init__()
        self._alpha = alpha
        self._channels = channels
        self.rand_conv = RandomizingConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size)
    
    def rand_output(self, x):
        assert x.size(1) % self._channels == 0
        num_splits = x.size(1) // self._channels
        x = rearrange(x, 'b (n k) h w -> (b n) k h w', n=num_splits, k=self._channels)
        x = self.rand_conv(x)
        x = rearrange(x, '(b n) k h w -> b (n k) h w', n=num_splits, k=self._channels)
        return x
    
    def forward(self, x):
        if random.random() < self._alpha:
            return x
        else:
            return self.rand_output(x)


class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim=256, kernel_size=3, rand_conv=True, max_norm=None):
        super().__init__()
        self.obs_shape = obs_shape
        self.rand_conv = RandConv(kernel_size=kernel_size) if rand_conv else None
        self.max_norm = max_norm
        self.feature_dim = feature_dim
        self.model = nn.Sequential(
            nn.ConstantPad2d((2, 2, 2, 2), 0),
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.ConstantPad2d((1, 2, 1, 2), 0),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConstantPad2d((1, 1, 1, 1), 0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        if self.rand_conv:
            x = self.rand_conv(x)
        x = self.model(x)
        if self.max_norm is not None:
            x = self.normalize(x)
        return x
    
    def normalize(self, x):
        norms = x.norm(dim=-1)
        norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
        x = x / norm_to_max
        return x
    

class Actor(nn.Module):
    def __init__(self, obs_shape, num_actions, encoder_feature_dim, kernel_size=3, rand_conv=True, max_norm=None):
        super().__init__()
        self.encoder = Encoder(
            obs_shape, encoder_feature_dim, kernel_size=kernel_size, rand_conv=rand_conv, max_norm=max_norm)
        self.head = nn.Linear(self.encoder.feature_dim, num_actions)

    def forward(self, x):
        x = self.encoder(x)
        pi = self.head(x)
        return pi


class DeterministicTransitionModel(nn.Module):
    def __init__(self, encoder_feature_dim, num_actions, hidden_dim=512, max_norm=None):
        super().__init__()
        self.fc = nn.Linear(encoder_feature_dim + num_actions, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.mu = nn.Linear(hidden_dim, encoder_feature_dim)
        self.num_actions = num_actions
        self.max_norm = max_norm

    def forward(self, h, acs):
        assert h.size(0) == acs.size(0)
        acs = F.one_hot(acs, num_classes=self.num_actions)
        x = self.fc(torch.cat([h, acs], dim=-1))
        x = torch.relu(self.ln(x))
        mu = torch.relu(self.mu(x))
        if self.max_norm is not None:
            mu = self.normalize(mu)
        return mu
    
    def normalize(self, x):
        norms = x.norm(dim=-1)
        norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
        x = x / norm_to_max
        return x
