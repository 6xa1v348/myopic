import torch
import torch.nn as nn

from model_helpers import Actor
from training_helpers import *


class Agent(object):
    def __init__(self,
                 obs_shape=(2, 60, 60),
                 num_actions=2,
                 device='cpu',
                 encoder_feature_dim=256,
                 kernel_size=3,
                 projection=False,
                 rand_conv=False,
                 gamma=0.999,
                 max_norm=None):
        if not max_norm:
            max_norm = None
        else:
            max_norm = 0.5 / (1. - gamma)

        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.max_norm = max_norm

        self.kernel_size = kernel_size
        self.projection = projection
        self.rand_conv = rand_conv

        self.actor = Actor(
            obs_shape, num_actions, encoder_feature_dim, kernel_size=kernel_size, rand_conv=rand_conv, max_norm=max_norm
        ).to(device)
        self.fc = nn.Linear(self.actor.encoder.feature_dim, 64).to(device)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def representation(self, obs):
        h = self.actor.encoder(obs)
        if self.projection:
            h = torch.relu(self.fc(h))
        return h
    
    def save(self, f):
        torch.save({
            'actor': self.actor.state_dict(),
            'fc': self.fc.state_dict()
        }, f + '.pt')

    def load(self, f):
        state_dict = torch.load(f + '.pt')
        self.actor.load_state_dict(state_dict['actor'])
        self.fc.load_state_dict(state_dict['fc'])
        print('Loaded agent from %s' % (f + '.pt'))


class PSE(Agent):
    def __init__(self,
                 obs_shape=(2, 60, 60),
                 num_actions=2,
                 device='cpu',
                 encoder_feature_dim=256,
                 kernel_size=3,
                 projection=False,
                 rand_conv=False,
                 lr=0.0026,
                 alpha=5.0,
                 gamma=0.999,
                 temperature=0.5,
                 soft_coupling_temperature=0.01,
                 max_grad_norm=5.0,
                 use_coupling_weights=True,
                 max_norm=None,
                 **kwargs):
        super().__init__(obs_shape=obs_shape, num_actions=num_actions, device=device,
                         encoder_feature_dim=encoder_feature_dim, kernel_size=kernel_size,
                         projection=projection, rand_conv=rand_conv, gamma=gamma, max_norm=max_norm)
        assert self.max_norm is None
        
        self.lr = lr
        self.alpha = alpha
        self.temperature = temperature
        self.soft_coupling_temperature = soft_coupling_temperature
        self.max_grad_norm = max_grad_norm
        self.use_coupling_weights = use_coupling_weights

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr)
        self.train()

    def parameters(self):
        return list(self.actor.parameters()) + list(self.fc.parameters())
    
    def decay_learning_rate(self, epoch):
        lr = self.lr * pow(0.999, epoch - 1)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def representation_alignment_loss(self, optimal_data_tuple):
        obs_1, acs_1, _, _ = optimal_data_tuple[0]
        obs_2, acs_2, _, _ = optimal_data_tuple[1]

        obs_1 = torch.tensor(obs_1).float().to(self.device)
        obs_2 = torch.tensor(obs_2).float().to(self.device)
        representation_1 = self.representation(obs_1)
        representation_2 = self.representation(obs_2)

        similarity_matrix = cosine_similarity(representation_1, representation_2)
        cost_matrix = calculate_action_cost_matrix(acs_1, acs_2)
        metric_values = metric_fixed_point(cost_matrix, gamma=self.gamma)
        metric_values = torch.from_numpy(metric_values).to(self.device)

        loss = soft_simclr_loss(similarity_matrix, metric_values,
                                temperature=self.temperature,
                                soft_coupling_temperature=self.soft_coupling_temperature,
                                use_coupling_weights=self.use_coupling_weights)
        return loss
    
    def update(self, obs, acs, optimal_data_tuple, **kwargs):
        self.train()
        total_loss = 0.
        losses = {}

        imitation_loss = cross_entropy_loss(self, obs, acs)
        losses['imitation_loss'] = imitation_loss.item()
        total_loss += imitation_loss
        if self.alpha > 0:
            alignment_loss = self.representation_alignment_loss(optimal_data_tuple)
            losses['alignment_loss'] = alignment_loss.item()
            total_loss += self.alpha * alignment_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        return losses
    

class MultiPSE(Agent):
    def __init__(self,
                 obs_shape=(2, 60, 60),
                 num_actions=2,
                 device='cpu',
                 encoder_feature_dim=256,
                 kernel_size=3,
                 projection=False,
                 rand_conv=False, 
                 lr=0.0026,
                 alpha=5.0,
                 gamma=0.999,
                 temperature=0.5,
                 soft_coupling_temperature=0.01,
                 max_grad_norm=5.0,
                 max_norm=None,
                 **kwargs):
        super().__init__(obs_shape=obs_shape, num_actions=num_actions, device=device,
                         encoder_feature_dim=encoder_feature_dim, kernel_size=kernel_size,
                         projection=projection, rand_conv=rand_conv, gamma=gamma, max_norm=max_norm)
        assert self.max_norm is None

        self.lr = lr
        self.alpha = alpha
        self.temperature = temperature
        self.soft_coupling_temperature = soft_coupling_temperature
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr)
        self.train()

    def parameters(self):
        return list(self.actor.parameters()) + list(self.fc.parameters())
    
    def decay_learning_rate(self, epoch):
        lr = self.lr * pow(0.999, epoch - 1)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def representation_alignment_loss(self, optimal_data_tuple):
        obs_1, acs_1, _, _ = optimal_data_tuple[0]
        obs_2, acs_2, _, _ = optimal_data_tuple[1]

        obs_1 = torch.tensor(obs_1).float().to(self.device)
        obs_2 = torch.tensor(obs_2).float().to(self.device)
        representation_1 = self.representation(obs_1)
        representation_2 = self.representation(obs_2)

        similarity_matrix = cosine_similarity(representation_1, representation_2)
        cost_matrix = calculate_action_cost_matrix(acs_1, acs_2)
        metric_values = metric_fixed_point(cost_matrix, gamma=self.gamma)
        metric_values = torch.from_numpy(metric_values).to(self.device)

        loss = multi_soft_simclr_loss(similarity_matrix, metric_values,
                                      temperature=self.temperature,
                                      soft_coupling_temperature=self.soft_coupling_temperature)
        return loss
    
    def update(self, obs, acs, optimal_data_tuple, **kwargs):
        self.train()
        total_loss = 0.
        losses = {}

        imitation_loss = cross_entropy_loss(self, obs, acs)
        losses['imitation_loss'] = imitation_loss.item()
        total_loss += imitation_loss
        if self.alpha > 0:
            alignment_loss = self.representation_alignment_loss(optimal_data_tuple)
            losses['alignment_loss'] = alignment_loss.item()
            total_loss += self.alpha * alignment_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        return losses
