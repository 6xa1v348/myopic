import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from sac_ae import Actor, Critic


class BisimAgent(object):
    """Bisimulation Agent."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        actor_lr=1e-3,
        actor_action_max=1.,
        actor_log_std_min=-10.,
        actor_log_std_max=2.,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        encoder_max_norm=None,
        **kwargs):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq

        if not encoder_max_norm: encoder_max_norm = None
        assert encoder_max_norm is None

        self.actor = Actor(obs_shape=obs_shape,
                           action_shape=action_shape,
                           hidden_dim=hidden_dim,
                           encoder_feature_dim=encoder_feature_dim,
                           action_max=actor_action_max,
                           num_layers=num_layers,
                           log_std_min=actor_log_std_min,
                           log_std_max=actor_log_std_max,
                           encoder_max_norm=encoder_max_norm
        ).to(device)
        self.critic = Critic(obs_shape=obs_shape,
                             action_shape=action_shape,
                             hidden_dim=hidden_dim,
                             encoder_feature_dim=encoder_feature_dim,
                             num_layers=num_layers,
                             encoder_max_norm=encoder_max_norm
        ).to(device)
        self.critic_target = Critic(obs_shape=obs_shape,
                                    action_shape=action_shape,
                                    hidden_dim=hidden_dim,
                                    encoder_feature_dim=encoder_feature_dim,
                                    num_layers=num_layers,
                                    encoder_max_norm=encoder_max_norm
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr)
        
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=encoder_lr)
        
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, obs, torch_mode=False):
        with torch.no_grad():
            if not torch_mode:
                obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            if torch_mode:
                return mu
            return mu.cpu().data.numpy().flatten()
        
    def sample_action(self, obs, multiproc=False, torch_mode=False):
        with torch.no_grad():
            if not torch_mode:
                if multiproc:
                    obs = torch.FloatTensor(obs).to(self.device)
                else:
                    obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            if torch_mode:
                return pi
            if not multiproc:
                return pi.cpu().data.numpy().flatten()
            else:
                return pi.cpu().data.numpy()
            
    def compute_norm(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            h = self.critic.encoder(obs)
            norm = h.norm(dim=-1).mean()
            return norm.cpu().data.numpy().flatten()
        
    def update_critic(self, obs, action, reward, next_obs, not_done, L):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=False)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss)
        L.log('train_actor/target_entropy', self.target_entropy)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean())

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss)
        L.log('train_alpha/value', self.alpha)
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_encoder(self, obs, reward, L):
        h = self.critic.encoder(obs)

        similarity_matrix = torch.cdist(h, h, p=2)
        cost_matrix = torch.cdist(reward, reward, p=1)
        loss = F.mse_loss(similarity_matrix, target=cost_matrix)
        L.log('train_ae/loss', loss)
        return loss
    
    def update(self, replay_buffer, L, step):
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample()
        L.log('train/batch_reward', reward.mean())

        self.update_critic(obs, action, reward, next_obs, not_done, L)
        encoder_loss = self.update_encoder(obs, reward, L)
        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )
    
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )