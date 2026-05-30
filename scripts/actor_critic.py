import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


"""
Actor-Critic Heads for MultiModalViT (UZH-FPV Drone Racing)
===========================================================

Contains:
    - ActorHead:                MLP mapping MultiModalViT CLS embedding → 4D continuous action
                                (thrust, roll rate, pitch rate, yaw rate) with Gaussian policy (mu, std).
    - CriticHead:               MLP mapping CLS embedding -> scalar state value V(s).
    - MultiModalViTActorCritic: wraps a pretrained MultiModalViT encoder together with actor and critic heads, yielding an RL-ready model for PPO-style training
                                in drone racing environments.

This module does not handle training logic; it only defines the policy/value architecture on top of the multimodal ViT backbone.
"""


class ActorHead(nn.Module):
    def __init__(self, in_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim,action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self,x):
        h = self.net(x)
        mu = self.mu(h)
        log_std = self.log_std.clamp(-5,2)
        std = log_std.exp().expand_as(mu)
        return mu, std
    
class CriticHead(nn.Module):
    def __init__(self,in_dim,hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )        
    
    def forward(self,x):
        return self.net(x)
    
class MultiModalViTActorCritic(nn.Module):
    def __init__(self, vit_encoder, embed_dim=256, action_dim=4):
        super().__init__()
        self.vit_encoder = vit_encoder
        self.actor = ActorHead(embed_dim, action_dim)
        self.critic = CriticHead(embed_dim)
    
    def forward(self, img, imu):
        z = self.vit_encoder.forward_features(img,imu)

        mu, std = self.actor(z)
        value = self.critic(z)

        return mu, std, value
    
    def act(self, img, imu):
        z = self.vit_encoder.forward_features(img,imu)
        mu, std = self.actor(z)
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(z)
        return action, log_prob, value
    
    def evaluate_actions(self,img,imu,action):
        z = self.vit_encoder.forward_features(img, imu)
        mu, std = self.actor(z)

        dist = Normal(mu, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(z)

        return log_prob, entropy, value


