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

    """
        Actor head for continuous action policy. 
        It takes the latent state vector produced by Multi-modal ViT model (in_dim) and outputs a Gaussian over continuous control commands [thrust, roll_rate, pitch_rate, yaw_rate], 
        allowing stochastic exploration while training and givng a smooth deterministic mean policy at test time.  
    """
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
        h = self.net(x)                 #obtain hidden features: [1, 256]
        mu = self.mu(h)                     #indicates which action values are best in the current state         
                                            #produces mean vector of Gaussian actions: [1, 4], action_dim = 4 ([thrust, roll_rate, pitch_rate, yaw_rate])
        log_std = self.log_std.clamp(-5,2)  #lower (-5) and upper (2) bound of exploration; log_std shape:[4]
        std = log_std.exp().expand_as(mu)   #control exploration i.e. determine how much exploratory/uncertain to be around that action
                                            #shape: [1, 4]; std = exp(-5) = 0.0067 (very low but non-zero exploration) & std = exp(2) = 7.39 (uuper bound on exploration) and then expand it to match shape of mu [1,4] 
                                            #i.e. replicate the vector (mu) over the batch dimension to match the shape of mean
        return mu, std                  #mean (mu) and standard deviation (std) of Gaussian policy for each action dimension and each action
    
class CriticHead(nn.Module):

    """
        Critic head for value estination.
        It takes the input feature vector from Multi-modal ViT model, and outputs a single scalar value, estimating how good the current state is.
        Used by RL algorithm for computing advantage and value loss.
    """
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

    """
        Actor-Critic model with multi-modal Vision Transformer encoder.

        Combines image and IMU inputs via a ViT-based encoder to produce a shared latent state, then:
            -> the Actor head outputs a Gaussian policy (mu,std) and sampling logic used in "act" to select actions and compute their log probabilities
            -> the Critic head outputs state values used in both "act" and "evaluate_actions" for value/advantage estimation and training
    """
    
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
        """
            Function call to sample an action. 
            Based on the current observation (image+IMU), it builds a latent state with ViT encoder,
            samples a continuous action from a Gaussian policy, computes its log probability, and
            evaluates the state value with the critic.
        """
        z = self.vit_encoder.forward_features(img,imu)      #get a latent represntation from image+IMU
        mu, std = self.actor(z)
        dist = Normal(mu, std)                              #each state in the batch gets its own normal distribution over actions
        action = dist.rsample()                             #samples an action from the distribution using reparameterization, keeps sampling operation differentiable w.r.t mu and std so gradients can flow through the policy
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(z)
        return action, log_prob, value                      #returns: the actual continuous actions to send to the environment, 
                                                            #log probabilities of those actions under the current policy, 
                                                            #critic's estimate of how good each state is (for baseline and advantage)
    
    def evaluate_actions(self,img,imu,action):
        """
            Function call to evaluate given actions under the current policy (rather than sampling new ones)
        """
        z = self.vit_encoder.forward_features(img, imu)
        mu, std = self.actor(z)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(z)
        return log_prob, entropy, value


