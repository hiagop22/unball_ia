import hydra
import torch
import numpy as np
from torch import nn
from omegaconf import DictConfig
from typing import List
from torch.optim.optimizer import Optimizer
from .data import Experience
from .utils import weights_init

device = torch.device('cuda:0')

class BasicBlock(nn.Module):
    def __init__(self, hidden_size: int, activation = nn.Mish):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            activation(),
            nn.Linear(hidden_size,hidden_size),
            activation(),
            )

    def forward(self, x):
        return self.block(x) + x
    


class DDPGActor(nn.Module):
    """
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment = (V,W)
    """
    def __init__(self, obs_size: int, n_actions: int, activation=nn.Mish):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_size, 512),
            activation(),
            BasicBlock(hidden_size=512, activation=activation),
            nn.Linear(512, n_actions),
            nn.Tanh(),
            )

    def forward(self, x):
        x = self.layers(x)

        return x
        
class DDPGCritic(nn.Module):
    def __init__(self, obs_size: int,  n_actions: int, activation=nn.Mish):

        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(obs_size+n_actions, 512),
            activation(),
            BasicBlock(512, activation=activation),
            nn.Linear(512, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.layers(x)
        
        return x
    
class TD3Critic(nn.Module):
    def __init__(self, obs_size: int,  n_actions: int, activation=nn.Mish):

        super().__init__()
        
        self.layers1 = nn.Sequential(
            nn.Linear(obs_size+n_actions, 512),
            activation(),
            BasicBlock(512, activation=activation),
            nn.Linear(512, 1),
        )

        self.layers2 = nn.Sequential(
            nn.Linear(obs_size+n_actions, 512),
            activation(),
            BasicBlock(512, activation=activation),
            nn.Linear(512, 1),
        )

    def forward(self, state, action):
        x1 = torch.cat([state, action], 1)
        x1 = self.layers1(x1)

        x2 = torch.cat([state, action], 1)
        x2 = self.layers2(x2)
        
        return x1, x2
    
    def C1(self, state, action):
        x1 = torch.cat([state, action], 1)
        x1 = self.layers1(x1)
        
        return x1
     
class DDPGAgent():
    def __init__(
        self,
        model_conf: DictConfig,
        memory_conf: DictConfig,
        optimizer_conf: DictConfig,
        scheduler_conf: DictConfig,
        process_state_conf: DictConfig, 
        max_grad_norm: DictConfig,
        gamma: float,
        tau: float,
        **kwargs,
    ):

        self.actor = hydra.utils.instantiate(model_conf.actor).apply(weights_init)
        self.actor = self.actor.to(device)

        self.critic = hydra.utils.instantiate(model_conf.critic).apply(weights_init)
        self.critic = self.critic.to(device)

        self.process_state = hydra.utils.instantiate(process_state_conf)
        self.memory = hydra.utils.instantiate(memory_conf)

        self.scheduler_conf = scheduler_conf
        self.optimizer_conf = optimizer_conf
        
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        self.tau = tau

        self.target_actor = hydra.utils.instantiate(model_conf.actor)
        self.target_actor = self.target_actor.to(device)

        self.target_critic = hydra.utils.instantiate(model_conf.critic)
        self.target_critic = self.target_critic.to(device)

        self.hard_update_target_networks()

        self.actor_opt, self.critic_opt, self.actor_sch, self.critic_sch = self.configure_optimizers()

    def hard_update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
    
    def configure_optimizers(self) -> List[Optimizer]:
        actor_optimizer = hydra.utils.instantiate(
            self.optimizer_conf.actor, params=self.actor.parameters()
        )

        actor_scheduler = hydra.utils.instantiate(
            self.scheduler_conf.actor, optimizer=actor_optimizer
        )

        critic_optimizer = hydra.utils.instantiate(
            self.optimizer_conf.critic, params=self.critic.parameters()
        )

        critic_scheduler = hydra.utils.instantiate(
            self.scheduler_conf.critic, optimizer=critic_optimizer
        )

        return (actor_optimizer, critic_optimizer, actor_scheduler, critic_scheduler)

    def get_action(self, state):
        state = torch.tensor([self.process_state.process(state)], dtype=torch.float).to(device)
        
        self.actor.eval()

        action = self.actor(state)
        self.actor.train()

        action = action.detach().cpu().numpy()
        action = np.clip(action, -1, 1)[0]

        return action

    def train_networks(self, batch_size, step, reward):

        idxs, states, actions, rewards, dones, next_states, _ = self.memory.sample(batch_size)
        
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        next_states = next_states.to(device)

        # critic
        q_values = self.critic(states, actions)
        next_actions = self.target_actor(next_states)
        next_q = self.target_critic(next_states, next_actions)

        q_targets = rewards + (1-dones)*self.gamma * next_q

        critic_loss = nn.MSELoss()(q_targets, q_values)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm.critic)
        self.critic_opt.step()

        # actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm.actor)
        self.actor_opt.step()

        self.soft_update_actor_target_network()
        self.soft_update_critic_target_network()

        output = {
            'critic_loss': critic_loss.cpu().detach().numpy(),
            'actor_loss': actor_loss.cpu().detach().numpy(),
            'actor_lr': self.actor_opt.param_groups[0]['lr'],
            'critic_lr': self.critic_opt.param_groups[0]['lr'],
            'memory_size': len(self.memory),
            'reward': reward,
        }

        return output

    def soft_update_actor_target_network(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def soft_update_critic_target_network(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_experience(self, state, action, reward, done, new_state):

        exp = Experience(self.process_state.process(state), 
                        action, 
                        reward,
                        done, 
                        self.process_state.process(new_state),
                        )
        
        self.memory.append(exp)
       
    def step_schedulers(self, actor_loss, critic_loss):

        self.actor_sch.step(actor_loss)
        self.critic_sch.step(critic_loss)



class TD3Agent(DDPGAgent):
    def __init__(
        self,
        model_conf: DictConfig,
        memory_conf: DictConfig,
        optimizer_conf: DictConfig,
        scheduler_conf: DictConfig,
        process_state_conf: DictConfig, 
        max_grad_norm: DictConfig,
        gamma: float,
        tau: float,
        policy_delay: float,
        target_noise: DictConfig,
        noise_clip: float,
    ):

        super().__init__(model_conf=model_conf,
                         memory_conf=memory_conf,
                         optimizer_conf=optimizer_conf,
                         scheduler_conf=scheduler_conf,
                         process_state_conf=process_state_conf,
                         max_grad_norm=max_grad_norm,
                         gamma=gamma,
                         tau=tau)
        
        self.noise_clip = noise_clip
        self.target_noise = target_noise
        self.policy_delay = policy_delay
        self.actor = hydra.utils.instantiate(model_conf.actor).apply(weights_init)
        self.actor = self.actor.to(device)

        self.critic = hydra.utils.instantiate(model_conf.critic).apply(weights_init)
        self.critic = self.critic.to(device)

        self.target_actor = hydra.utils.instantiate(model_conf.actor).apply(weights_init)
        self.target_actor = self.target_actor.to(device)

        self.target_critic = hydra.utils.instantiate(model_conf.critic).apply(weights_init)
        self.target_critic = self.target_critic.to(device)

        self.hard_update_target_networks()

        self.actor_opt, self.critic_opt, self.actor_sch, self.critic_sch = self.configure_optimizers()

    def train_networks(self, batch_size, step, reward):

        idxs, states, actions, rewards, dones, next_states, _ = self.memory.sample(batch_size)
        
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        next_states = next_states.to(device)

        # critic
        q_values1, q_values2 = self.critic(states, actions)
        
        with torch.no_grad():
            noise = (
                    torch.randn_like(actions) * self.target_noise
                ).clamp(-self.noise_clip, self.noise_clip)
            
            next_actions = self.target_actor(next_states) + noise
            next_actions = torch.clamp(next_actions, -1.0, 1.0)

            next_q1, next_q2 = self.target_critic(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            
            q_targets = rewards + (1.0 - dones) * self.gamma * next_q

        critic_loss = nn.MSELoss()(q_targets, q_values1) + nn.MSELoss()(q_targets, q_values2)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm.critic)
        self.critic_opt.step()

        # actor
        actor_loss = -self.critic.C1(states, self.actor(states)).mean()

        if step % self.policy_delay == 0:
            self.soft_update_actor_target_network()
            self.soft_update_critic_target_network()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm.actor)
            self.actor_opt.step()

        if self.memory.is_per:
            errors = np.abs((q_values1 - q_targets).detach().cpu().numpy())
            for idx, error in zip(idxs, errors):
                self.memory.update(idx, error)

        output = {
            'critic_loss': critic_loss.cpu().detach().numpy(),
            'actor_loss': actor_loss.cpu().detach().numpy(),
            'actor_lr': self.actor_opt.param_groups[0]['lr'],
            'critic_lr': self.critic_opt.param_groups[0]['lr'],
            'memory_size': len(self.memory),
            'reward': reward,
        }

        return output