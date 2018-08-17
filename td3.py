import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models import Actor, CriticTD3

# https://github.com/sfujim/TD3/edit/master/TD3.py
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, memory, args):

        # actor
        self.actor = Actor(state_dim, action_dim, max_action,
                           layer_norm=args.layer_norm)
        self.actor_target = Actor(
            state_dim, action_dim, max_action, layer_norm=args.layer_norm)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr)

        # critic
        self.critic = CriticTD3(state_dim, action_dim,
                                layer_norm=args.layer_norm)
        self.critic_target = CriticTD3(
            state_dim, action_dim, layer_norm=args.layer_norm)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr)

        # cuda
        if torch.cuda.is_available():
            self.actor = self.actor.cuda()
            self.actor_target = self.actor_target.cuda()
            self.critic = self.critic.cuda()
            self.critic_target = self.critic_target.cuda()

        # misc
        self.criterion = nn.MSELoss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.memory = memory

        # hyper-parameters
        self.tau = args.tau
        self.discount = args.discount
        self.batch_size = args.batch_size
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq

    def select_action(self, state, noise=None):
        state = FloatTensor(
            state.reshape(-1, self.state_dim))
        action = self.actor(state).cpu().data.numpy().flatten()

        if noise is not None:
            action += noise.sample()

        return np.clip(action, -self.max_action, self.max_action)

    def train(self, iterations):

        for it in tqdm(range(iterations)):

            # Sample replay buffer
            x, y, u, r, d = self.memory.sample(self.batch_size)
            state = FloatTensor(x)
            next_state = FloatTensor(y)
            action = FloatTensor(u)
            reward = FloatTensor(r)
            done = FloatTensor(1 - d)

            # Select action according to policy and add clipped noise
            noise = np.clip(np.random.normal(0, self.policy_noise, size=(
                self.batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
            next_action = self.actor_target(
                next_state) + FloatTensor(noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_Q1, target_Q2 = self.critic_target(
                    next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * self.discount * target_Q)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = self.criterion(
                current_Q1, target_Q) + self.criterion(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % self.policy_freq == 0:

                # Compute actor loss
                Q1, Q2 = self.critic(state, self.actor(state))
                actor_loss = -Q1.mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

    def load(self, filename):
        self.actor.load_model(filename, "actor")
        self.critic.load_model(filename, "critic")

    def save(self, output):
        self.actor.save_model(output, "actor")
        self.critic.save_model(output, "critic")


class DTD3(object):
    def __init__(self, state_dim, action_dim, max_action, memory, args):

        # misc
        self.criterion = nn.MSELoss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.memory = memory
        self.n = args.n_actor

        # actor
        self.actors = [Actor(state_dim, action_dim, max_action,
                             layer_norm=args.layer_norm) for i in range(self.n)]
        self.actors_target = [Actor(
            state_dim, action_dim, max_action, layer_norm=args.layer_norm) for i in range(self.n)]
        self.actors_optimizer = [torch.optim.Adam(
            self.actors[i].parameters(), lr=args.actor_lr) for i in range(self.n)]

        for i in range(self.n):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())

        # critic
        self.critic = CriticTD3(state_dim, action_dim,
                                layer_norm=args.layer_norm)
        self.critic_target = CriticTD3(
            state_dim, action_dim, layer_norm=args.layer_norm)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr)

        # cuda
        if torch.cuda.is_available():
            for i in range(self.n):
                self.actors[i] = self.actors[i].cuda()
                self.actors_target[i] = self.actors_target[i].cuda()
            self.critic = self.critic.cuda()
            self.critic_target = self.critic_target.cuda()

        # shared memory
        for i in range(self.n):
            self.actors[i].share_memory()
            self.actors_target[i].share_memory()
        self.critic.share_memory()
        self.critic_target.share_memory()

        # hyper-parameters
        self.tau = args.tau
        self.discount = args.discount
        self.batch_size = args.batch_size
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq

    def train(self, iterations, actor_index):

        for it in tqdm(range(iterations)):

            # Sample replay buffer
            states, n_states, actions, rewards, dones = self.memory.sample(
                self.batch_size)

            # Select action according to policy and add clipped noise
            noise = np.clip(np.random.normal(0, self.policy_noise, size=(
                self.batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
            next_action = self.actors_target[actor_index](
                n_states) + FloatTensor(noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_Q1, target_Q2 = self.critic_target(
                    n_states, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards + (1 - dones) * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(states, actions)

            # Compute critic loss
            critic_loss = self.criterion(
                current_Q1, target_Q) + self.criterion(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % self.policy_freq == 0:

                # Compute actor loss
                Q1, Q2 = self.critic(states, self.actors[actor_index](states))
                actor_loss = -Q1.mean()

                # Optimize the actor
                self.actors_optimizer[actor_index].zero_grad()
                actor_loss.backward()
                self.actors_optimizer[actor_index].step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actors[actor_index].parameters(), self.actors_target[actor_index].parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

    def load(self, filename):
        for i in range(self.n):
            self.actors[i].load_model(filename, "actor_" + str(i))
        self.critic.load_model(filename, "critic")

    def save(self, output):
        for i in range(self.n):
            self.actors[i].save_model(output, "actor_" + str(i))
        self.critic.save_model(output, "critic")
