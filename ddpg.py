import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models import ActorERL as Actor
from models import CriticERL as Critic

# https://github.com/sfujim/TD3/edit/master/OurDDPG.py
# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, memory, args):

        # actor
        self.actor = Actor(state_dim, action_dim, max_action, init=True)
        self.actor_target = Actor(
            state_dim, action_dim, max_action, init=True)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr)

        # crtic
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
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

    def show_lr(self):
        print(self.actor_optimizer.state_dict())

    def select_action(self, state, noise=None):
        state = FloatTensor(
            state.reshape(-1, self.state_dim))
        action = self.actor(state).cpu().data.numpy().flatten()

        if noise is not None:
            action += noise.sample()

        return np.clip(action, -self.max_action, self.max_action)

    def train(self, iterations):

        for _ in tqdm(range(iterations)):

            # Sample replay buffer
            x, y, u, r, d = self.memory.sample(self.batch_size)
            state = FloatTensor(x)
            action = FloatTensor(u)
            next_state = FloatTensor(y)
            done = FloatTensor(1 - d)
            reward = FloatTensor(r)

            # Q target = reward + discount * Q(next_state, pi(next_state))
            with torch.no_grad():
                target_Q = self.critic_target(
                    next_state, self.actor_target(next_state))
                target_Q = reward + (done * self.discount * target_Q)

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = self.criterion(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

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

    def train_critic(self, iterations):

        for _ in tqdm(range(iterations)):

            # Sample replay buffer
            x, y, u, r, d = self.memory.sample(self.batch_size)
            state = FloatTensor(x)
            action = FloatTensor(u)
            next_state = FloatTensor(y)
            done = FloatTensor(1 - d)
            reward = FloatTensor(r)

            # Q target = reward + discount * Q(next_state, pi(next_state))
            with torch.no_grad():
                target_Q = self.critic_target(
                    next_state, self.actor_target(next_state))
                target_Q = reward + (done * self.discount * target_Q)

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = self.criterion(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

    def load(self, filename):
        self.actor.load_model(filename, "actor")
        self.critic.load_model(filename, "critic")

    def save(self, output):
        self.actor.save_model(output, "actor")
        self.critic.save_model(output, "critic")
