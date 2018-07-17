from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from RL.DDPG.model import Actor, Critic
from RL.DDPG.random_process import OrnsteinUhlenbeckProcess
from RL.DDPG.util import *

criterion = nn.MSELoss()


class DDPG(object):
    def __init__(self, nb_states, nb_actions, memory, args):

        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        self.actor = Actor(self.nb_states, self.nb_actions)
        self.actor_target = Actor(self.nb_states, self.nb_actions)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = Critic(self.nb_states, self.nb_actions)
        self.critic_target = Critic(self.nb_states, self.nb_actions)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)

        # scale the actor parameters
        self.actor.scale_params(0.1)

        # Make sure target is with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = memory
        self.random_process = OrnsteinUhlenbeckProcess(
            nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # hyper-parameters
        self.reward_scale = 1.
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.discount

        if USE_CUDA:
            self.cuda()

    def train(self):

        # Sample batch
        batch = self.memory.sample(self.batch_size)
        state_batch = to_tensor(batch.states).view(-1, self.nb_states)
        action_batch = to_tensor(batch.actions).view(-1, self.nb_actions)
        reward_batch = to_tensor(batch.rewards).view(-1, 1)
        next_state_batch = to_tensor(
            batch.next_states).view(-1, self.nb_states)
        done_batch = to_tensor(batch.dones).view(-1, 1)

        # Prepare for the target q batch
        next_q_values = self.critic_target(
            [next_state_batch, self.actor_target(next_state_batch)]).detach()

        target_q_batch = self.reward_scale * reward_batch + \
            self.discount * (1. - done_batch) * next_q_values

        # Critic update
        self.critic_optim.zero_grad()

        q_batch = self.critic([state_batch, action_batch])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()

        self.critic_optim.step()

        # Actor update
        self.actor_optim.zero_grad()

        policy_loss = -1. * self.critic([state_batch, self.actor(state_batch)])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def get_actor_size(self):
        return np.shape(self.get_actor_params())[0]

    def get_actor(self):
        return deepcopy(self.actor)

    def set_actor(self, actor):
        self.actor = actor

    def get_critic(self):
        return deepcopy(self.critic)

    def get_actor_params(self):
        return self.actor.get_params()

    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions)
        return action

    def select_action(self, s_t, noise=True):
        """
        Returns action after seeing state 
        """
        action = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0)
        if noise:
            action += self.is_training * \
                max(self.epsilon, 0) * self.random_process.sample()
        action = np.clip(action, -1., 1.)

        return action

    def reset(self):
        self.random_process.reset_states()

    def load_model(self, filename):
        self.actor.load_model(filename)
        self.critic.load_model(filename)

    def save_model(self, output):
        self.actor.save_model(output)
        self.critic.save_model(output)

    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
