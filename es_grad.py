from copy import deepcopy
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import cma
import pandas as pd

import gym
import gym.spaces
import numpy as np
from tqdm import tqdm

from ES import CMAES
from models import RLNN
from random_process import GaussianNoise
from memory import Memory
from util import *

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class Actor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, args):
        super(Actor, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = args.layer_norm

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.actor_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x):

        if not self.layer_norm:
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = self.max_action * F.tanh(self.l3(x))

        else:
            x = F.relu(self.n1(self.l1(x)))
            x = F.relu(self.n2(self.l2(x)))
            x = self.max_action * F.tanh(self.l3(x))

        return x

    def update(self, memory, batch_size, critic, actor_t):

        # Sample replay buffer
        states, _, _, _, _ = memory.sample(batch_size)

        # Compute actor loss
        actor_loss = -critic(states, self(states)).mean()

        # Optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), actor_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


def evaluate(actor, env, memory=None, n_episodes=1, random=False, noise=None, render=False):
    """
    Computes the score of an actor on a given number of runs
    """

    if not random:
        def policy(state):
            state = FloatTensor(state.reshape(-1))
            action = actor(state).cpu().data.numpy().flatten()

            if noise is not None:
                action += noise.sample()

            return np.clip(action, -1, 1)

    else:
        def policy(state):
            return env.action_space.sample()

    scores = []
    steps = 0

    for _ in range(n_episodes):

        score = 0
        obs = deepcopy(env.reset())
        done = False

        while not done:

            # get next action and act
            action = policy(obs)
            n_obs, reward, done, _ = env.step(action)
            done_bool = 0 if steps + \
                1 == env._max_episode_steps else float(done)
            score += reward
            steps += 1

            # adding in memory
            if memory is not None:
                memory.add((obs, n_obs, action, reward, done_bool))
            obs = n_obs

            # render if needed
            if render:
                env.render()

            # reset when done
            if done:
                env.reset()

        scores.append(score)

    return np.mean(scores), steps


class Critic(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(Critic, self).__init__(state_dim, action_dim, 1)

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = args.layer_norm

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x, u):

        if not self.layer_norm:
            x = F.relu(self.l1(torch.cat([x, u], 1)))
            x = F.relu(self.l2(x))
            x = self.l3(x)

        else:
            x = F.relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x = F.relu(self.n2(self.l2(x)))
            x = self.l3(x)

        return x

    def update(self, memory, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Q target = reward + discount * Q(next_state, pi(next_state))
        with torch.no_grad():
            target_Q = critic_t(n_states, actor_t(n_states))
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimate
        current_Q = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str,)
    parser.add_argument('--env', default='HalfCheetah-v2', type=str)
    parser.add_argument('--start_steps', default=10000, type=int)

    # DDPG parameters
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--reward_scale', default=1., type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true')

    # TD3 parameters
    parser.add_argument('--use_td3', dest='use_td3', action='store_true')
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_freq', default=2, type=int)

    # Gaussian noise parameters
    parser.add_argument('--gauss_sigma', default=0.1, type=float)

    # OU process parameters
    parser.add_argument('--ou_noise', dest='ou_noise', action='store_true')
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)

    # ES parameters
    parser.add_argument('--pop_size', default=10, type=int)
    parser.add_argument('--p_steps', default=0.5, type=float)
    parser.add_argument('--n_steps', default=1000, type=int)

    # Training parameters
    parser.add_argument('--n_actor', default=1, type=int)
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--mem_size', default=1000000, type=int)

    # Testing parameters
    parser.add_argument('--filename', default="", type=str)
    parser.add_argument('--n_test', default=1, type=int)

    # misc
    parser.add_argument('--output', default='results', type=str)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--render', dest='render', action='store_true')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    with open(args.output + "/parameters.txt", 'w') as file:
        for key, value in vars(args).items():
            file.write("{} = {}\n".format(key, value))

    # environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # memory
    memory = Memory(args.mem_size, state_dim, action_dim)

    # critic
    critic = Critic(state_dim, action_dim, max_action, args)
    # critic.load_model(
    #     "results/ddpg_5/hc/HalfCheetah-v2-run1/1000000_steps", "critic")
    critic_t = Critic(state_dim, action_dim, max_action, args)
    critic_t.load_state_dict(critic.state_dict())

    # actor
    actor = Actor(state_dim, action_dim, max_action, args)
    actor_t = Actor(state_dim, action_dim, max_action, args)
    actor_t.load_state_dict(actor.state_dict())

    if USE_CUDA:
        critic.cuda()
        critic_t.cuda()
        actor.cuda()
        actor_t.cuda()

    # es
    # es = cma.CMAEvolutionStrategy(
    #     actor.get_params(), 0.01, inopts={"CMA_diagonal": True, "popsize": args.pop_size})
    es = CMAES(actor.get_size(), sigma_init=0.05,
               pop_size=args.pop_size, antithetic=False, full=False, rank_fitness=True)

    # training
    total_steps = 0
    df = pd.DataFrame(columns=["total_steps", "average_score", "best_score"])

    while total_steps < args.max_steps:

        actors_params = es.ask(args.pop_size)
        fitness = []

        # udpate some actors
        fs_b = [None for _ in range(args.pop_size)]
        for i in range(args.pop_size):
            actor.set_params(actors_params[i])
            actor.optimizer = torch.optim.Adam(
                actor.parameters(), lr=args.actor_lr)

            u = np.random.rand()
            if u < args.p_steps and total_steps > args.start_steps:

                # evaluate before
                f, _ = evaluate(actor, env, memory=None, n_episodes=args.n_episodes,
                                render=False)
                fs_b[i] = f

                # do some gradient descent steps
                for _ in range(args.n_steps):
                    actor.update(memory, args.batch_size, critic, actor_t)

                # set new parameters
                actors_params[i] = actor.get_params()

                # print scores
                prLightPurple('EA actor fitness before:{}'.format(f))

        # evaluate all actors
        actor_steps = 0
        fs = []
        for actor_params in actors_params:

            actor.set_params(actor_params)
            f, steps = evaluate(actor, env, memory=memory, n_episodes=args.n_episodes,
                                render=args.render)
            actor_steps += steps
            fs.append(f)
            # / ! \ signe
            fitness.append(-f)

            # print scores
            prLightPurple('EA actor fitness after:{}'.format(f))

        # update critic
        for _ in tqdm(range(actor_steps)):
            critic.update(memory, args.batch_size, actor_t, critic_t)

        # update es and agent
        es.tell(actors_params, fitness)
        # save stuff
        df.to_pickle(args.output + "/log.pkl")

        # saving scores
        best_score = f
        res = {"total_steps": total_steps,
               "average_score": np.mean(fs),
               "best_score": np.max(fs)}
        for i in range(args.pop_size):
            res["score_before_{}".format(i)] = fs_b[i]
            res["score_after_{}".format(i)] = fs[i]
        df = df.append(res, ignore_index=True)

        total_steps += actor_steps
        print("Total steps", total_steps)
