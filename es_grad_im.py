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

from ES import sepCMAES, sepCEM, sepMCEM
from models import RLNN
from collections import namedtuple
from random_process import GaussianNoise
from memory import Memory, Archive
from samplers import IMSampler
from util import *


Sample = namedtuple('Sample', ('params', 'score',
                               'gens', 'start_pos', 'end_pos'))
Theta = namedtuple('Theta', ('mu', 'cov', 'samples'))
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
            x = F.leaky_relu(self.l1(x))
            x = F.leaky_relu(self.l2(x))
            x = self.max_action * F.tanh(self.l3(x))

        else:
            x = F.leaky_relu(self.n1(self.l1(x)))
            x = F.leaky_relu(self.n2(self.l2(x)))
            x = self.max_action * F.tanh(self.l3(x))

        return x

    def update(self, memory, batch_size, critic, actor_t):

        # Sample replay buffer
        states, _, _, _, _ = memory.sample(batch_size)

        # Compute actor loss
        if args.use_td3:
            actor_loss = -critic(states, self(states))[0].mean()
        else:
            actor_loss = -critic(states, self(states)).mean()

        # Optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), actor_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


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


class CriticTD3(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(CriticTD3, self).__init__(state_dim, action_dim, 1)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n4 = nn.LayerNorm(400)
            self.n5 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip

    def forward(self, x, u):

        if not self.layer_norm:
            x1 = F.relu(self.l1(torch.cat([x, u], 1)))
            x1 = F.relu(self.l2(x1))
            x1 = self.l3(x1)

        else:
            x1 = F.relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x1 = F.relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

        if not self.layer_norm:
            x2 = F.relu(self.l4(torch.cat([x, u], 1)))
            x2 = F.relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x2 = F.relu(self.n4(self.l4(torch.cat([x, u], 1))))
            x2 = F.relu(self.n5(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2

    def update(self, memory, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Select action according to policy and add clipped noise
        noise = np.clip(np.random.normal(0, self.policy_noise, size=(
            batch_size, action_dim)), -self.noise_clip, self.noise_clip)
        n_actions = actor_t(n_states) + FloatTensor(noise)
        n_actions = n_actions.clamp(-max_action, max_action)

        # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
        with torch.no_grad():
            target_Q1, target_Q2 = critic_t(n_states, n_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
            nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
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

            return np.clip(action, -max_action, max_action)

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
    parser.add_argument('--elitism', dest="elitism",  action='store_true')
    parser.add_argument('--n_grad', default=5, type=int)
    parser.add_argument('--sigma_init', default=0.05, type=float)
    parser.add_argument('--damp', default=0.001, type=float)
    parser.add_argument('--mult_noise', dest='mult_noise', action='store_true')

    # Sampler parameters
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--epsilon', type=float, default=0)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--im_buffer', dest='im_buffer', action='store_true')

    # Training parameters
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--mem_size', default=1000000, type=int)
    parser.add_argument('--n_noisy', default=0, type=int)

    # Testing parameters
    parser.add_argument('--filename', default="", type=str)
    parser.add_argument('--n_test', default=1, type=int)

    # misc
    parser.add_argument('--output', default='results', type=str)
    parser.add_argument('--period', default=5000, type=int)
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
    if args.use_td3:
        critic = CriticTD3(state_dim, action_dim, max_action, args)
        critic_t = CriticTD3(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())

    else:
        critic = Critic(state_dim, action_dim, max_action, args)
        critic_t = Critic(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())

    # actor
    actor = Actor(state_dim, action_dim, max_action, args)
    actor_t = Actor(state_dim, action_dim, max_action, args)
    actor_t.load_state_dict(actor.state_dict())
    a_noise = GaussianNoise(action_dim, sigma=args.gauss_sigma)

    if USE_CUDA:
        critic.cuda()
        critic_t.cuda()
        actor.cuda()
        actor_t.cuda()

    # CEM
    es = sepCEM(actor.get_size(), mu_init=actor.get_params(), sigma_init=args.sigma_init, damp=args.damp,
                pop_size=args.pop_size, antithetic=not args.pop_size % 2, parents=args.pop_size // 2, elitism=args.elitism)

    # archives and sampler
    sample_archive = Archive()
    thetas_archive = Archive()
    mu, cov = es.get_distrib_params()
    thetas_archive.add_sample(Theta(mu, cov, []))
    sampler = IMSampler(sample_archive, thetas_archive,  k=1)

    # training
    n_ind = 0
    step_cpt = 0
    curr_gen = 0
    total_steps = 0
    actor_steps = 0
    reused_steps = 0
    df = pd.DataFrame(columns=["total_steps", "average_score",
                               "average_score_rl", "average_score_ea", "best_score"])
    while total_steps < args.max_steps:

        fitness = np.zeros(args.pop_size)
        es_params, n_r, id_r, f_r = sampler.ask(args.pop_size, es)
        print("Reused {} samples".format(n_r))

        # reused samples
        for i in range(n_r):

            # saving reused fitnesses
            if i >= args.n_grad:
                fitness[i] = f_r[i]
            else:
                fitness = np.append(fitness, f_r[i])
                es_params = np.append(es_params, [es_params[i]], axis=0)

            # print reused score
            prGreen('Reused actor fitness:{}'.format(f_r[i]))

            # duplicating samples in buffer
            if args.im_buffer:
                sample_r = sample_archive[id_r[i]]
                start_r = sample_r.start_pos
                end_r = sample_r.end_pos
                memory.repeat(start_r, end_r)
                reused_steps += end_r - start_r

            # adding to archives
            sample_archive.add_gen(id_r[i], curr_gen)
            thetas_archive[curr_gen].samples.append(id_r[i])

        # udpate the rl actors and the critic
        if total_steps > args.start_steps:

            for i in range(args.n_grad):

                # set params
                actor.set_params(es_params[i])
                actor_t.set_params(es_params[i])
                actor.optimizer = torch.optim.Adam(
                    actor.parameters(), lr=args.actor_lr)

                # critic update
                for _ in tqdm(range((actor_steps + reused_steps) // args.n_grad)):
                    critic.update(memory, args.batch_size, actor, critic_t)

                # actor update
                for _ in tqdm(range(actor_steps + reused_steps)):
                    actor.update(memory, args.batch_size,
                                 critic, actor_t)

                # get the params back in the population
                es_params[i] = actor.get_params()

        actor_steps = 0
        reused_steps = 0

        # evaluate noisy actor(s)
        for i in range(args.n_noisy):
            actor.set_params(es_params[i])
            f, steps = evaluate(actor, env, memory=memory, n_episodes=args.n_episodes,
                                render=args.render, noise=a_noise)
            actor_steps += steps
            prCyan('Noisy actor {} fitness:{}'.format(i, f))

        # evaluate all actors
        for i in range(args.pop_size):

            # evaluate actor
            if i >= n_r or i < args.n_grad:
                actor.set_params(es_params[i])
                f, steps = evaluate(actor, env, memory=memory, n_episodes=args.n_episodes,
                                    render=args.render)
                fitness[i] = f

                # adding to archives
                if i >= args.n_grad:
                    sample_archive.add_sample(Sample(es_params[i], f, [curr_gen], total_steps + actor_steps,
                                                     total_steps + actor_steps + steps))
                    thetas_archive[curr_gen].samples.append(n_ind)
                    n_ind += 1
                actor_steps += steps

                # print scores
                prLightPurple('Actor fitness:{}'.format(f))

        # update ea
        es.tell(es_params, fitness)
        mu, cov = es.get_distrib_params()
        thetas_archive.add_sample(Theta(
            mu, cov, []))

        # update step counts
        total_steps += actor_steps
        step_cpt += actor_steps
        curr_gen += 1

        # save stuff
        if step_cpt >= args.period:

            df.to_pickle(args.output + "/log.pkl")
            res = {"total_steps": total_steps,
                   "average_score": np.mean(fitness),
                   "average_score_half": np.mean(np.partition(fitness[:args.pop_size], args.pop_size // 2 - 1)[args.pop_size // 2:]),
                   "average_score_rl": np.mean(fitness[:args.n_grad]),
                   "average_score_ea": np.mean(fitness[args.n_grad:args.pop_size]),
                   "best_score": np.max(fitness),
                   "n_reused": n_r}

            os.makedirs(args.output + "/{}_steps".format(total_steps),
                        exist_ok=True)
            critic.save_model(
                args.output + "/{}_steps".format(total_steps), "critic")
            actor.set_params(es.mu)
            actor.save_model(
                args.output + "/{}_steps".format(total_steps), "actor_mu")
            df = df.append(res, ignore_index=True)
            step_cpt = 0
            print(res)

        print("Total steps", total_steps)
