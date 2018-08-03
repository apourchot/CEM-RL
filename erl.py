# !/usr/bin/env python3
import argparse
from copy import deepcopy

import gym
import numpy as np
import pandas as pd

from GA import GA
from models import ActorERL as Actor
from ddpg import DDPG
from td3 import TD3
from random_process import *
from util import *
from memory import Memory

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


def evaluate(actor, n_episodes=1, random=False, train=False, noise=None, render=False, add_memory=True):
    """
    Computes the score of an actor on a given number of runs
    """

    if not random:
        def policy(state):
            state = FloatTensor(
                state.reshape(-1, state_dim))
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
            n_obs, reward, done, info = env.step(action)
            score += reward
            steps += 1

            # adding in memory
            if add_memory:
                memory.add((obs, n_obs, action, reward, done))
            obs = n_obs

            # render if needed
            if render:
                env.render()

            # reset when done
            if done:
                env.reset()

        # training agent
        if train:
            agent.train(steps)

        scores.append(score)

    return np.mean(scores), steps


def train_ea(n_episodes=1, debug=False, render=False):
    """
    Train the EA process
    """

    batch_steps = 0
    actor = Actor(state_dim, action_dim, max_action)
    if USE_CUDA:
        actor.cuda()
    actors_params = ea.ask()
    fitness = []

    # evaluate all actors
    for actor_params in actors_params:

        actor.set_params(actor_params)
        f, steps = evaluate(actor, n_episodes=n_episodes,
                            render=render, train=True)
        batch_steps += steps
        fitness.append(f)

        # print scores
        if debug:
            prLightPurple('EA actor fitness:{}'.format(f))

    # update ea
    ea.tell(fitness)

    return batch_steps


def train_rl(n_episodes=1, debug=False, render=False, random=False):
    """
    Train the deep RL agent
    """

    # noisy ddpg agent
    f, steps = evaluate(agent.actor, n_episodes=n_episodes,
                        noise=a_noise, render=render, random=random, train=True)

    # print score
    if debug:
        prCyan('noisy RL agent fitness:{}'.format(f))

    # evaluate ddpg agent
    f, _ = evaluate(agent.actor, n_episodes=n_episodes,
                    render=render, add_memory=False)

    # print score
    if debug:
        prRed('RL agent fitness:{}'.format(f))

    return steps, f


def train(n_gen, n_episodes, omega, output=None, debug=False, render=False):
    """
    Train the whole process
    """

    total_steps = 0
    df = pd.DataFrame(columns=["total_steps", "best_score"])

    for n in range(n_gen):

        random = total_steps < args.start_steps
        steps_ea = train_ea(n_episodes=n_episodes,
                            debug=debug, render=render)
        steps_rl, f = train_rl(n_episodes=n_episodes, debug=debug,
                               render=render, random=random)
        total_steps += steps_rl + steps_ea

        # saving model and scores
        agent.save(output)
        df.to_pickle(output + "/log.pkl")

        # saving scores
        best_score = f
        df = df.append({"total_steps": total_steps,
                        "best_score": best_score}, ignore_index=True)

        # printing generation resume
        if debug:
            prPurple('Generation#{}: Total steps:{}\n'.format(n, total_steps))

        # adding agent to population
        if (n + 1) % omega == 0 and args.pop_size > 0:
            if debug:
                prRed('Transfered RL agent into pop')
            ea.add_ind(agent.actor.get_params(), f)


def test(n_test, filename, debug=False, render=False):
    """
    Test an agent
    """

    # load weights
    actor = Actor(state_dim, action_dim, max_action)
    if USE_CUDA:
        actor.cuda()
    actor.load_model(filename)

    # evaluate
    f, _ = evaluate(actor, n_episodes=n_test,
                    noise=False, render=render)

    # print scores
    if debug:
        prLightPurple(
            'Average fitness:{}'.format(f))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str,)
    parser.add_argument('--env', default='HalfCheetah-v2', type=str)
    parser.add_argument('--start_steps', default=10000, type=int)

    # DDPG parameters
    parser.add_argument('--actor_lr', default=0.00005, type=float)
    parser.add_argument('--critic_lr', default=0.0005, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--reward_scale', default=1., type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true')

    # TD3 parameters
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_freq', default=2, type=int)

    # EA parameters
    parser.add_argument('--pop_size', default=10, type=int)
    parser.add_argument('--elite_frac', default=0.1, type=float)
    parser.add_argument('--mut_rate', default=0.9, type=float)
    parser.add_argument('--mut_amp', default=0.1, type=float)

    # Gaussian noise parameters
    parser.add_argument('--gauss_sigma', default=0.1, type=float)

    # Action noise OU process parameters
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)

    # Parameter noise parameters
    parser.add_argument('--param_init_std', default=0.01, type=float)
    parser.add_argument('--param_scale', default=0.2, type=float)
    parser.add_argument('--param_adapt', default=1.01, type=float)

    # Training parameters
    parser.add_argument('--n_gen', default=100, type=int)
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--omega', default=10, type=int)
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

    # The environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # Random seed
    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    # replay buffer
    memory = Memory(args.mem_size)

    # DDPG agent
    a_noise = OrnsteinUhlenbeckProcess(
        action_dim, mu=args.ou_mu, theta=args.ou_theta, sigma=args.ou_sigma)
    agent = TD3(state_dim, action_dim, max_action, memory, args)

    # EA process
    ea = GA(agent.actor.get_size(), pop_size=args.pop_size, mut_amp=args.mut_amp, mut_rate=args.mut_rate,
            elite_frac=args.elite_frac, generator=lambda: Actor(state_dim, action_dim, max_action).get_params())

    if args.mode == 'train':
        train(n_gen=args.n_gen, n_episodes=args.n_episodes, omega=args.omega,
              output=args.output, debug=args.debug, render=args.render)

    elif args.mode == 'test':
        test(args.n_test, args.filename, render=args.render, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
