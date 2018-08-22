# !/usr/bin/env python3
import argparse
import sys
from copy import deepcopy

import gym
import gym.spaces
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import logging
from natsort import natsorted
from tqdm import tqdm

from models import Actor
from random_process import *
from util import *
from memory import Memory, SharedMemory

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


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
            n_obs, reward, done, info = env.step(action)
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
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true')
    parser.add_argument('--use_td3', dest='use_td3', action='store_true')
    parser.add_argument('--env', default="HalfCheetah-v2", type=str)
    parser.add_argument('--n_actor', default=1, type=int)
    parser.add_argument('--period', default=5000, type=int)
    parser.add_argument('--n_episodes', default=10, type=int)
    parser.add_argument('--dir_name', default="", type=str)

    args = parser.parse_args()

    # The environment
    envs = [gym.make(args.env) for _ in range(args.n_actor)]
    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]
    max_action = int(envs[0].action_space.high[0])

    actors = [Actor(state_dim, action_dim, max_action,
                    layer_norm=args.layer_norm) for _ in range(args.n_actor)]

    df = pd.DataFrame(columns=["total_steps", "average_score", "best_score"] +
                      ["score_{}".format(i) for i in range(args.n_actor)])

    dir_list = next(os.walk(args.dir_name))[1]

    for dir_ in tqdm(dir_list, position=1):

        subdir_list = next(os.walk(args.dir_name + '/' + dir_))[1]
        subdir_list = natsorted(subdir_list)

        for subdir_ in tqdm(subdir_list, position=2):

            step_c = 0
            step = 0

            n_step = int(subdir_.split("_")[0])
            step_c += n_step - step
            if step_c >= args.period:

                step_c = 0

                # load actors
                for i in range(args.n_actor):
                    actors[i].load_model(
                        args.dir_name + '/' + dir_ + "/" + subdir_, "actor_{}".format(i))

                # test actors on ten seeds
                fs = []
                for i in range(args.n_actor):
                    f, _ = evaluate(actors[i], envs[i], n_episodes=args.n_episodes,
                                    noise=None, random=False, memory=None)
                    fs.append(f)

                # save score in new log
                res = {"total_steps": n_step,
                       "average_score": np.mean(fs), "best_score": np.max(fs)}
                for i in range(args.n_actor):
                    res["score_{}".format(i)] = fs[i]
                print(res)
                df = df.append(res, ignore_index=True)
                df.to_pickle(args.dir_name + "/" + dir_ + "/log_n.pkl")

            step = n_step
