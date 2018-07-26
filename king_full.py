import argparse
from copy import deepcopy

import gym
import numpy as np
import pandas as pd

from EA.GA import GA
from EA.ES import OpenES, SNES
from RL.DDPG.model import Actor, Critic
from RL.DDPG.ddpg import DDPG
from RL.DDPG.util import *
from memory import Memory
from normalized_env import NormalizedEnv


def evaluate(actor, n_episodes=1, noise=False, render=False, training=False):
    """
    Computes the score of an actor on a given number of runs
    """

    def policy(obs):
        action = to_numpy(actor(to_tensor(np.array([obs])))).squeeze(0)
        if noise:
            action += agent.random_process.sample()
        return action

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
            memory.append(obs, action, reward, deepcopy(n_obs), done)
            obs = n_obs

            # train the DDPG agent if needed
            if training and (steps % 5 == 0):
                agent.train()

            # render if needed
            if render:
                env.render()

            # reset when done
            if done:
                env.reset()

        scores.append(score)

    return np.mean(scores), steps


def train_ea(n_episodes=1, debug=False, gen_index=0, render=False):
    """
    Train EA process
    """

    batch_steps = 0
    actor = Actor(nb_states, nb_actions)
    params = ea.ask()
    fitness = []

    # evaluate all actors
    for param in params:

        actor_params = param[3:3 + actor_size]
        actor.set_params(actor_params)
        f, steps = evaluate(actor, n_episodes=n_episodes,
                            noise=False, render=render, training=True)
        batch_steps += steps
        fitness.append(f)

        # print scores
        if debug:
            prLightPurple(
                'Generation#{}: EA actor fitness:{}'.format(
                    gen_index, f))

        i += 1

    # update ea
    ea.tell(fitness)

    return batch_steps


def train(n_gen, n_episodes, output=None, debug=False, render=False):
    """
    Train the whole process
    """

    total_steps = 0
    df = pd.DataFrame(columns=["total_steps", "best_score"])

    for n in range(n_gen):

        # ea training
        steps_ea = train_ea(n_episodes=n_episodes,
                            gen_index=n, debug=debug, render=render)
        total_steps += steps_ea

        # update king
        best_params = ea.best_actor()
        king_index = ea.best_index()

        args.reward_scale = best_params[0]
        args.actor_lr = best_params[1]
        args.critic_lr = best_params[2]

        if n % omega == 0:
            global agent
            agent = DDPG(nb_states, nb_actions, memory, args)
            actor = Actor(nb_states, nb_actions)
            actor.set_params(best_params[3:3 + actor_size])
            agent.set_actor(actor)

            critic = Critic(nb_states, nb_actions)
            critic.set_params(best_params[-critic_size:])
            agent.set_critic(critic)

        # printing scores
        if debug:
            prPurple('Generation#{}: Total steps:{} Best Score:{} \n'.format(
                n, total_steps, ea.best_fitness()))

        # saving model and scores
        if n % 10 == 0:
            best_params = ea.best_actor()
            best_actor_params = best_params[3:3 + actor_size]
            best_critic_params = best_params[-critic_size:]

            best_actor = Actor(nb_states, nb_actions)
            best_actor.set_params(best_actor_params)
            best_critic = Critic(nb_states, nb_actions)
            best_critic.set_params(best_critic_params)

            best_actor.save_model(output)
            best_critic.save_model(output)
            df.to_pickle(output + "/log.pkl")

        # saving scores
        best_score = ea.best_fitness()
        df.append({"total_steps": total_steps,
                   "best_score": best_score}, ignore_index=True)


def test(n_test, filename, debug=False, render=False):
    """
    Test an agent
    """

    # load weights
    actor = Actor(nb_states, nb_actions)
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
    parser.add_argument('--env', default='Pendulum-v0', type=str)

    # DDPG parameters
    parser.add_argument('--hidden1', default=400, type=int)
    parser.add_argument('--hidden2', default=300, type=int)
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--reward_scale', default=1., type=float)

    # EA parameters
    parser.add_argument('--pop_size', default=10, type=int)
    parser.add_argument('--elite_frac', default=0.1, type=float)
    parser.add_argument('--mut_rate', default=0.9, type=float)
    parser.add_argument('--mut_amp', default=0.1, type=float)

    # Noise process parameters
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)

    # Training parameters
    parser.add_argument('--n_gen', default=100, type=int)
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--omega', default=10, type=int)
    parser.add_argument('--mem_size', default=6000000, type=int)

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
    env = NormalizedEnv(gym.make(args.env))
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    # Random seed
    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    # replay buffer
    memory = Memory(args.mem_size)

    # DDPG agent
    agent = DDPG(nb_states, nb_actions, memory, args)
    actor_size = agent.get_actor_size()
    critic_size = agent.get_critic_size()

    # EA process
    def generator():
        return np.hstack((np.random.rand(), np.exp(np.random.uniform(low=-5, high=-1) * 2.3),
                          np.exp(np.random.uniform(
                              low=-5, high=-1) * 2.3),
                          Actor(nb_states, nb_actions).get_params(),
                          Critic(nb_states, nb_actions).get_params()
                          ))

    ea = GA(3 + actor_size + critic_size, pop_size=args.pop_size, mut_amp=args.mut_amp,
            mut_rate=args.mut_rate, elite_frac=args.elite_frac, generator=generator)

    # Trying ES type algorithms, but without much success
    # ea = OpenES(agent.get_actor_size(), pop_size=args.pop_size, mut_amp=args.mut_amp,
    #           generator=lambda: Actor(nb_states, nb_actions).get_params())

    if args.mode == 'train':
        train(n_gen=args.n_gen, n_episodes=args.n_episodes,
              output=args.output, debug=args.debug, render=args.render)

    elif args.mode == 'test':
        test(args.n_test, args.filename, render=args.render, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
