import argparse
from copy import deepcopy

import gym
import numpy as np
import pandas as pd
from tqdm import tqdm

from EA.GA import GA, GAv2
from EA.ES import OpenES, SNES
from RL.DDPG.model import Actor, Critic
from RL.DDPG.ddpg import DDPG
from RL.DDPG.util import *
from memory import Memory
from normalized_env import NormalizedEnv


def evaluate(actor, agent, n_episodes=1, noise=False, render=False, training=False, add_memory=True):
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
            if add_memory:
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

    h_params = ea_h.ask()
    a_params = ea_a.ask()

    fitness_h = []
    fitness_a = []

    # evaluate all actors
    for i in range(args.pop_size):

        h_param = h_params[i]
        a_param = a_params[i]

        # set new hyper parameters
        args.reward_scale = h_param[0]
        args.actor_lr = h_param[1]
        args.critic_lr = h_param[2]
        global agent
        critic = agent.get_critic()
        agent = DDPG(nb_states, nb_actions, memory, args)

        # set new critic
        agent.set_critic(critic)

        # set new actor
        actor = Actor(nb_states, nb_actions)
        actor.set_params(a_param)
        agent.set_actor(actor)

        f_a, f_h, steps = train_rl(agent, n_batch=1000, render=render)
        batch_steps += steps
        fitness_a.append(f_a)
        fitness_h.append(f_h)

        # get new weigths
        a_params[i] = agent.get_actor_params()

    # update ea
    ea_h.tell(fitness_h)
    ea_a.set_new_params(a_params)
    ea_a.tell(fitness_a)

    return batch_steps


def train_rl(agent, n_batch=10, render=False, debug=True):
    """
    Train the deep RL agent
    """

    # score before
    f_b, steps_b = evaluate(agent.get_actor(), agent, n_episodes=1,
                            noise=False, render=render, training=False)

    # learning
    if len(memory) > args.batch_size:
        for _ in tqdm(range(n_batch)):
            agent.train()

    # score after
    f, steps = evaluate(agent.get_actor(), agent, n_episodes=1,
                        noise=False, render=render, training=False)
    incr = (f - f_b) / np.abs(f_b)

    # printing
    if debug:
        prLightGray("Hyper-parameters:{}, {}, {}".format(args.reward_scale,
                                                         args.actor_lr, args.critic_lr))
        prLightPurple('EA actor fitness before:{}'.format(f_b))
        prLightPurple('EA actor fitness after:{}'.format(f))
        prRed('EA actor fitness relative increase:{}'.format(incr))

    return f, incr, steps + steps_b


def train(n_gen, n_episodes, output=None, debug=False, render=False):
    """
    Train the whole process
    """

    total_steps = 0
    df = pd.DataFrame(columns=["total_steps", "best_score"])

    for n in range(n_gen):

        steps_ea = train_ea(n_episodes=n_episodes,
                            gen_index=n, debug=debug, render=render)
        total_steps += steps_ea

        # printing scores
        if debug:
            prPurple('Generation#{}: Total steps:{} Best Score:{} \n'.format(
                n, total_steps, ea_a.best_fitness()))

        # saving model and scores
        # if n % 10 == 0:
        #     best_params = ea_a.best_actor()
        #     best_actor_params = best_params[3:3 + actor_size]
        #     best_critic_params = best_params[-critic_size:]
# 
        #     best_actor = Actor(nb_states, nb_actions)
        #     best_actor.set_params(best_actor_params)
        #     best_critic = Critic(nb_states, nb_actions)
        #     best_critic.set_params(best_critic_params)
# 
        #     best_actor.save_model(output)
        #     best_critic.save_model(output)
        #     df.to_pickle(output + "/log.pkl")

        # saving scores
        best_score = ea_a.best_fitness()
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
    f, _ = evaluate(actor, agent, n_episodes=n_test,
                    noise=False, render=render, train=False)

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
    def gen_h():
        return np.hstack((np.random.rand(), np.exp(np.random.uniform(low=-5, high=-1) * 2.3),
                          np.exp(np.random.uniform(
                              low=-5, high=-1) * 2.3)
                          ))

    def gen_a():
        return Actor(nb_states, nb_actions).get_params()

    ea_h = GAv2(3, pop_size=args.pop_size, mut_amp=0.1,
                mut_rate=args.mut_rate, elite_frac=args.elite_frac, generator=gen_h)

    ea_a = GAv2(actor_size, pop_size=args.pop_size, mut_amp=args.mut_amp,
                mut_rate=args.mut_rate, elite_frac=args.elite_frac, generator=gen_a)

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
