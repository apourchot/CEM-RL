from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np

from RL.DDPG.util import hard_update


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, states_dim, actions_dim):
        super(Actor, self).__init__()
        self.ln1 = nn.LayerNorm(states_dim)
        self.fc1 = nn.Linear(states_dim, 128)

        self.ln2 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)

        self.ln3 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, actions_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # self.init_weights()

    def init_weights(self):
        """
        Weights init
        """
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

    def forward(self, x):

        # out = self.ln1(x)
        out = self.fc1(x)
        out = self.relu(out)

        out = self.ln2(out)
        out = self.fc2(out)
        out = self.relu(out)

        out = self.ln3(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([v.numpy().flatten() for v in
                                   self.state_dict().values()]))

    def scale_params(self, scale):
        """
        Multiply all parameters by the scale
        """
        for param in self.parameters():
            param.data.copy_(scale * param.data)

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        state_dict = self.state_dict()
        cpt = 0

        # putting parameters in the right shape
        for k, v in zip(state_dict.keys(), state_dict.values()):
            tmp = np.product(v.size())
            state_dict[k] = torch.from_numpy(
                params[cpt:cpt + tmp]).view(v.size())
            cpt += tmp

        # setting parameters of the network
        self.load_state_dict(state_dict)

    def load_model(self, filename):
        """
        Loads the model
        """
        if filename is None:
            return

        self.load_state_dict(
            torch.load('{}/actor.pkl'.format(filename))
        )

    def save_model(self, output):
        """
        Saves the model
        """
        torch.save(
            self.state_dict(),
            '{}/actor.pkl'.format(output)
        )


class Critic(nn.Module):
    def __init__(self, states_dim, actions_dim):
        super(Critic, self).__init__()
        self.ln1 = nn.LayerNorm(states_dim)
        self.fc1 = nn.Linear(states_dim, 200)

        self.ln2 = nn.LayerNorm(actions_dim)
        self.fc2 = nn.Linear(actions_dim, 200)

        self.ln3 = nn.LayerNorm(400)
        self.fc3 = nn.Linear(400, 300)

        self.ln4 = nn.LayerNorm(300)
        self.fc4 = nn.Linear(300, 1)
        self.relu = nn.ReLU()

        # self.init_weights()

    def forward(self, xs):
        x, a = xs

        # out_x = self.ln1(x)
        out_x = self.fc1(x)
        out_x = self.relu(out_x)

        # out_a = self.ln2(a)
        out_a = self.fc2(a)
        out_a = self.relu(out_a)

        out = self.ln3(torch.cat([out_x, out_a], 1))
        out = self.fc3(out)
        out = self.relu(out)

        out = self.ln4(out)
        out = self.fc4(out)
        return out

    def init_weights(self):
        """
        Weights init
        """
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())

    def load_model(self, filename):
        """
        Loads the model
        """
        if filename is None:
            return

        self.load_state_dict(
            torch.load('{}/actor.pkl'.format(filename))
        )

    def save_model(self, output):
        """
        Saves the model
        """
        torch.save(
            self.state_dict(),
            '{}/critic.pkl'.format(output)
        )
