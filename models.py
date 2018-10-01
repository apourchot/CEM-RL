from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from util import to_numpy

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class RLNN(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(RLNN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_grads(self):
        """
        Returns the current gradient
        """
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]

    def load_model(self, filename, net_name):
        """
        Loads the model
        """
        if filename is None:
            return

        self.load_state_dict(
            torch.load('{}/{}.pkl'.format(filename, net_name),
                       map_location=lambda storage, loc: storage)
        )

    def save_model(self, output, net_name):
        """
        Saves the model
        """
        torch.save(
            self.state_dict(),
            '{}/{}.pkl'.format(output, net_name)
        )


class Actor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, layer_norm=False, init=True):
        super(Actor, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        if layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = layer_norm

    def forward(self, x):

        if not self.layer_norm:
            x = torch.tanh(self.l1(x))
            x = torch.tanh(self.l2(x))
            x = self.max_action * torch.tanh(self.l3(x))

        else:
            x = torch.tanh(self.n1(self.l1(x)))
            x = torch.tanh(self.n2(self.l2(x)))
            x = self.max_action * torch.tanh(self.l3(x))

        return x


class Critic(RLNN):
    def __init__(self, state_dim, action_dim, layer_norm=False):
        super(Critic, self).__init__(state_dim, action_dim, 1)

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = layer_norm

    def forward(self, x, u):

        if not self.layer_norm:
            x = torch.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x = torch.leaky_relu(self.l2(x))
            x = self.l3(x)

        else:
            x = torch.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x = torch.leaky_relu(self.n2(self.l2(x)))
            x = self.l3(x)

        return x


class CriticTD3(RLNN):
    def __init__(self, state_dim, action_dim, layer_norm=False):
        super(CriticTD3, self).__init__(state_dim, action_dim, 1)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        if layer_norm:
            self.n4 = nn.LayerNorm(400)
            self.n5 = nn.LayerNorm(300)
        self.layer_norm = layer_norm

    def forward(self, x, u):

        if not self.layer_norm:
            x1 = torch.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x1 = torch.leaky_relu(self.l2(x1))
            x1 = self.l3(x1)

        else:
            x1 = torch.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x1 = torch.leaky_relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

        if not self.layer_norm:
            x2 = torch.leaky_relu(self.l4(torch.cat([x, u], 1)))
            x2 = torch.leaky_relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x2 = torch.leaky_relu(self.n4(self.l4(torch.cat([x, u], 1))))
            x2 = torch.leaky_relu(self.n5(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2
