from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from util import to_numpy


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ActorERL(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, init=False):
        super(ActorERL, self).__init__()

        self.l1 = nn.Linear(state_dim, 128)
        self.n1 = LayerNorm(128)
        self.nn1 = nn.LayerNorm(128)

        self.l2 = nn.Linear(128, 128)
        self.n2 = LayerNorm(128)

        self.l3 = nn.Linear(128, action_dim)

        self.max_action = max_action
        if init:
            self.l3.weight.data.mul_(0.1)
            self.l3.bias.data.mul_(0.1)

    def forward(self, x):

        x = F.tanh(self.n1(self.l1(x)))
        x = F.tanh(self.n2(self.l2(x)))
        x = F.tanh(self.l3(x))

        return self.max_action * x

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

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]

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


class CriticERL(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticERL, self).__init__()

        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(action_dim, 128)

        self.l3 = nn.Linear(256, 256)
        self.n3 = LayerNorm(256)

        self.l4 = nn.Linear(256, 1)
        self.l4.weight.data.mul_(0.1)
        self.l4.bias.data.mul_(0.1)

    def forward(self, x, u):

        x = F.elu(self.l1(x))
        u = F.elu(self.l2(u))
        x = torch.cat((x, u), 1)

        x = F.elu(self.n3(self.l3(x)))
        x = self.l4(x)

        return x

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

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]

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


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, layer_norm=False):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        if layer_norm:
            self.n1 = nn.LayerNorm(400)
        self.l2 = nn.Linear(400, 300)
        if layer_norm:
            self.n2 = nn.LayerNorm(300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.layer_norm = layer_norm

    def forward(self, x):

        x = F.relu(self.l1(x))
        if self.layer_norm:
            x = self.n1(x)

        x = F.relu(self.l2(x))
        if self.layer_norm:
            x = self.n2(x)

        x = self.max_action * F.tanh(self.l3(x))

        return x

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

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]

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
    def __init__(self, state_dim, action_dim, layer_norm=False):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        if layer_norm:
            self.n1 = nn.LayerNorm(400)
        self.l2 = nn.Linear(400, 300)
        if layer_norm:
            self.n2 = nn.LayerNorm(300)
        self.l3 = nn.Linear(300, 1)

        self.layer_norm = layer_norm

    def forward(self, x, u):

        x = F.relu(self.l1(torch.cat([x, u], 1)))
        if self.layer_norm:
            x = self.n1(x)

        x = F.relu(self.l2(x))
        if self.layer_norm:
            x = self.n2(x)

        x = self.l3(x)

        return x

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

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]

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


class CriticTD3(nn.Module):
    def __init__(self, state_dim, action_dim, layer_norm):
        super(CriticTD3, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        if layer_norm:
            self.n1 = nn.LayerNorm(400)
        self.l2 = nn.Linear(400, 300)
        if layer_norm:
            self.n2 = nn.LayerNorm(300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        if layer_norm:
            self.n4 = nn.LayerNorm(400)
        self.l5 = nn.Linear(400, 300)
        if layer_norm:
            self.n5 = nn.LayerNorm(300)
        self.l6 = nn.Linear(300, 1)

        self.layer_norm = layer_norm

    def forward(self, x, u):

        x1 = F.relu(self.l1(torch.cat([x, u], 1)))
        if self.layer_norm:
            x1 = self.n1(x1)

        x1 = F.relu(self.l2(x1))
        if self.layer_norm:
            x1 = self.n2(x1)

        x1 = self.l3(x1)

        x2 = F.relu(self.l4(torch.cat([x, u], 1)))
        if self.layer_norm:
            x2 = self.n4(x2)

        x2 = F.relu(self.l5(x2))
        if self.layer_norm:
            x2 = self.n5(x2)

        x2 = self.l6(x2)

        return x1, x2

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

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]

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


class CriticTD3ERL(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticTD3ERL, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(action_dim, 128)

        self.l3 = nn.Linear(256, 256)
        self.n3 = nn.LayerNorm(256)

        self.l4 = nn.Linear(256, 1)
        self.l4.weight.data.mul_(0.1)
        self.l4.bias.data.mul_(0.1)

        # Q2 architecture
        self.l5 = nn.Linear(state_dim, 128)
        self.l6 = nn.Linear(action_dim, 128)

        self.l7 = nn.Linear(256, 256)
        self.n7 = nn.LayerNorm(256)

        self.l8 = nn.Linear(256, 1)
        self.l8.weight.data.mul_(0.1)
        self.l8.bias.data.mul_(0.1)

    def forward(self, x, u):

        x1 = F.elu(self.l1(x))
        u1 = F.elu(self.l2(u))
        x1 = torch.cat((x1, u1), 1)

        x1 = F.elu(self.n3(self.l3(x1)))
        x1 = self.l4(x1)

        x2 = F.elu(self.l5(x))
        u2 = F.elu(self.l6(u))
        x2 = torch.cat((x2, u2), 1)

        x2 = F.elu(self.n7(self.l7(x2)))
        x2 = self.l8(x2)

        return x1, x2

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

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]

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
