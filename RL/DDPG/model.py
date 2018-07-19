from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np

from RL.DDPG.util import hard_update, to_numpy

USE_CUDA = torch.cuda.is_available()


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

    def forward(self, x):

        out = self.ln1(x)
        out = self.fc1(out)
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
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_params_grad(self):
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in
                                   self.parameters()]))

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if USE_CUDA:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def set_params_grad(self, params):
        """
        Set the params gradient
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if USE_CUDA:
                param.grad.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.grad.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def scale_params(self, scale):
        """
        Multiply all parameters by the given scale
        """
        for param in self.parameters():
            param.data.copy_(scale * param.data)

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

        out_x = self.ln1(x)
        out_x = self.fc1(out_x)
        out_x = self.relu(out_x)

        out_a = self.ln2(a)
        out_a = self.fc2(out_a)
        out_a = self.relu(out_a)

        out = self.ln3(torch.cat([out_x, out_a], 1))
        out = self.fc3(out)
        out = self.relu(out)

        out = self.ln4(out)
        out = self.fc4(out)
        return out

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_params_grad(self):
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in
                                   self.parameters()]))

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if USE_CUDA:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def set_params_grad(self, params):
        """
        Set the params gradient
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if USE_CUDA:
                param.grad.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.grad.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def scale_params(self, scale):
        """
        Multiply all parameters by the given scale
        """
        for param in self.parameters():
            param.data.copy_(scale * param.data)

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
