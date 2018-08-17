import numpy as np
import torch
import torch.multiprocessing as mp

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor

# Code based on https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# and https://github.com/jingweiz/pytorch-distributed/blob/master/core/memories/shared_memory.py


class Memory():

    def __init__(self, memory_size, state_dim, action_dim):

        # params
        self.memory_size = memory_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos = mp.Value('l', 0)
        self.full = mp.Value('b', False)

        self.states = FloatTensor(torch.zeros(
            self.memory_size, self.state_dim))
        self.actions = FloatTensor(torch.zeros(
            self.memory_size, self.action_dim))
        self.n_states = FloatTensor(
            torch.zeros(self.memory_size, self.state_dim))
        self.rewards = FloatTensor(torch.zeros(self.memory_size, 1))
        self.dones = FloatTensor(torch.zeros(self.memory_size, 1))

    def size(self):
        if self.full.value:
            return self.memory_size
        return self.pos.value

    # Expects tuples of (state, next_state, action, reward, done)

    def add(self, datum):

        state, n_state, action, reward, done = datum

        self.states[self.pos.value][:] = FloatTensor(state)
        self.n_states[self.pos.value][:] = FloatTensor(n_state)
        self.actions[self.pos.value][:] = FloatTensor(action)
        self.rewards[self.pos.value][:] = FloatTensor([reward])
        self.dones[self.pos.value][:] = FloatTensor([done])

        self.pos.value += 1
        if self.pos.value == self.memory_size:
            self.full.value = True
            self.pos.value = 0

    def sample(self, batch_size):

        upper_bound = self.memory_size if self.full.value else self.pos.value
        batch_inds = torch.LongTensor(
            np.random.randint(0, upper_bound, size=batch_size))

        return (self.states[batch_inds],
                self.n_states[batch_inds],
                self.actions[batch_inds],
                self.rewards[batch_inds],
                self.dones[batch_inds])


class SharedMemory():

    def __init__(self, memory_size, state_dim, action_dim):

        # params
        self.memory_size = memory_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos = mp.Value('l', 0)
        self.full = mp.Value('b', False)

        self.states = FloatTensor(torch.zeros(
            self.memory_size, self.state_dim))
        self.actions = FloatTensor(torch.zeros(
            self.memory_size, self.action_dim))
        self.n_states = FloatTensor(
            torch.zeros(self.memory_size, self.state_dim))
        self.rewards = FloatTensor(torch.zeros(self.memory_size, 1))
        self.dones = FloatTensor(torch.zeros(self.memory_size, 1))

        self.states.share_memory_()
        self.actions.share_memory_()
        self.n_states.share_memory_()
        self.rewards.share_memory_()
        self.dones.share_memory_()

        self.memory_lock = mp.Lock()

    def size(self):
        if self.full.value:
            return self.memory_size
        return self.pos.value

    # Expects tuples of (state, next_state, action, reward, done)

    def _add(self, datum):

        state, n_state, action, reward, done = datum

        self.states[self.pos.value][:] = FloatTensor(state)
        self.n_states[self.pos.value][:] = FloatTensor(n_state)
        self.actions[self.pos.value][:] = FloatTensor(action)
        self.rewards[self.pos.value][:] = FloatTensor([reward])
        self.dones[self.pos.value][:] = FloatTensor([done])

        self.pos.value += 1
        if self.pos.value == self.memory_size:
            self.full.value = True
            self.pos.value = 0

    def add(self, experience):
        with self.memory_lock:
            return self._add(experience)

    def _sample(self, batch_size):

        upper_bound = self.memory_size if self.full.value else self.pos.value
        batch_inds = torch.LongTensor(
            np.random.randint(0, upper_bound, size=batch_size))

        return (self.states[batch_inds],
                self.n_states[batch_inds],
                self.actions[batch_inds],
                self.rewards[batch_inds],
                self.dones[batch_inds])

    def sample(self, batch_size):
        with self.memory_lock:
            return self._sample(batch_size)
