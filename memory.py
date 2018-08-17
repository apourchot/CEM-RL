import numpy as np
import torch
import torch.multiprocessing as mp

# Code based on https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# and https://github.com/jingweiz/pytorch-distributed/blob/master/core/memories/shared_memory.py


class Memory():

    def __init__(self, max_size):

        # params
        self.max_size = max_size
        self.storage = [None] * max_size
        self.position = 0
        self.filled = 0

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.storage[self.position] = data
        self.position = (self.position + 1) % self.max_size
        self.filled = min(self.filled + 1, self.max_size)

    def sample(self, batch_size=100):
        ind = np.random.randint(
            0, self.filled, size=min(batch_size, self.filled))
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class SharedMemory():

    def __init__(self, memory_size, state_dim, action_dim):

        # params
        self.memory_size = memory_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos = mp.Value('l', 0)
        self.full = mp.Value('b', False)

        self.states = torch.zeros(self.memory_size, self.state_dim)
        self.actions = torch.zeros(self.memory_size, self.action_dim)
        self.n_states = torch.zeros(self.memory_size, self.state_dim)
        self.rewards = torch.zeros(self.memory_size, 1)
        self.dones = torch.zeros(self.memory_size, 1)

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

    def cuda(self):
        self.states.cuda()
        self.actions.cuda()
        self.n_states.cuda()
        self.rewards.cuda()
        self.dones.cuda()

    # Expects tuples of (state, next_state, action, reward, done)
    def _add(self, datum):

        state, n_state, action, reward, done = datum

        self.states[self.pos.value][:] = torch.FloatTensor(state)
        self.n_states[self.pos.value][:] = torch.FloatTensor(n_state)
        self.actions[self.pos.value][:] = torch.FloatTensor(action)
        self.rewards[self.pos.value][:] = torch.FloatTensor([reward])
        self.dones[self.pos.value][:] = torch.FloatTensor([done])

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
