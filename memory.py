import numpy as np

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer


class Memory(object):
    def __init__(self, max_size):
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
