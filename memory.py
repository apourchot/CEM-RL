# from https://github.com/ChenglongChen/pytorch-madrl/blob/master/common/Memory.py
from collections import namedtuple
import random

Experience = namedtuple("Experience",
                        ("states", "actions", "rewards", "next_states", "dones"))


class Memory(object):
    """
    Replay memory buffer
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _append_one(self, state, action, reward, next_state=None, done=None):
        """
        Add one element
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(
            state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def append(self, states, actions, rewards, next_states=None, dones=None):
        """
        Add elements
        """
        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s, a, r, n_s, d in zip(states, actions, rewards, next_states, dones):
                    self._append_one(s, a, r, n_s, d)
            else:
                for s, a, r in zip(states, actions, rewards):
                    self._append_one(s, a, r)
        else:
            self._append_one(states, actions, rewards, next_states, dones)

    def sample(self, batch_size):
        """
        Sample a batch
        """
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)
