import numpy as np;
from scipy.signal import lfilter;
from collections import deque, namedtuple;
import random;

def discount(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1];

OIARTV = namedtuple('OIARTV',['observation','internals','action','reward','terminal','value']);
class Batch:

    def __init__(self, reward_discount = 0.95, maxlen = 1000):
        self._gamma = reward_discount;

        self._memory = deque(maxlen = maxlen);
        self._buffer = deque();

    def append(self, observations, internals, action, reward, terminal, value):
        self._buffer.append(OIARTV(observations, internals, action,reward,terminal,value));
        if terminal:
            self.process_buffer();

    def calculate_returns(self, rewards, final_value=0):
        # The return is defined as
        # R_t = \sum_{k=0}^{T-t} gamma^{k} * r_{t+k} + gamma^{T-t+1} V_{T+1}
        # where r is the reward and V is the value function
        returns = discount(rewards, self._gamma)+self._gamma**(np.arange(len(rewards))[::-1]+1)*final_value;
        return returns;

    def calculate_advantages(self, rewards, values, final_value = 0):
        # The advantage is defined as
        # A_t = r_t + V_{t+1} - V_t
        # where r is the reward and V is the value function
        rewards = np.array(rewards);
        values = np.array(values);
        values = np.append(values,final_value);
        return rewards+values[1:]-values[:-1];

    def process_buffer(self):
        observations, internals, actions, rewards, terminals, values  = [list(l) for l in zip(*self._buffer)];
        returns = self.calculate_returns(rewards)
        advantages = self.calculate_advantages(rewards, values)

        memory_items = zip(observations, internals, actions, returns, advantages);
        self._memory.extend(memory_items);
        self._buffer.clear();

    def iter(self, N = 10, batch_size = 128):
        shuffled = [];
        for _ in range(N):
            batch = [];
            while len(batch)<batch_size:
                if len(shuffled) == 0:
                    shuffled = list(self._memory)
                    random.shuffle(shuffled);
                batch += shuffled[:batch_size - len(batch)];
            yield batch;
