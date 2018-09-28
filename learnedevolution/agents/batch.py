def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1];

class BatchProvider(object):
    def __init__(self, *,
        reward_discount = 0.99,
        advantage_discount = 0.95):

        self._reward_discount = float(reward_discount);
        self._advantage_discount = advantage_discount;

        self._new = deque(maxlen = epochs * horizon);
        self._memory = deque(maxlen = epochs * horizon);
        self._buffer = deque(maxlen = horizon);

    def observe(self,* , state, reward, terminal, predicted_value, action):
        self._buffer.append((
            state,
            reward,
            terminal,
            predicted_value,
            action
            ));
        if terminal:
            self._process_buffer();
        if self._buffer.maxlen <= len(self._buffer):
            self._process_buffer();

    def _process_buffer(self):
        """Calculates the advantages and returns for the current buffer, adds the buffer to the memory and clears the buffer.
        """
        if len(self._buffer) <= 2:
            return;
        # Unpack buffer
        states, rewards, terminals, values, actions  = [list(l) for l in zip(*self._buffer)];

        # Calculate advantage (TODO: add paper)
        values = np.array(values);
        delta = rewards[:-1] + self._reward_discount * values[1:] - values[:-1]
        advantages = discount(delta, self._reward_discount*self._advantage_discount);

        T = len(self._buffer)-1;
        advantages = np.empty(T, 'float32');
        lastgae = 0;
        for t in reversed(range(T)):
            nonterminal = 1 - terminals[t+1];
            delta = rewards[t] + self._reward_discount * values[t+1]*nonterminal - values[t];
            advantages[t] = lastgae =  delta + self._reward_discount * self._advantage_discount * nonterminal * lastgae;




        # Calculate return
        returns = advantages + values[:-1];
        returns = discount(rewards[:-1], self._reward_discount);
        observations  =list(zip(states, actions, advantages, returns));
        self._new.extend(observations);
        self._memory.extend(observations);
        self._buffer.clear();
        self._buffer.append((states[-1], rewards[-1], terminals[-1], values[-1], actions[-1]));

    def get_batch(self, batch_size = 32):
        batch_size = min(len(self), batch_size);
        return [np.array(l) for l in zip(*random.sample(self._memory, batch_size))];

    def get_new(self):
        new = copy.deepcopy(self._new);
        self._new.clear();
        return [np.array(l) for l in zip(*new)]

    def iterate_batch(self, batch_size = 32):
        batch_size = min(len(self), batch_size);
        permuted = np.random.permutation(self._memory);
        for i in range(np.ceil(len(permuted)/batch_size).astype(int)):
            yield [np.array(l) for l in zip(*permuted[i*batch_size:(i+1)*batch_size])];




    def __len__(self):
        return len(self._memory);

    def __getitem__(self, key):
        return self._memory[key];
    @property
    def maxlen(self):
        return self._memory.maxlen;
