import numpy as np;
import scipy;
import tensorflow as tf;
from collections import deque, namedtuple;

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1];

SARTV = namedtuple('SARTV',['state','action','reward','terminal','value']);
class Batch:

    def __init__(self, reward_discount = 0.95, maxlen = 1000):
        self._gamma = reward_discount;

        self._memory = deque(maxlen = maxlen);
        self._buffer = deque();

    def append(self, state, action, reward, terminal, value):
        self._buffer.append(SARTV(state,action,reward,terminal,value));
        if terminal:
            self.process_buffer();

    def process_buffer(self):
        states,actions, rewards, terminals, values  = [list(l) for l in zip(*self._buffer)];
        returns = discount(rewards, self._gamma)+self._gamma**(np.arange(len(rewards))[::-1]+1)*values[-1];
        advantage = returns - values;


class PPOAgent:
    def __init__(self, policy_fn, observation_space, actions_space,
        value_loss_factor = 0.1,
        entropy_loss_factor = 0.):
        # initialize the empty placeholder and result dict to allow for
        # probing the tensorflow graph easily
        self._tf_placeholders = dict();
        self._tf_results = dict();

        # Setup the instance variables.
        self._policy_fn = policy_fn;

        self._value_loss_factor = float(value_loss_factor);
        self._entropy_loss_factor = float(entropy_loss_factor);



        # Define the tensorflow graph
        self._initialize_tensorflow(observation_space, actions_space);

        # Get a session and initialize all global variables.
        self._session = tf.Session();
        self._session.run(tf.global_variables_initializer());
        writer = tf.summary.FileWriter("/tmp/log/1", self._session.graph)

    def _initialize_tensorflow(self, observation_space, action_space):
        # Define placeholder for the observations of the agent
        observations = self._rp(name="observations", shape = [None]+list(observation_space.shape));

        # Initialize the policy and reference policy
        self._policy = self._policy_fn("policy", observations, action_space);
        self._reference_policy = self._policy_fn("reference_policy", observations, action_space, reference = True);

        # Set up learning operations in tensorflow graph

        ## setup losses
        with tf.variable_scope('losses'):
            value_loss = self._setup_values_loss();
            policy_loss = self._setup_policy_loss();
            entropy_loss = self._setup_entropy_loss();
            total_loss = self._rr(policy_loss + self._entropy_loss_factor * entropy_loss + self._value_loss_factor * value_loss, 'total_loss')

        with tf.variable_scope('optimization'):
            total_policy_loss = policy_loss + self._entropy_loss_factor * entropy_loss;
            trainable_variables = self._policy.trainable_variables;
            self.tf_optimizer = tf.train.AdamOptimizer();
            self.tf_train_op = self.tf_optimizer.minimize(total_loss, var_list=trainable_variables)

    def _setup_values_loss(self):
        with tf.variable_scope('value_loss'):
            returns = self._rp(name="returns", shape=[None]);
            value_loss = self._rr(tf.reduce_sum(tf.square(self._policy.tf_value - returns)), "value_loss");
            return value_loss;

    def _setup_policy_loss(self):
        with tf.variable_scope('policy_loss'):
            advantages = self._rp(name="advantages", shape=[None]);
            actions = self._rp(name="actions", shape = self._policy.distribution.batch_shape)
            with tf.variable_scope('log_prob'):
                log_prob = self._policy.distribution._log_prob(actions);
            policy_loss = self._rr(-tf.reduce_sum(advantages*log_prob), "policy_loss")
            return policy_loss;

    def _setup_entropy_loss(self):
        with tf.variable_scope('entropy_loss'):
            with tf.variable_scope('entropy'):
                entropy = self._policy.distribution._entropy();
            entropy_loss = self._rr(entropy, "entropy_loss");

        return entropy_loss;







    def act(self, state, deterministic = False):
        return self._policy.act(self._session, state, deterministic);

    def learn(self, batch_iterable):
        pass;




    def _rp(self, **kwargs):
        return self._register_placeholder(**kwargs);

    def _register_placeholder(self, *,
        name = None,
        dtype = None,
        shape = None,
        placeholder = None):
        if placeholder is None:
            placeholder = tf.placeholder(name = name, dtype=tf.float32, shape=shape);
        elif name is None:
            name = placeholder.name;
        if name in self._tf_placeholders:
            raise ValueError("Placeholder with name {} already exists".format(name));
        self._tf_placeholders[name] = placeholder;
        return placeholder

    def _gp(self,*, name):
        assert name in self._tf_placeholders, "{} is not a registered placeholder".format(name);
        return self._tf_placeholders[name];

    def _get_placeholders (self, *, names):
        assert hasattr(names, "__iter__"), "Names should be iterable to get placeholders";
        return [self._gp(name = name) for name in names];

    def _rr(self, tensor, name):
        tensor = tf.identity(tensor,name= name);
        return self._register_result(name = name, tensor=tensor);

    def _register_result(self, *,
        name = None,
        tensor = None):
        assert name is not None;
        assert tensor is not None;
        assert name not in self._tf_results;
        self._tf_results[name] = tensor;
        return tensor;
