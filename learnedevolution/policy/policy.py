import numpy as np;
import tensorflow as tf
from ..utils.parse_config import ParseConfig
from ..utils.parse_config import config_factory;

class Policy(ParseConfig):
    def __init__(self,
    observation_space,
    policy_network,
    value_network = None):
        self.observation_space = observation_space
        self.policy_network = policy_network
        if value_network is not None:
            self.value_network = value_network
        else:
            self.value_network = policy_network

    def copy(self):
        return Policy(
            observation_space = self.observation_space,
            policy_network = self.policy_network,
            value_network = self.value_network
        )
    def __call__(self, name):
        self = self.copy()
        self.name = name
        self._setup()
        return self


    def _setup(self):
        with tf.variable_scope(self.name):
            # Observation placeholder
            self.observation = tf.placeholder(
                dtype = tf.float32,
                shape = [None] + list(self.observation_space.state_space['shape']),
                name = "observation");

            # setup policy network
            with tf.variable_scope('policy'):
                self.policy_features = self.policy_network(self.observation);
                self.mean = tf.layers.dense(self.policy_features,
                self.observation_space.action_space['shape'][0],
                name='mean');
                self.logstd = tf.layers.dense(self.policy_features,
                self.observation_space.action_space['shape'][0],
                name='logstd');

                self.pd = DiagGaussianPd(self.mean, self.logstd)
                self.stochastic = tf.placeholder(dtype=tf.bool, shape=(), name = "stochastic")
                self.action = tf.case([(self.stochastic, lambda:self.pd.sample())], default=lambda: self.pd.mode());

            # setup value network
            with tf.variable_scope('value'):
                self.value_features = self.value_network(self.observation);
                self.vpred = tf.layers.dense(self.value_features, 1, name="value");
        self.scope = tf.get_variable_scope().name

    def act(self, observation, stochastic = False):
        sess = tf.get_default_session();
        action, value = sess.run([self.action, self.vpred],
        {self.observation: observation, self.stochastic:stochastic});
        return action[0], value[0];

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    @classmethod
    def _get_kwargs(cls, config, key=""):
        def default_network(last_out):
            last_out = tf.layers.dense(last_out, 128, name='fc1')
            last_out = tf.nn.elu(last_out);
            last_out = tf.layers.dense(last_out, 128, name='fc2')
            last_out = tf.nn.elu(last_out);
            return last_out;
        cls._config_required(
            "observation_space",
            "policy_network",
            "value_network"
        )
        cls._config_defaults(
            policy_network = default_network,
            value_network = None
        )
        kwargs = super()._get_kwargs(config, key = key);
        print(config);
        print(key)

        from ..states import states_classes;

        kwargs['observation_space'] = config_factory(states_classes, config, key+'.observation_space')
        return kwargs


class DiagGaussianPd(object):
    def __init__(self, mean, logstd):
        self.mean = mean
        tf.summary.histogram("mean", mean);
        self.logstd = tf.clip_by_value(logstd, -6, 2, name="clipped_log_std");
        tf.summary.histogram("logstd", self.logstd);
        self.std = tf.exp(self.logstd)
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.mean
    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)
    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)
    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
    def sample(self):
        return tf.random_normal(tf.shape(self.mean), mean=self.mean, stddev=self.std)

    def sample_placeholder(self, prefix =[], name=""):
        return tf.placeholder(dtype=tf.float32, shape = self.mean.shape, name=name);
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

    def logp(self, x):
        return - self.neglogp(x)
