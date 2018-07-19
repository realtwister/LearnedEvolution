from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np;

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
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

    def logp(self, x):
        return - self.neglogp(x)

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=False, summaries = False, should_act = True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)

        sequence_length = None
        ob = tf.get_default_graph().get_tensor_by_name("observations:0");
        if ob is None:
            ob = U.get_placeholder(name="observations", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope('pol'):
            last_out = ob
            for i in range(num_hid_layers):
                last_out = tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1))
                last_out = tf.nn.elu(last_out);
                #last_out = tf.nn.tanh(last_out)


        with tf.variable_scope("distribution") as dist_scope:
            if True:#Old std
                if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                    mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final',kernel_initializer=U.normc_initializer(0.01))
                    logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())+mean * 0.0
                else:

                    flat = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final',kernel_initializer=U.normc_initializer(0.01))
                    mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
            else:
                    mean = tf.layers.dense(last_out, ac_space.shape[0], name='mean',kernel_initializer=U.normc_initializer(0.01))
                    logstd = tf.ones([ac_space.shape[0]])*tf.layers.dense(last_out, 1, name='logstd',kernel_initializer=U.normc_initializer(0.01))
            logstd_offset = tf.get_variable("logstd_offset",initializer=tf.constant(0.), trainable=False);
            logstd_placeholder = tf.placeholder(tf.float32, shape=());
            self.add_std_offset = U.function([logstd_placeholder], [tf.assign(logstd_offset, tf.add(logstd_offset,logstd_placeholder))])
            self.pd = DiagGaussianPd(mean, logstd_offset+logstd)
            stochastic = tf.placeholder(dtype=tf.bool, shape=(), name = "stochastic")
            ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())

        if should_act:

            with tf.variable_scope('vf'):
                #obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
                last_out = ob
                for i in range(num_hid_layers):
                    last_out = tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0))
                    last_out = tf.nn.tanh(last_out);
                self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]


            self.state_in = []
            self.state_out = []

            self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
