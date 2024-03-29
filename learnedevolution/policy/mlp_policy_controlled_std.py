from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
import numpy as np;
from baselines.common.distributions import make_pdtype

class MlpPolicyControlled(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, summaries = False, should_act = True):
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
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer(), trainable=False)
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
                self._logstd = np.zeros([1, pdtype.param_shape()[0]//2]);
                self._increase_op = tf.assign(logstd, logstd+0.2);
                self._decrease_op = tf.assign(logstd, logstd-0.2);
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final')

        with tf.variable_scope("distribution"):
            self.pd = pdtype.pdfromflat(pdparam)

        if should_act:
            with tf.variable_scope("obfilter"):
                self.ob_rms = RunningMeanStd(shape=ob_space.shape)

            with tf.variable_scope('vf'):
                #obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
                last_out = ob
                for i in range(num_hid_layers):
                    last_out = tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0))
                    last_out = tf.nn.tanh(last_out);
                self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]


            self.state_in = []
            self.state_out = []

            with tf.variable_scope("distribution"):
                stochastic = tf.placeholder(dtype=tf.bool, shape=(), name = "stochastic")
                ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())

            self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None]);
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

    def increase_logstd(self,sess):
        sess.run(self._increase_op);
    def decrease_logstd(self,sess):
        sess.run(self._decrease_op);
