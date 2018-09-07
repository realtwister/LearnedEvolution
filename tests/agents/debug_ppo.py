import tensorflow as tf;
import numpy as np;
from copy import deepcopy;

from learnedevolution.agents.ppo import PPOAgent;
from learnedevolution.agents.batch import Batch;

class Policy:
    def __call__(self, scope, observations, action_space, reference=False):
        self = deepcopy(self);
        with tf.variable_scope(scope) as scope:
            self._scope = scope.name;
            self._setup_policy_network(observations, action_space);
            if not reference:
                self._setup_value_network(observations);
        self.observations = observations;
        return self;


    def _setup_policy_network(self, observations, action_space):
        with tf.variable_scope("policy_network"):
            layer = tf.layers.dense(observations, 10, tf.nn.relu);
            with tf.variable_scope('distribution'):
                # mean = tf.layers.dense(layer, action_space.shape[0], name="mean");
                # log_covariance = tf.layers.dense(layer, action_space.shape[0], name="log_covariance");
                # self.distribution = tf.distributions.Normal(loc=mean, scale= tf.exp(log_covariance));
                self.logits = tf.layers.dense(layer, action_space.n, tf.nn.softmax);
                self.distribution = tf.distributions.Categorical(self.logits);
                self.sample = self.distribution.sample();
                self.mode = self.distribution.mode();

    def _setup_value_network(self, observations):
        with tf.variable_scope("value_network"):
            layer = tf.layers.dense(observations, 128, tf.nn.relu);
            self.tf_value = tf.layers.dense(layer, 1);

    def feeddict(self, state):
        return {
            self.observations: [state],
        }

    def act(self,session, state, deterministic):
        if deterministic:
            return session.run(self.mode, self.feeddict(state));
        return session.run(self.sample, self.feeddict(state));

    @property
    def trainable_variables(self):
        return tf.trainable_variables(self._scope);


import gym;
env = gym.make('CartPole-v1');

policy  = Policy();
batch = Batch();

agent = PPOAgent(policy, env.observation_space, env.action_space);

observation = env.reset();
done = False;
while not done:
    action = agent.act(observation)[0];
    observation, reward, done, info = env.step(action);
