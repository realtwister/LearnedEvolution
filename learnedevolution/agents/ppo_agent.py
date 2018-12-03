from .agent import Agent
from .ppo import BatchProvider, PPO
from ..utils.parse_config import config_factory

import tensorflow as tf

class PPOAgent(Agent):
    def __init__(self,
        observation_space = None,
        policy = None,
        reward_discount = 0.99,
        advantage_discount = 0.95,
        horizon = 100,
        clip_param = 0.2,
        entropy_param = -0.01,
        value_param = 0.001,
        epochs = 4,
        batch_size = 64,
        learning_rate = 1e-6
        ):
        self.batch = BatchProvider(
            reward_discount = reward_discount,
            advantage_discount = advantage_discount,
            epochs = epochs,
            horizon = horizon);
        self.ppo = PPO(None, policy, self.batch,
            clip_param = clip_param,
            entropy_param = entropy_param,
            value_param = value_param,
            epochs = epochs,
            batch_size = batch_size,
            learning_rate = learning_rate,
            observation_space = observation_space.gym_state_space,
            action_space = observation_space.gym_action_space
        )

    def reset(self):
        self.reward = None
        self.terminal = None
        self.observation = None
        pass;

    def seed(self, seed):
        tf.set_random_seed(seed)
        self.batch.seed(seed);

    def act(self, observation):
        if self.reward is None:
            self.ppo.reset(observation)
        else:
            self.ppo.observe(observation, self.reward, self.terminal)
        return self.ppo.act()[0]

    def observe(self, reward, terminal = False):
        self.reward = reward
        self.terminal = terminal

    def save(self, filename):
        self.ppo.save(filename);

    def restore(self, filename):
        self.ppo.restore(filename);

    def close(self):
        self.ppo.close();



    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'observation_space',
            'policy',
            'reward_discount',
            'advantage_discount',
            'horizon',
            'clip_param',
            'entropy_param',
            'value_param',
            'epochs',
            'batch_size',
            'learning_rate'
        )
        cls._config_defaults(
            reward_discount = 0.99,
            advantage_discount = 0.95,
            horizon = 100,
            clip_param = 0.2,
            entropy_param = -0.01,
            value_param = 0.001,
            epochs = 4,
            batch_size = 64,
            learning_rate = 1e-6,
            policy = {}
        )

        kwargs = super()._get_kwargs(config, key = key);

        from ..states import states_classes;

        kwargs['observation_space'] = config_factory(states_classes, config, key+'.observation_space')

        # We need to create the policy function
        #
        from ..policy.mlp_policy import MlpPolicy;
        def policy_fn(name, ob_space, ac_space, summaries = False, should_act = True):
            policy = MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,hid_size=128, num_hid_layers=2, summaries= summaries, should_act= should_act);
            return policy;
        kwargs['policy'] = policy_fn


        return kwargs
