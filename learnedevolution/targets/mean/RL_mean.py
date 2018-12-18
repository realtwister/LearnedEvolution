from .mean_target import MeanTarget;
from ...utils.parse_config import config_factory;

import os

class RLMean(MeanTarget):
    _API = 2.;
    def __init__(self, *,
        observation_space,
        reward_function,
        agent
        ):
        self._observation_space = observation_space;
        self._reward_fn = reward_function;
        self._agent = agent;
        self.learning = True;

    def _reset(self, initial_mean, initial_covariance):
        self._observation_space.reset();
        self._reward_fn.reset();
        self.reward = 0
        self._agent.reset();
        self.i = 0;

    def _seed(self, seed):
        self._agent.seed(seed);

    def _calculate(self, population):
        if self.i != 0:
            # calculate reward
            self.reward = self._reward_fn(population.population, population.fitness)

            # observe reward
            self._agent.observe(self.reward);

        # calculate observation
        self.observation = self._observation_space.encode(population)

        # calculate action
        self.action = self._agent.act(self.observation)

        # decode action
        self.mean = self._observation_space.decode(self.action)

        self.i += 1;

        return self.mean;
    def _calculate_deterministic(self,population):
        # calculate observation
        self.observation = self._observation_space.encode(population)
        action,_ = self._agent.ppo._policy.act(False, self.observation);
        self._action = action;
        self._target = self._observation_space.decode(action);
        self.i +=1;
        return self._target;

    def _terminating(self, population):
        if self.learning:
            self.reward = self._reward_fn(population.population, population.fitness)
            self._agent.observe(self.reward, True)
            self.observation = self._observation_space.encode(population)
            self._agent.act(self.observation)

    def save(self, savedir):
        filename = os.path.join(savedir,"RLMean");
        self._agent.save(filename);

    def restore(self, restoredir):
        filename = os.path.join(restoredir,"RLMean");
        self._agent.restore(filename);

    def close(self):
        self._agent.close();

    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'observation_space',
            'reward_function',
            'agent',
        )
        cls._config_defaults(
            observation_space = dict(
                type = "InvariantSpace"
            ),
            reward_function = dict(
                type = "DifferentialReward"
            ),
            agent = dict(
                type = "PPOAgent",
                policy = dict(
                    type = "MlpPolicy"
                )
            ),
        )

        kwargs = super()._get_kwargs(config, key = key);

        from ...states import states_classes;
        from ...rewards import rewards_classes;
        from ...agents import agent_classes;

        kwargs['observation_space'] = config_factory(
            states_classes,
            config,
            key+'.observation_space')

        kwargs['reward_function'] = config_factory(
            rewards_classes,
            config,
            key+'.reward_function')

        kwargs['agent'] = config_factory(agent_classes,
            config,
            key+'.agent')

        return kwargs;
