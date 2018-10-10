import numpy as np;
import os;
from tensorforce.agents import PPOAgent;



from learnedevolution.targets.mean.mean_target import MeanTarget;
from learnedevolution.states.new_normalized_state import NewNormalizedState;

class TensorforceMean(MeanTarget):
    _API = 2.;

    def __init__(self, dimension, population_size, rewards, convergence_criteria, logdir = None):
        super().__init__();
        self.p['population_size'] = population_size;
        self.p['dimension'] = dimension;

        self._state = NewNormalizedState(population_size,dimension);
        self._init_agent(logdir);
        self._prev_end = 0;

        self._rewards = rewards
        self._convergence_criteria = convergence_criteria;
        self._non_converged = -1000;
        self.learning = True;

    def _init_agent(self, log_dir):
        if log_dir is None:
            log_dir = "/tmp/thesis/tmp/"
        self._agent = PPOAgent(
            states=self._state.state_space,
            actions=self._state.action_space,
            network=[
                dict(type='dense', size=128),
                #dict(type="internal_lstm", size=128),
                dict(type='dense', size=64),
            ],
            batching_capacity=1000,
            step_optimizer=dict(
                type='adam',
                learning_rate=1e-6
            ),
            likelihood_ratio_clipping=0.10,
            entropy_regularization=0.,
            gae_lambda=0.95,
            discount= 0.99,
            baseline_mode="states",
            baseline = dict(
                type="mlp",
                sizes=[128,128]
            ),
            baseline_optimizer=dict(
                type='multi_step',
                optimizer=dict(
                    type='adam',
                    learning_rate=1e-3
                ),
                num_steps=5
            ),
            summarizer=dict(
                directory = os.path.join(log_dir,'agent'),
                labels=[
                    'losses',
                    'inputs',
                    'regularization',
                    'configuration',
                    'graph',
                ],
            )
        )

    def _reset(self, initial_mean, initial_covariance):
        for reward in self._rewards:
            reward.reset();

        self._mean = initial_mean;
        self._covariance = initial_covariance;

        self._current_rewards = [];

        self._state.reset();
        self._agent.reset();
        self._should_observe = False;

    def _calculate(self, population, deterministic=False):
        self._current_state = state = self._state.encode(population);
        self._observe(population)
        action = self._agent.act(state,deterministic = deterministic, independent=not self.learning);

        self._action = action;
        self._target = self._state.decode(action);
        if np.any(np.isnan(self._target)):
            print(self._target);
            raise Exception("mean is nan");
        if np.any(np.isinf(self._target)):
            print(self._target);
            raise Exception("mean is inf");

        return self._target;

    def _calculate_deterministic(self,population):
        return self._calculate(population, True);

    def _observe(self, population):
        self._current_reward = reward = self._calculate_reward(population);
        self._current_rewards +=[reward];

        if self._should_observe and self.learning:
            self._agent.observe(reward = reward, terminal= False);
        self._should_observe = True;

    def _calculate_reward(self, population):
        reward = 0.;
        for reward_fn,w in self._rewards.items():
            reward += w * reward_fn(population.population, population.fitness);
        return reward;

    def _terminating(self, population):
        reward = 0;
        mean_reward = np.percentile(self._current_rewards,90);
        for criterion in self._convergence_criteria:
            reward += criterion.reward;
        self._current_reward += reward;
        if self.learning:
            self._agent.observe(reward = reward, terminal=True);
