import numpy as np;

from tensorforce.agents import PPOAgent;

from .mean_target import MeanTarget;
from ...rewards.differential_reward import DifferentialReward;
from ...rewards.divergence_penalty import DivergencePenalty;

class TensorforcePPOMean(MeanTarget):
    def __init__(self, dimension, population_size):
        super().__init__();
        self.p['population_size'] = population_size;
        self.p['dimension'] = dimension;
        self._init_agent();
        self._prev_end = 0;
        self._rewards = {DifferentialReward():0, DivergencePenalty():1};

    def _state_space(self, gym = False):
        return dict(shape=(2*(self.p['dimension']+1)*self.p['population_size']), type='float');

    def _action_space(self, gym = False):
        return dict(type='float', shape=(self.p['dimension']));

    def _network(self):
        l2_reg = 0.01
        return [
            dict(type='dense', size=64, l2_regularization=l2_reg),
            dict(type='dense', size=64, l2_regularization=l2_reg),
        ];

    def _init_agent(self):
        self._agent = PPOAgent(
            states = self._state_space(),
            actions = self._action_space(),
            network = self._network(),
            # Agent
            states_preprocessing=None,
            actions_exploration=None,
            reward_preprocessing=None,
            # MemoryModel
            update_mode=dict(
                unit='episodes',
                # 10 episodes per update
                batch_size=3,
                # Every 10 episodes
                frequency=4
            ),
            memory=dict(
                type='latest',
                include_next_states=False,
                capacity=2000
            ),
            # DistributionModel
            distributions=None,
            entropy_regularization=0.03,
            # PGModel
            baseline_mode='states',
            baseline=dict(
                type='mlp',
                sizes=[64,64]
            ),
            discount = 0.95,
            baseline_optimizer=dict(
                type='multi_step',
                optimizer=dict(
                    type='adam',
                    learning_rate=1e-5
                ),
                num_steps=5
            ),
            gae_lambda=0.97,
            # PGLRModel
            likelihood_ratio_clipping=0.05,
            # PPOAgent
            step_optimizer=dict(
                type='adam',
                learning_rate=1e-4
            ),
            subsampling_fraction=0.2,
            optimization_steps=50
        )

    def _reset(self, initial_mean, initial_covariance):
        #self._reset_baseline(problem, initial_mean);
        for reward in self._rewards:
            reward.reset();
        self._agent.reset();
        self._step = 0;

        self._prev_state = np.zeros((self.p['population_size']*(self.p['dimension']+1)))
        self._mean = initial_mean;
        self._should_observe = False;

    def _calculate(self, population, fitness):
        self._observe(population, fitness);

        state = self._calculate_state(population, fitness);
        mean_difference = self._agent.act(states=state.flatten(), deterministic= False);
        self._target = self._mean + mean_difference;
        if np.any(np.isnan(self._target)):
            print(self._target);
            raise Exception("mean is nan");
        if np.any(np.isinf(self._target)):
            print(self._target);
            raise Exception("mean is inf");
        self._step += 1;
        return self._target;

    def _calculate_reward(self, population, fitness):
        reward = 0;
        for reward_fn,w in self._rewards.items():
            reward += w * reward_fn(population ,fitness);
        return reward;

    def _observe(self, population, fitness):
        reward = self._calculate_reward(population, fitness);

        if self._should_observe:
            self._agent.observe(False, reward);
        else:
            self._should_observe = True;

    def _calculate_state(self, population, fitness):
        state = population-self._mean;
        norm_fitness = (fitness- np.mean(fitness))/np.std(fitness);
        state = np.append(state, norm_fitness[:, None], axis=1);
        state = state[fitness.argsort(),:].flatten();
        if np.any(np.isnan(state)):
            print("state is nan");
        if np.any(np.isinf(state)):
            print("state is inf");

        total_state = np.stack([state,self._prev_state])
        self._prev_state = state;
        return total_state.flatten();

    def _update_mean(self, mean):
        self._mean = mean;

    def _terminating(self, population, fitness):
        mean_fitness = np.mean(fitness);
        end = mean_fitness;
        state = self._calculate_state(population, fitness);
        if False and end < self._prev_end:
            self._agent.observe(True, 10);
        else:
            self._agent.observe(True, 0);
        self._prev_end = end;
