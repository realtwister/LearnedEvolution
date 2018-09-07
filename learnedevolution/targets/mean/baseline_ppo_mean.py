import numpy as np;
from gym.spaces import Box;

from baselines.ppo1 import PPO, BatchProvider;
from ...policy.mlp_policy import MlpPolicy

from .mean_target import MeanTarget;
from ...states.normalized_state import NormalizedState;
from ...states.new_normalized_state import NewNormalizedState;
from ...states.benchmark_state import BenchmarkState;

class BaselinePPOMean(MeanTarget):
    _API = 2.;
    def __init__(self, dimension, population_size, rewards, convergence_criteria, logdir = None):
        super().__init__();
        self.p['population_size'] = population_size;
        self.p['dimension'] = dimension;
        self._init_agent(logdir);
        self._prev_end = 0;

        self._rewards = rewards
        self._state = NormalizedState(population_size,dimension);
        self._convergence_criteria = convergence_criteria;
        self._non_converged = -1000;
        self.learning = True;

    def _state_space(self, gym = False):
        if gym:
            return Box(high = 10, low  = -10,shape=(2*(self.p['dimension']+1)*self.p['population_size'],));
        return dict(shape=(2*(self.p['dimension']+1)*self.p['population_size']), type='float');

    def _action_space(self, gym = False):
        if gym:
            return Box(high = 100, low= -100, shape=(self.p['dimension'], ));
        return dict(type='float', shape=(self.p['dimension']));

    def _init_agent(self, log_dir):
        def policy_fn(name, ob_space, ac_space, summaries = False, should_act = True):
            self._policy = MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,hid_size=128, num_hid_layers=2, summaries= summaries, should_act= should_act);
            return self._policy;
        batch = BatchProvider(epochs = 4, horizon = 100, reward_discount = 0.95);
        self._agent = PPO(None, policy_fn, batch,
            clip_param = 0.03,
            adam_epsilon = 1e-5,
            entropy_param = 0.05,
            value_param=1.,
            observation_space = self._state_space(True),
            action_space = self._action_space(True),
            log_dir = log_dir);

    def _reset(self, initial_mean, initial_covariance):
        #self._reset_baseline(problem, initial_mean);
        for reward in self._rewards:
            reward.reset();

        # if np.abs(self._convergence_criteria[0].convergence_ratio)<0.25:
        #     self._non_converged +=1;
        #     if self._non_converged >= 300:
        #         self._agent._policy.decrease_logstd(self._agent._session);
        #         self._non_converged = 0;
        #
        # else:
        #     self._non_converged = 0;


        self._step = 0;
        self._current_rewards = [];

        self._prev_state = np.zeros((self.p['population_size']*(self.p['dimension']+1)))

        self._should_observe = False;
        self._mean = initial_mean;
        self._covariance = initial_covariance;
        self._state.reset();

    def _calculate(self, population):
        if self.learning:
            self._observe(population);
            action,_ = self._agent.act();
        else:
            self._current_reward = reward = self._calculate_reward(population);
            self._current_state = state = self._state.encode(population);
            self._current_rewards += [reward];
            action,_ = self._agent._policy.act(True, state);
        self._target = self._state.decode(action);
        if np.any(np.isnan(self._target)):
            print(self._target);
            raise Exception("mean is nan");
        if np.any(np.isinf(self._target)):
            print(self._target);
            raise Exception("mean is inf");
        self._step += 1;

        return self._target;

    def _calculate_deterministic(self,population):
        self._current_reward = reward = self._calculate_reward(population);
        self._current_state = state = self._state.encode(population);
        self._current_rewards += [reward];
        action,_ = self._agent._policy.act(False, state);
        self._target = self._state.decode(action);
        self._step +=1;
        return self._target;



    def _observe(self, population):
        self._current_reward = reward = self._calculate_reward(population);
        self._current_state = state = self._state.encode(population);
        self._current_rewards += [reward];

        if self._should_observe:
            self._agent.observe(state, reward, False);
        else:
            self._agent.reset(state);
            self._should_observe = True;

    def _calculate_reward(self, population):
        reward = 0;
        for reward_fn,w in self._rewards.items():
            reward += w * reward_fn(population.population, population.fitness);
        return reward;

    def _calculate_state(self, population):
        return self._state.encode(population);
        norm_factor = np.sqrt(np.linalg.eig(self._covariance)[0][0]);
        state = (population-self._mean)/norm_factor;
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

    def _update_covariance(self, covariance):
        self._covariance = covariance;

    def _terminating(self, population):
        reward = 0;
        mean_reward = np.percentile(self._current_rewards,90);
        for criterion in self._convergence_criteria:
            reward += criterion.reward;
        self._current_reward += reward;
        if self.learning:
            state = self._calculate_state(population);
            self._agent.observe(state, reward, True);

    def close(self):
        self._agent.close();
