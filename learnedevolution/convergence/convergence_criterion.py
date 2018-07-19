import numpy as np;

class ConvergenceCriterion(object):
    def __init__(self,
        max_iter = 100,
        max_streak = 5,
        start_epsilon = 1,
        gamma = 0.01,
        reward_per_step =1):
        self.max_iter = max_iter;
        self.max_streak = max_streak;
        self.epsilon = start_epsilon;
        self.gamma = gamma;
        self.reward_per_step = reward_per_step;
        self.convergence_ratio = 0;

    def reset(self, mean, covariance):
        self.converged = False;
        self.max = -float('Inf');
        self.streak = 0;
        self.iter = 0;
        if self.convergence_ratio > 0.25:
            self.epsilon *= 3/4;
            self.convergence_ratio = 0;
        if self.convergence_ratio < -0.5:
            self.epsilon *= 4/3;
            self.convergence_ratio = 0;

    def __call__(self, fitness, mean, covariance):
        if not self.converged:
            self.converged = self.is_converged(fitness, mean, covariance);
        return self.converged;

    def is_converged(self, fitness, mean, covariance):
        self.iter +=1;

        mean_fitness = np.mean(fitness);
        max_fitness = np.max(fitness);

        # Check if converging
        if max_fitness > self.max:
            self.max = max_fitness;
        if np.abs(self.max-mean_fitness) < self.epsilon:
            self.streak += 1;
            if self.streak >= self.max_streak:
                self.convergence_ratio += self.gamma*(1-self.convergence_ratio);
                return True;
        else:
            self.streak = 0;
        if self.iter >= self.max_iter:
            self.convergence_ratio -= self.gamma*(1+self.convergence_ratio);
            return True;
        return False;

    @property
    def reward_factor(self):
        if self.converged:
            return (self.max_iter-self.iter)+self.streak;
        return 0;

    @property
    def reward(self):
        return self.reward_factor*self.reward_per_step;
