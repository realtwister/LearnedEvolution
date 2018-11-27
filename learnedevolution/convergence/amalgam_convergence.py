import numpy as np;

from .convergence import Convergence;

class AMaLGaMConvergence(Convergence):
    def __init__(self,
        threshold = 1e-1,
        theta_SDR = 1.,
        eta_DEC = 0.9,
        NIS_MAX = 5,
        epsilon = 1e-30,
        max_iter = 200,
        reward_per_step =1,
        wait = 0):
        self.threshold = threshold;
        self.epsilon = epsilon;
        self.theta_SDR = theta_SDR;
        self.eta_DEC = eta_DEC;
        self.eta_INC = 1./eta_DEC;
        self.NIS_MAX = NIS_MAX;
        self.max_iter = max_iter;
        self.reward_per_step = reward_per_step;
        self._wait = wait;

    def reset(self, mean, covariance):
        self.c_multiplier = 1.;
        self.NIS = 0;
        self.t = 0;

        self.best_f = -float('inf');


        self._wait -= 1;
        self.converged = False;
        self.iter = 0;

    def __call__(self, population):
        if not self.converged:
            self.converged = self.is_converged(population);
        return self.converged;

    def update_multiplier(self, population):
        if np.any(population.fitness>self.best_f):
            self.NIS = 0;
            self.c_multiplier = max(1., self.c_multiplier);
            self.SDR(population);

        else:
            if self.c_multiplier <= 1:
                self.NIS += 1;
            if self.c_multiplier > 1 or self.NIS >= self.NIS_MAX:
                self.c_multiplier *= self.eta_DEC;
            if self.c_multiplier < 1 and self.NIS < self.NIS_MAX:
                self.c_multiplier = 1;

    def SDR(self, population):
        x_avg = np.mean(population.population[population.fitness>self.best_f], axis=0);
        delta = np.abs(population.mean-x_avg);
        variances = np.abs(np.diag(population.covariance));
        if np.any(delta/np.sqrt(variances)>self.theta_SDR):
            self.c_multiplier *= self.eta_INC;

    def is_converged(self, population):
        self.iter +=1;
        if self.iter >= self.max_iter:
            return True;

        # Check if converging
        if self._wait<0:
            self.update_multiplier(population);
            if self.c_multiplier != 1.:
                print(self.c_multiplier);
            return self.c_multiplier < self.threshold;
        return False;

    @property
    def reward_factor(self):
        if self.converged:
            return (self.max_iter-self.iter);
        return 0;

    @property
    def reward(self):
        return self.reward_factor*self.reward_per_step;

    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'threshold',
            'theta_SDR',
            'eta_DEC',
            'NIS_MAX',
            'epsilon',
            'max_iter',
            'reward_per_step',
            'wait'
        )
        cls._config_defaults(
            threshold = 1e-1,
            theta_SDR = 1.,
            eta_DEC = 0.9,
            NIS_MAX = 5,
            epsilon = 1e-30,
            max_iter = 200,
            reward_per_step =1,
            wait = 0
        )

        return super()._get_kwargs(config, key = key);
