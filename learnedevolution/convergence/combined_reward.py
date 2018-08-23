import numpy as np;

from .time_convergence import TimeConvergence;

class CombinedReward(TimeConvergence):
    def __init__(self, max_iter=100,**kwargs):
        super().__init__(max_iter);
        self.on_init(**kwargs);

    def on_init(self, **kwargs):
        pass;

    def on_reset(self, mean, covariance):
        pass;

    def reset(self, mean, covariance):
        super().reset(mean,covariance);
        self.on_reset(mean, covariance);

    def on_step(self, fitness, mean, covariance):
        pass;

    def calculate_reward(self):
        return 0;


    def __call__(self, fitness, mean, covariance):

        self.on_step(fitness, mean, covariance);
        return super().__call__(fitness, mean, covariance);

    @property
    def reward(self):
        return self.calculate_reward();

class CombinedDifferentialReward(CombinedReward):
    def on_reset(self, mean, covariance):
        self._max_fitnesss = [];


    def on_step(self, fitness, mean, covariance):
        self._max_fitnesss += [np.max(fitness)];

    def calculate_reward(self):
        maxs = np.array(self._max_fitnesss);
        max_all = np.max(maxs);
        maxs = np.log(max_all - maxs + 1e-9);
        D = (maxs[0] - maxs);
        return np.sum(D[1:]/ np.arange(1,len(maxs)));
