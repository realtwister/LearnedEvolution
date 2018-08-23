import numpy as np;

from .reward import Reward;
class FitnessReward(Reward):

    def _calculate(self, population, fitness):

        return np.mean(fitness);
