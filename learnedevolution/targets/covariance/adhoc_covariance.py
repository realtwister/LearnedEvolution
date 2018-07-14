import numpy as np;

from .covariance_target import CovarianceTarget;

class AdhocCovariance(CovarianceTarget):
    def __init__(self,
        selection_ratio = 0.5,
        max_no_improvement_stretch = 5,
        threshold = 1,
        inc = 1.1,
        dec = 0.9):
        CovarianceTarget.__init__(self);
        self._selection_ratio = selection_ratio;
        self._max_no_improvement_stretch  = max_no_improvement_stretch;
        self._threshold = threshold;
        self._inc = inc;
        self._dec = dec;

    def _reset(self, initial_mean, initial_covariance ):
        self._multiplier = 1;
        self._no_improvement_stretch = 0;
        self._prev_best = None;

    def _calculate(self, population, fitness):
        population =  population[np.argsort(fitness)];
        self._update_multiplier(population,fitness);

        N_select = int(population.shape[0]*self._selection_ratio);
        selected = population[:N_select];

        self._target = np.cov(selected.T);
        return self._multiplier * self._target;

    def _update_multiplier(self, population, fitness):
        if self._prev_best is not None:
            if self._prev_best > fitness[0]:
                self._no_improvement_stretch = 0;
                self._multiplier = max(self._multiplier,1)
                average_improved = np.mean(population[(fitness< self._prev_best),:-1], axis=0);
                L = np.linalg.cholesky(self._multiplier*self._target);
                SDR = np.max(np.abs(np.linalg.solve(L, (average_improved-self._mean))));
                if SDR> self._threshold:
                    self._multiplier = self._multiplier;

            else:
                if self._multiplier <= 1:
                    self._no_improvement_stretch += 1;
                if self._multiplier > 1 or self._no_improvement_stretch >= self._max_no_improvement_stretch:
                    self._multiplier = self._multiplier*self._dec;

                if self._multiplier < 1 and self._no_improvement_stretch <self._max_no_improvement_stretch:
                    self._multiplier = 1;


        self._prev_best = fitness[0];




    def update_mean(self, mean):
        self._mean = mean;
