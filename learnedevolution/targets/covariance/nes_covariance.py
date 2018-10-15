import numpy as np;

from .covariance_target import CovarianceTarget;

class NESCovariance(CovarianceTarget):
    _API=2.
    def __init__(self, eta_sigma=0.5, eta_B=0.7, epsilon = 1e-30):
        self.eta_sigma = eta_sigma;
        self.eta_B = eta_B;
        self.epsilon = epsilon;

    def _reset(self, initial_mean, initial_covariance):
        A = np.linalg.cholesky(initial_covariance);
        self._sigma = np.linalg.det(A)**(1./len(initial_mean));
        self._B = A/self._sigma;

    def _update_mean(self, mean):
        self._old_mean = self._mean;
        self._mean = mean;
        return;
    def _calculate(self, population):
        I=np.eye(population.dimension);
        sorted_idx = np.argsort(population.fitness)[::-1];

        #Calculate utility
        N = population._N;
        ks = np.arange(1,N+1);
        numerator = np.max(np.stack([np.zeros(N), np.log(.5*N+1)-np.log(ks)]),axis=0)
        u = numerator/np.sum(numerator)-1./N

        # Calculate dM
        dM = np.zeros(shape=(population.dimension,population.dimension));
        for k in range(N):
            i = sorted_idx[k];
            dM += u[k]*(np.outer(population.raw_population[i],population.raw_population[i])-I);

        # calculate dsigma
        dsigma = np.trace(dM)/population.dimension;

        # calculate dB
        dB = dM-dsigma*I;

        # calculate new B and sigma;
        self._sigma = self._sigma*np.exp(.5*self.eta_sigma*dsigma);
        self._B = self._B*np.exp(.5*self.eta_B*dB);

        # calculate covariance
        self._covariance = self._sigma**2*np.dot(self._B,self._B)

        return self._covariance;

    def _calculate_deterministic(self,population):
        return self._calculate(population);
    def _terminating(self, population):
        pass;
