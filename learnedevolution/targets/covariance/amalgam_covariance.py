import numpy as np;

from .covariance_target import CovarianceTarget;

class AMaLGaMCovariance(CovarianceTarget):
    _API=2.
    def __init__(self,
        theta_SDR = 1.,
        eta_DEC = 0.9,
        alpha_Sigma = [-1.1,1.2,1.6],
        NIS_MAX = 25,
        tau = 0.35,
        epsilon = 1e-20):

        self.epsilon = epsilon;
        self.theta_SDR = theta_SDR;
        self.eta_DEC = eta_DEC;
        self.eta_INC = 1./eta_DEC;
        self.NIS_MAX = NIS_MAX;
        self.alpha_Sigma = alpha_Sigma;
        self.tau = tau;

    def _reset(self, initial_mean, initial_covariance):
        self.mean = initial_mean;
        self.old_mean = initial_mean;

        self.covariance = initial_covariance;

        self.d = len(initial_mean);

        self.Sigma = initial_covariance;
        self.c_multiplier = 1.;
        self.NIS = 0;
        self.t = 0;

        self.best_f = -float('inf');


    def _update_mean(self, mean):
        self.old_mean = self.mean;
        self.mean = mean;

    def _calculate(self, population):
        self.update_matrix(population);
        self.update_multiplier(population);
        self.t += 1;
        self.best_f = max(self.best_f, np.max(population.fitness));

        self.covariance = self.Sigma*self.c_multiplier;

        return self.covariance;

    def update_matrix(self, population):
        F = population.fitness;
        sel_idx = F.argsort()[-np.ceil(self.tau*len(population)).astype(int):][::-1]

        alpha = self.alpha_Sigma;
        eta_Sigma = 1.-np.exp(alpha[0]*len(sel_idx)**alpha[1]/self.d**alpha[2]);

        current_update = np.zeros((self.d,self.d));
        selection = population.population[sel_idx];
        for individual in selection:
            delta = individual-self.old_mean;
            current_update += np.outer(delta,delta)
        current_update /= (selection.shape[0]);

        self.Sigma *= (1-eta_Sigma);
        self.Sigma += eta_Sigma*current_update;

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
        delta = np.abs(self.mean-x_avg);
        variances = np.abs(np.diag(self.covariance));
        if np.any(delta/np.sqrt(variances)>self.theta_SDR):
            self.c_multiplier *= self.eta_INC;







    def _terminating(self, population):
        pass;
