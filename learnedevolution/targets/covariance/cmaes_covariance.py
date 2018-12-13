import numpy as np;

from .covariance_target import CovarianceTarget;

class CMAESCovariance(CovarianceTarget):
    _API=2.
    def __init__(self,
        dimension,
        weights,
        c_sigma,
        d_sigma,
        c_c,
        c_1,
        c_mu,
        epsilon = 1e-20):
        self.epsilon = epsilon;

        self.d = dimension
        self._EN = None

        self.w = weights

        self.c_sigma = c_sigma
        self.d_sigma = d_sigma

        self.c_c = c_c
        self.c_1 = c_1
        self.c_mu = c_mu



    def _reset(self, initial_mean, initial_covariance):
        self.g = 0
        # Necessary init for step-size
        self.p_sigma = np.zeros(self.d)
        self.log_sigma = 0

        self.p_c = np.zeros(self.d)

    def _calculate(self, population):
        sorted_idx = np.argsort(population.fitness)[::-1]

        if callable(self.w):
            w = [self.w(i, len(population)) for i in range(len(population))]
        else:
            w = list(self.w)
        assert len(w) >= len(population)
        w = np.array([max(0,w) for weight in w])

        mu_eff = 1/np.sum(w**2)

        xw = np.sum(w*population.raw_population[sorted_idx])

        self.update_p_sigma(population, xw, mu_eff)

        self.update_p_c(population, xw, mu_eff)

        #NOTE: need to update C before sigma since we need the old sigma to calculate C

        self.update_C(population, sorted_idx, w)

        self.update_log_sigma()

        self.g +=1


        return np.exp(2*self.log_sigma)*self.C;

    def update_p_sigma(self, population, xw, mu_eff):
        B,_* =  population.svd
        self.p_sigma *= (1-self.c_sigma)
        self.p_sigma += np.sqrt(self.c_sigma*(2-self.c_sigma)*mu_eff)*B@xw


    def update_log_sigma(self):
        exp = self.c_sigma/self.d_sigma
        exp *= np.linalg.norm(self.p_sigma)/self.EN - 1
        self.log_sigma += exp

    def update_p_c(self, population, xw, mu_eff):
        self.p_c *= (1-self.c_c)
        self.p_c += self.h_sigma * np.sqrt(self.c_c*(2-self.c_c)*mu_eff)*population.morph(xw)

    def update_C(self,population, sorted_idx, w):
        self.C *= 1 + self.c_1*self.d_h_sigma -  self.c_1 - self.c_mu* np.sum(w)
        self.C += self.c_1 * np.outer(self.p_c,self.p_c)

        last_term = 0
        B, D, _ = population.svd
        for weight,idx in zip(w,sorted_idx):
            if weight < 0:
                weight *= self.d/ np.linalg.norm(population.raw_population[idx])

            y = (population.population[idx]-population.m)/np.exp(self.log_sigma)
            last_term += np.outer(y,y)
        self.C += self.c_mu * last_term




    @property
    def EN(self):
        if self._EN is None:
            EN = np.sqrt(self.d)*(1.-1./(4.*self.d)+1./(21.*self.d**2))
            self._EN = EN
        return self._EN

    @property
    def h_sigma(self):
        rhs = (1.4+2/(d+1))*self.EN
        rhs *= np.sqrt(1-(1-self.c_sigma)**(2*(g+1)))
        return int(np.linalg.norm(self.p_sigma)<rhs)

    @property
    def d_h_sigma(self):
        return (1-self.h_sigma)*self.c_c*(2-self.c_c)



    def _calculate_deterministic(self,population):
        return self._calculate(population);

    def _terminating(self, population):
        pass;

    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'weights',
            'c_sigma',
            'd_sigma',
            'c_c',
            'c_1',
            'c_mu',
            'condition_number_epsilon'
        )
        cls._config_defaults(
            weights = lambda i,l: (np.log((l+1)/2)-np.log(i+1))/np.sum([max(0,np.log((l+1)/2)-np.log(i)) for i in range(1,l+1)]),
            c_sigma = 0.2,
            d_sigma = 1,
            c_c = 0.6,
            c_1 = .1,
            c_mu = .9,
            epsilon = 1e-20
        )

        return super()._get_kwargs(config, key = key);
