import numpy as np
from .covariance_target import CovarianceTarget;

class CMAESCovariance(CovarianceTarget):
    _API=2.
    def __init__(self,
        population_size,
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
        self.population_size = population_size


        self.w = weights

        if callable(self.w):
            w = [self.w(i, population_size) for i in range(population_size)]
        else:
            w = list(self.w)
        assert len(w) >= population_size
        self.w = np.array(w)

        self.c_sigma = c_sigma
        self.d_sigma = d_sigma

        self.c_c = c_c
        self.c_1 = c_1
        self.c_mu = c_mu

        self.reset_cache()


    def reset_cache(self):
        self._EN = None
        self._mu_eff = None

    def _reset(self, initial_mean, initial_covariance):
        self.g = 0
        # Necessary init for step-size
        self.p_sigma = np.zeros(self.d)
        self.log_sigma = 0

        self.p_c = np.zeros(self.d)
        self.C = np.eye(self.d)

    def _calculate(self, population):
        sorted_idx = np.argsort(population.fitness)[::-1]
        w = self.w

        ys = (population.population[sorted_idx]-population.mean)/self.sigma

        yw = self.weighted(ys)


        zw = self.weighted(population.raw_population[sorted_idx])

        self.update_p_c(population, yw)

        #NOTE: need to update C before sigma since we need the old sigma to calculate C

        self.update_C(population, sorted_idx,yw,ys)

        self.update_p_sigma(population, zw)

        self.update_log_sigma()

        self.g +=1


        return self.sigma**2 *self.C;

    def update_p_sigma(self, population, yw):
        self.p_sigma *= (1-self.c_sigma)
        self.p_sigma += np.sqrt(self.c_sigma*(2-self.c_sigma)*self.mu_eff)*yw


    def update_log_sigma(self):
        exp = self.c_sigma/self.d_sigma
        exp *= (np.linalg.norm(self.p_sigma)/self.EN) - 1
        self.log_sigma += exp

    def update_p_c(self, population, yw):
        self.p_c *= (1-self.c_c)
        self.p_c += np.sqrt(self.c_sigma*(2-self.c_c)*self.mu_eff)*yw

    def update_C(self, population, sorted_idx, yw, ys):

        self.C *= (1-self.c_1-self.c_mu)
        self.C += self.c_1*np.outer(self.p_c,self.p_c)

        C_mu = 0
        B, D, _ = population.svd
        for weight,y in zip(self.w,ys):
            if weight <=0:
                continue
            C_mu += weight*np.outer(y,y)
        self.C += self.c_mu * C_mu

    def weighted(self, values):
        return np.sum(self.w[self.w > 0, None]*values[self.w > 0], axis=0)

    @property
    def EN(self):
        if self._EN is None:
            EN = np.sqrt(self.d)*(1.-1./(4.*self.d)+1./(21.*self.d**2))
            self._EN = EN
        return self._EN

    @property
    def h_sigma(self):
        rhs = (1.4+2/(self.d+1))*self.EN
        rhs *= np.sqrt(1-(1-self.c_sigma)**(2*(self.g+1)))
        return int(np.linalg.norm(self.p_sigma)<rhs)

    @property
    def d_h_sigma(self):
        return (1-self.h_sigma)*self.c_c*(2-self.c_c)

    @property
    def sigma(self):
        return np.exp(self.log_sigma)

    @property
    def mu_eff(self):
        if self._mu_eff is None:
            self._mu_eff = 1/np.sum(self.w[self.w>0]**2)
        return self._mu_eff



    def _calculate_deterministic(self,population):
        return self._calculate(population);

    def _terminating(self, population):
        pass;

    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            'dimension',
            'population_size',
            'weights',
            'c_sigma',
            'd_sigma',
            'c_c',
            'c_1',
            'c_mu',
        )
        cls._config_defaults(
            weights = lambda i,l: (np.log((l+1)/2)-np.log(i+1))/np.sum([max(0,np.log((l+1)/2)-np.log(i)) for i in range(1,l+1)]),
            c_sigma = None,
            d_sigma = None,
            c_c = None,
            c_1 = None,
            c_mu = None,
            epsilon = 1e-20
        )

        kwargs =  super()._get_kwargs(config, key = key);
        d = kwargs['dimension']
        w = kwargs['weights']
        if callable(w):
            ws = [w(i, kwargs['population_size']) for i in range(kwargs['population_size'])]
        else:
            ws = list(w)
        ws = np.array(ws)
        mu_eff = 1./np.sum(ws[ws>0]**2)

        if kwargs['c_c'] is None:
            kwargs['c_c'] = (4.+mu_eff/d)/(d+2*mu_eff/d+4)

        if kwargs['c_sigma'] is None:
            kwargs['c_sigma'] = (mu_eff+2.)/(d+mu_eff+5)

        if kwargs['c_1'] is None:
            kwargs['c_1'] = 2./(d+1.3)**2

        if kwargs['c_mu'] is None:
            kwargs['c_mu'] = mu_eff/kwargs['dimension']**2
            if kwargs['c_mu']+kwargs['c_1'] > 1:
                kwargs['c_mu'] = 1.-kwargs['c_1']

        if kwargs['d_sigma'] is None:
            kwargs['d_sigma'] = 1+np.sqrt(mu_eff/kwargs['dimension'])

        return kwargs
