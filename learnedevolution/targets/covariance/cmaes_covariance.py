import numpy as np;

from .covariance_target import CovarianceTarget;

class CMAESCovariance(CovarianceTarget):
    _API=2.
    def __init__(self,
        cc=None,
        c1 = None,
        cmu = None,
        cs = None,
        ds = None,
        w = None,
        epsilon = 1e-20):
        self.epsilon = epsilon;

        self._cc = cc;
        self._c1 = c1;
        self._cmu = cmu;
        self._ds = ds;
        self._cs = cs;

        self._w = w;

    def _reset(self, initial_mean, initial_covariance):
        self.mean = initial_mean;
        self.old_mean = initial_mean;

        self.covariance = initial_covariance;

        self.d = len(initial_mean);

        # necessary initialization for C
        self.pc = np.zeros(self.d);
        self.sigma = 1.;
        self.C = np.eye(self.d);

        self.mu_eff = None;

        self.cc = self._cc;

        self.c1 = self._c1;
        if self.c1 is None:
            self.c1 = 2./self.d**2;
        self.cmu = self._cmu;
        if self.cmu is None:
            self.cmu = 1 - self.c1;


        # Necessary init for step-size
        self.ps = np.zeros(self.d);
        self.ds = self._ds;
        self.cs = self._cs;



    def w(self, N):
        if self._w is not None:
            return self._w(N);
        ks = np.arange(1,N+1);
        numerator = np.max(np.stack([np.zeros(N), np.log(.5*N+1)-np.log(ks)]),axis=0)
        return numerator/np.sum(numerator);


    def _update_mean(self, mean):
        self.old_mean = self.mean;
        self.mean = mean;

    def _calculate(self, population):
        sorted_idx = np.argsort(population.fitness)[::-1];

        # Initiate mu_eff;
        if self.mu_eff is None:
            self.current_w = self.w(len(population));
            self.mu_eff = 1./np.sum(self.current_w**2);
            if self._cc is None:
                self.cc = (4+self.mu_eff/self.d)/(self.d+4+2.*self.mu_eff/self.d);

            if self._cmu is None:
                self.cmu = min(self.cmu, self.mu_eff/self.d**2);

            if self.cs is None:
                self.cs = (self.mu_eff+2)/(self.d+self.mu_eff+5.);
            if self.ds is None:
                self.ds = 1+ 2.*max(0.,np.sqrt((self.mu_eff-1)/(self.d+1))-1)+self.cs;

        self.y = (population.population[sorted_idx]-population.mean)/self.sigma;
        self.yw = np.sum(self.current_w[:,None]*(population.population[sorted_idx]-population.mean), axis=0);


        self.update_pc();
        self.update_C();

        self.update_ps(population);
        self.update_sigma();

        return self.sigma**2*self.C;

    def update_pc(self):
        self.pc *=(1-self.cc);
        self.pc += np.sqrt((1-(1-self.cc)**2)*self.mu_eff)*(self.mean-self.old_mean)/self.sigma;

    def update_C(self):
        self.C *= (1.-self.c1-self.cmu);
        self.C += self.c1*np.outer(self.pc,self.pc);
        rankmu = np.zeros((self.d,self.d));
        for w,y in zip(self.current_w,self.y):
            rankmu += w*np.outer(y,y);
        self.C += self.cmu*rankmu;

    def update_ps(self, population):
        u,s,h = population.svd;
        self.ps *= (1-self.cs);
        self.ps += np.sqrt((1-(1-self.cs)**2)*self.mu_eff)*np.dot(self.mean-self.old_mean,u)/np.sqrt(s);

    def update_sigma(self):
        self.sigma *= np.exp(self.cs/self.ds*(np.linalg.norm(self.ps)/(np.sqrt(self.d)-1./(4*self.d))-1.));



    def _calculate_deterministic(self,population):
        return self._calculate(population);

    def _terminating(self, population):
        pass;
