import numpy as np;

class Population:
    def __init__(self, mean, covariance, N, fitness_fn):
        self._mean = mean;
        self._covariance = covariance;
        self._N = N;
        self._fitness_fn = fitness_fn;
        self._d = len(mean);

        self._raw_population = None;
        self._population = None;
        self._fitness = None;
        self._svd = None;

        self._seed();

    def _seed(self, *, seed = None, random_state = None):
        if random_state is not None:
            self._random = random_state;
        else:
            self._random = np.random.RandomState(seed);

    def _sample(self):
        self._raw_population = self._random.normal(size = [self._N, self._d]);
        self._population = None;
        self._fitness = None;

    def log_derivative(self):
        ic = np.linalg.pinv(self._covariance);
        d = self.population-self._mean;
        dmu = d@ ic
        dsigma = -.5*ic+.5*((ic@d.T[None,:,:]).T@d[:,None,:]@ic);
        return dmu,dsigma;


    def _calculate(self):
        u,s,h = self.svd;
        x = self.raw_population;
        x = self.morph(x.T).T;
        x += self._mean;
        self._population = x;
        self._fitness = None;

    def _evaluate(self):
        self._fitness = self._fitness_fn(self.population);

    def morph(self, x):
        u,s,h = self.svd;
        x = np.dot(  np.sqrt(s)* u,x);
        return x;


    def invert(self, population):
        if isinstance(population, self.__class__):
            population = population.population;
        u,s,h = self.svd;

        return np.dot((population-self._mean),u/np.sqrt(s));



    @property
    def raw_population(self):
        if self._raw_population is None:
            self._sample();
        return self._raw_population;

    @property
    def population(self):
        if self._population is None:
            self._calculate();
        return self._population;

    @property
    def fitness(self):
        if self._fitness is None:
            self._evaluate();
        return self._fitness;

    @property
    def svd(self):
        if self._svd is None:
            self._svd = np.linalg.svd(self._covariance);
        return self._svd;

    @property
    def mean(self):
        return self._mean;

    @property
    def covariance(self):
        return self._covariance;
