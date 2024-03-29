import numpy as np;
from collections import namedtuple;

import learnedevolution.utils.logging as logging;
from learnedevolution.protos.population_pb2 import Population as Population_proto


log = logging.child('population');

PopulationTuple = namedtuple('PopulationTuple',['size','dimension','mean','covariance', 'fitnessSummary', 'population', 'fitness'])
FitnessSummaryTuple = namedtuple('FitnessSummaryTuple',['median', 'mean', 'min', 'max','std'])

class Population:
    def __init__(self, mean, covariance, population_size, fitness_fn, epsilon =1e-30, random_state = None):
        self._mean = mean;
        self._covariance = covariance;
        self._N = population_size;
        self._fitness_fn = fitness_fn;
        self._d = len(mean);
        self._epsilon = epsilon;

        self._raw_population = None;
        self._population = None;
        self._fitness = None;
        self._svd = None;

        self._random = random_state
        if self._random is None:
            self._random = np.random.RandomState()

    def _seed(self, *, seed = None, random_state = None):
        if random_state is not None:
            self._random = random_state;
        else:
            self._random = np.random.RandomState(seed);
        #self._reset_random = self._random.copy();

    #def reset(self):
        #self._random = self._reset_random.copy();

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
        factor = np.sqrt(s);
        if len(x.shape) == 2:
            factor = factor[:,None];
        x = np.dot(  u,factor*x);
        return x;


    def invert(self, population):
        if isinstance(population, self.__class__):
            if self == population:
                return self.raw_population;
            population = population.population;
        u,s,h = self.svd;
        return np.dot((population-self._mean),u)/np.sqrt(s)[None,:];

    def invert_individual(self, individual):
        u,s,h = self.svd;
        delta = np.dot((individual-self._mean), u)/np.sqrt(s)[:];
        return delta;

    def __len__(self):
        return self._N;



    @property
    def raw_population(self):
        if self._raw_population is None:
            self._sample();
        return self._raw_population;

    @property
    def dimension(self):
        return len(self._mean);

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
            B,D,BT = np.linalg.svd(self._covariance);
            if np.any(np.sqrt(D)<self._epsilon):
                log.debug('Population eigenvalue smaller than epsilon(={})\n'.format(self._epsilon));
                D = np.clip(D, self._epsilon, None);
            self._svd = (B,D,BT);
        return self._svd;

    @property
    def mean(self):
        return self._mean;

    @property
    def covariance(self):
        return self._covariance;

    def proto(self, minified = True):
        proto = Population_proto();
        proto.size = self._N;
        proto.dimension = self._d;
        proto.mean.extend(self._mean)
        proto.covariance.extend(self._covariance.flatten());

        fitness = np.array(self.fitness);
        fitness_summary = proto.fitnessSummary;
        fitness_summary.min = fitness.min();
        fitness_summary.max = fitness.max();
        fitness_summary.mean = fitness.mean();
        fitness_summary.std = fitness.std();

        if not minified:
            proto.population = self.population.tobytes();
            proto.fitness = self.fitness.tobytes();

        #DEBUG
        # for descr, value in fitness_summary.ListFields():
        #     break;
        # else:
        #     print(proto);
        #     print(proto.SerializeToString());
        #     print(fitness);
        #     print(fitness.mean());
        #     fitness_summary.mean = fitness.mean();
        #     print(proto)
        #     print(fitness.tobytes());
        #     input();
        #ENDDEBUG


        return proto;

    @staticmethod
    def read_proto(proto):
        proto = Population_proto.FromString(proto);
        mean = np.array(proto.mean);
        covariance = np.array(proto.covariance).reshape((proto.dimension,proto.dimension));
        N = proto.size;

        population = np.array(proto.population);
        try:
            population = population.reshape((proto.size, proto.dimension));
        except:
            pass;
        fitness = np.array(proto.fitness);

        fs = proto.fitnessSummary;
        fitnessSummary =  FitnessSummaryTuple(
            fs.median,
            fs.mean,
            fs.min,
            fs.max,
            fs.std);
        return PopulationTuple(
                proto.size,
                proto.dimension,
                mean,
                covariance,
                fitnessSummary,
                population,
                fitness);
