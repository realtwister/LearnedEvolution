import numpy as np;
from scipy.stats import special_ortho_group

from .problem import Problem;
from ..utils.random import RandomGeneratable, RandomGenerator;

class SinusoidalNoiseProblem:
    def __init__(self,* ,
        period = 0.2,
        amplitude = 0.1):
        self._period = period;
        self._amplitude = amplitude;

    def __call__(self, problem_cls):
        self._problem = problem_cls;
        return self;

    def random(self, random_state, dimension,**kwargs):
        return DiscretizedProblem.random(random_state, dimension, self._problem, self._period, self._amplitude, **kwargs)

class SinusoidalNoisedProblem(Problem, RandomGeneratable):

    def __init__(self, parent, period, amplitude):
        assert isinstance(parent, Problem);

        super().__init__(parent.dimension);
        self._parent = parent;
        self._type = "Sinusoidal_"+parent.type;
        self._params = dict(
            period = period,
            amplitude = amplitude,
        );

    @staticmethod
    def random(random_state, dimension, problem_cls, period = 0.2, amplitude=0.1,  **kwargs):
        problem = problem_cls.random(random_state, dimension, **kwargs);

        if isinstance(period, tuple):
            delta = (period[1]- period[0]);

            period = delta*random_state.rand()+period[0];

        if isinstance(amplitude, tuple):
            delta = (amplitude[1]- amplitude[0]);

            amplitude = delta*random_state.rand()+amplitude[0];

        return DiscretizedProblem(problem, period, amplitude);

    def fitness(self, xs):
        super().fitness(xs);
        delta= xs-self._parent.optimum;
        sinus = np.cos(np.linalg.norm(delta, axis=-1)/self._params['period']*2*np.pi);
        return self._parent.fitness(xs)+self._params['amplitude']*sinus- self._params['amplitude'];

    @property
    def optimum(self):
        return self._parent.optimum;
