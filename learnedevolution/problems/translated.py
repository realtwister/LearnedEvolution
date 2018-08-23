import numpy as np;

from .problem import Problem;
from ..utils.random import RandomGeneratable, RandomGenerator;

class TranslateProblem:
    def __init__(self, problem_cls, spread= [100,None]):
        self._problem = problem_cls;
        self._spread = spread;

    def random(self, random_state, dimension,**kwargs):
        return TranslatedProblem.random(random_state, dimension, self._problem, self._spread, **kwargs)

class TranslatedProblem(Problem, RandomGeneratable):

    def __init__(self, problem, translation):
        assert isinstance(problem, Problem);

        super().__init__(problem.dimension);
        self._problem = problem;
        self._type = "Translated_"+problem.type;
        self._params = dict(
            translation = translation,
            problem = problem
        );

    @staticmethod
    def random(random_state, dimension, problem_cls, spread = [100,None], **kwargs):
        problem = problem_cls.random(random_state, dimension, **kwargs);
        if hasattr(spread,'__iter__'):
            spread = spread[:dimension];
            if spread[-1] is None:
                spread = spread[:-1]+[0]*(dimension-len(spread)+1);
        else:
            spread = [spread]**dimension;
        assert len(spread) == dimension;
        spread = np.array(spread);
        translation = random_state.rand(dimension)*spread**2;
        return TranslatedProblem(problem, np.sqrt(translation));

    def fitness(self, xs):
        super().fitness(xs);
        xs_translated = xs - self._params['translation'];
        return self._params['problem'].fitness(xs_translated);

    @property
    def optimum(self):
        problem_opt = self._params['problem'].optimum;
        return problem_opt + self._params['translation'];
