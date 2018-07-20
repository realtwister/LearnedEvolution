import numpy as np
from .protos.problem_pb2 import Problem as ProblemProto;

from ..utils.random import RandomGeneratable;


class Problem(RandomGeneratable):
    """Abstract problem class to provide an interface for the algorithms"""

    _type = "NONE";
    def __init__(self, dimension):
        # set properties
        self._dimension = dimension;

        self._evaluations = 0;


    def fitness(self, xs):
        self._evaluations += xs.shape[0];

    @property
    def optimum(self):
        raise NotImplementedError();

    @property
    def dimension(self):
        return self._dimension;

    @property
    def type(self):
        return self._type

    @property
    def proto(self):
        proto = ProblemProto();
        proto.type = self._type;
        proto.dimension = self._dimension;
        for key,value in self._params.items():
            proto.params.add(key=key, value = value);
        return proto;

    def __str__(self):
        res = "------- Problem -------\n"
        res += " Type:\t\t {}\n".format(self._type);
        res += " Parameters:\t {}".format(self._params);
        return res;

    def __eq__(self, other):
        if self._type != other._type:
            return False;
        for key in self._params:
            if key not in other._params:
                return False;
            if np.any(self._params[key] != other._params[key]):
                return False;
        return True;
