from ..utils.signals import method_event;

class Reward(object):

    def _reset(self):
        pass;

    @method_event('reset')
    def reset(self):
        self._reset();

    def _calculate(self, population, fitness):
        raise NotImplementedError;

    @method_event('call')
    def __call__(self, population, fitness):
        return self._calculate(population,fitness);
