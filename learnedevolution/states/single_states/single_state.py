from ...utils.parse_config import ParseConfig

class SingleState(ParseConfig):

    def invert(self, x, reference = None):
        raise NotImplementedError()

    def __call__(self, population, reference = None):
        return self._calculate(population, reference)

    def _calculate(self, population, reference = None):
        raise NotImplementedError()

    @property
    def state_space(self):
        raise NotImplementedError()

    @property
    def action_space(self):
        raise NotImplementedError()
