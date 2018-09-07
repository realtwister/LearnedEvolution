from ..utils.signals import method_event;

class State(object):

    def _reset(self):
        pass;

    method_event('reset')
    def reset(self):
        self._reset();

    def _encode(self, population):
        raise NotImplementedError;

    method_event('encode')
    def encode(self, population):
        return self._encode(population);

    def _decode(self, action):
        return action;

    method_event('decode')
    def decode(self, action):
        return self._decode(action);
