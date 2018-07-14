import tensorflow as tf;

from .encoding import Encoding;

class ConnectedEncoding(Encoding):
    def __init__(self, dimension, population_size = 100, history_size = 2):
        self._population_size = population_size;
        self._history_size = history_size;
        self._dimension = dimension;

    def setup_tensorflow(self):
        self._placeholder = tf.placeholder(shape=(None, self._population_size*(self._dimension+1)*self._history_size), dtype= tf.float32);
        return self._placeholder;

    def feeddict(self, population, fitness):
        return {self._placeholder: self._calculate_state(population, fitness)};

    @property
    def state(self):
