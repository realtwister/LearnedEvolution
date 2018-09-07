from .logger import Logger;
from ..algorithm import Algorithm;
from .recorder import Recorder;

class AlgorithmLogger(Logger):
    def __init__(self, algorithm, logdir):
        assert isinstance(algorithm, Algorithm);
        Logger.__init__(self, algorithm, logdir);

        self.init_recorder();

    def init_recorder(self):
        self.recorder.watch('_population','population');
        self.recorder.watch('_old_mean','mean');
        self.recorder.watch('_old_covariance','covariance');
        self.recorder.watch('_evaluated_fitness','fitness');
