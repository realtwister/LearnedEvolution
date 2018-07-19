
import tensorflow as tf;
from collections import defaultdict;

from .writer import Writer;
from .recorder import Recorder;
from learnedevolution.utils.signals import TimedCallback;

import logging;
log = logging.getLogger(__name__)



class Logger(object):
    def __init__(self, instance, logdir, algorithm = None, prefix=""):
        self._instance = instance;
        self._writer = Writer(logdir);

        self._watching = defaultdict(dict);
        self._prefix = prefix;
        self._algorithm = instance;

        if algorithm is not None:
            self._algorithm = algorithm;
        self._recorder = Recorder(self);
        self._children =[];

    def record(self, depth = 100, suffix = None, metadata={}):
        if depth>0:
            for child in self._children:
                child.record(depth-1,suffix, metadata);
        self.recorder.record(suffix, metadata);




    def create_child(self, instance):
        child = Logger(instance, self._writer.writer.get_logdir(), self._algorithm, self._prefix+instance.__class__.__name__+".");
        self._children += [child];
        return child;


    def watch_scalar(self, variable, event, tag = None, once_in = 100, offset = -1):
        if variable in self._watching and 'scalar' in self._watching[variable]:
            raise Exception('Variable "{}" is already being watched'.format(variable));
        if tag is None:
            tag = variable;

        log.info("Watching {} with tag {}".format(variable, tag));

        def _watcher(*args, **kwargs):
            self._writer.add_scalar(self._prefix+tag, getattr(self._instance,variable), self._algorithm.current_step);
        tc = TimedCallback(event,  sender = self._algorithm, fns = _watcher, once_in = once_in, offset = offset);
        self._watching[variable]['scalar'] = tc;
        tc.connect();

    def watch_histogram(self, variable, event, tag = None, once_in = 100, offset = -1):
        if variable in self._watching and 'histogram' in self._watching[variable]:
            raise Exception('Variable "{}" is already being watched'.format(variable));
        if tag is None:
            tag = variable;

        log.info("Watching {} with tag {}".format(variable, tag));

        def _watcher(*args, **kwargs):
            self._writer.add_histogram(self._prefix+tag, getattr(self._instance,variable), self._algorithm.current_step);
        tc = TimedCallback(event,  sender = self._algorithm, fns = _watcher, once_in = once_in, offset = offset);
        self._watching[variable]['histogram'] = tc;
        tc.connect();



    @property
    def writer(self):
        self._writer.writer;

    @writer.setter
    def writer(self, writer):
        raise ValueError('writer is a protected property');

    @property
    def recorder(self):
        return self._recorder;
