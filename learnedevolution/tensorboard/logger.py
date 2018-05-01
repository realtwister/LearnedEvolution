
import tensorflow as tf;
from collections import defaultdict;

from .writer import Writer;
from learnedevolution.utils.signals import TimedCallback;

import logging;
log = logging.getLogger(__name__)


class Logger(object):
    def __init__(self, instance, logdir):
        self._instance = instance;
        self._writer = Writer(logdir);

        self._watching = defaultdict(dict);


    def watch_scalar(self, variable, event, tag = None, once_in = 100, offset = -1):
        if variable in self._watching and 'scalar' in self._watching[variable]:
            raise Exception('Variable "{}" is already being watched'.format(variable));
        if tag is None:
            tag = variable;

        log.info("Watching {} with tag {}".format(variable, tag));

        def _watcher(*args, **kwargs):
            self._writer.add_scalar(tag, getattr(self._instance,variable), self._instance.current_step);
        tc = TimedCallback(event,  sender = self._instance, fns = _watcher, once_in = once_in, offset = offset);
        self._watching[variable]['scalar'] = tc;
        tc.connect();

    def watch_histogram(self, variable, event, tag = None, once_in = 100, offset = -1):
        if variable in self._watching and 'histogram' in self._watching[variable]:
            raise Exception('Variable "{}" is already being watched'.format(variable));
        if tag is None:
            tag = variable;

        log.info("Watching {} with tag {}".format(variable, tag));

        def _watcher(*args, **kwargs):
            self._writer.add_histogram(tag, getattr(self._instance,variable), self._instance.current_step);
        tc = TimedCallback(event,  sender = self._instance, fns = _watcher, once_in = once_in, offset = offset);
        self._watching[variable]['histogram'] = tc;
        tc.connect();



    @property
    def writer(self):
        self._writer.writer;

    @writer.setter
    def writer(self, writer):
        raise ValueError('writer is a protected property');
