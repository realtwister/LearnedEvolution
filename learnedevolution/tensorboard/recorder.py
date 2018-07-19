import tensorflow as tf;
import numpy as np;
from tensorflow.core.framework import tensor_pb2
from learnedevolution.utils.signals import TimedCallback;

import json;

import logging;
log = logging.getLogger(__name__)

PLUGIN_NAME = 'structevol'

class Recorder(object):
    def __init__(self, logger):
        self._logger = logger;
        self._instance = logger._instance;
        self._algorithm = logger._algorithm;

        self._stop_recording_callback = TimedCallback('after_terminate', fns = self._stop_recording, one_shot = True, sender = self._algorithm);
        self._record_step_callback = TimedCallback('after_step', fns = self._record_step, sender = self._algorithm);
        self.recording = False;

        self._watching = {};

    def watch(self, variable, tag=None):
        if tag is None:
            tag = variable;
        self._watching[variable] = tag;
        log.debug('Recorder for {} is now watching {} as {}'.format(self._instance.__class__.__name__, variable, tag))

    def record(self, suffix = None, metadata={}):
        assert not self.recording, 'Trying to start recording while already recording';
        self.recording = True;
        self._stop_recording_callback.connect();
        self._record_step_callback.connect();
        self._data = {key:[] for key in self._watching};
        log.debug('recording {} for {}'.format(self._data.keys(), self._instance.__class__.__name__));
        if suffix is not None:
            self._suffix = "."+suffix;
        else:
            self._suffix = "";
        self._metadata = metadata;


    def _record_step(self, *args, **kwargs):
        i = self._instance;
        for var in self._watching:
            self._data[var].append(getattr(i, var));


    def _stop_recording(self,*args, **kwargs):
        log.debug("Adding run for step {}".format(self._algorithm.current_step));
        self._record_step()
        summary = self._create_summary();
        self._logger._writer.add_summary(summary,self._algorithm.current_step);
        self._logger._writer.flush();
        self._record_step_callback.disconnect();
        self.recording = False;


    def _create_summary(self):
        summary = tf.Summary();
        summary_metadata = tf.SummaryMetadata(
            display_name=None,
            summary_description=None,
            plugin_data=tf.SummaryMetadata.PluginData(
                plugin_name=PLUGIN_NAME,
                content=json.dumps(self._metadata).encode()
            )
        )
        for variable, tag in self._watching.items():
            array = np.array(self._data[variable]);

            proto = tf.make_tensor_proto(array);
            summary.value.add(
                tag = self._logger._prefix+tag+self._suffix,
                metadata = summary_metadata,
                tensor = proto
            );



        return summary;
