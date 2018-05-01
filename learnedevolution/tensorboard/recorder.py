import tensorflow as tf;
import numpy as np;
from tensorflow.core.framework import tensor_pb2
from learnedevolution.utils.signals import TimedCallback;

import logging;
log = logging.getLogger(__name__)

PLUGIN_NAME = 'structevol'

class Recorder(object):
    def __init__(self, logger):
        self._logger = logger;
        self._instance = logger._instance;

        self._stop_recording_callback = TimedCallback('after_terminate', fns = self._stop_recording, one_shot = True, sender = self._instance);
        self._record_step_callback = TimedCallback('after_step', fns = self._record_step, sender = self._instance);
        self.recording = False;

        self._watching = {};

    def watch(self, variable, tag=None):
        if tag is None:
            tag = variable;
        self._watching[variable] = tag;

    def record(self):
        assert not self.recording, 'Trying to start recording while already recording';
        self.recording = True;
        self._stop_recording_callback.connect();
        self._record_step_callback.connect();
        self._data = {key:[] for key in self._watching};


    def _record_step(self, *args, **kwargs):
        i = self._instance;
        for var in self._watching:
            self._data[var].append(getattr(i, var));


    def _stop_recording(self,*args, **kwargs):
        log.debug("Adding run for step {}".format(self._instance.current_step));
        summary = self._create_summary();
        self._logger._writer.add_summary(summary,self._instance.current_step);
        self._logger._writer.flush();
        self._record_step_callback.disconnect();
        self.recording = False;


    def _create_summary(self):
        summary = tf.Summary();
        summary_metadata = tf.SummaryMetadata(
            display_name=None,
            summary_description=None,
            plugin_data=tf.SummaryMetadata.PluginData(
                plugin_name=PLUGIN_NAME
            )
        )
        for variable, tag in self._watching.items():
            array = np.array(self._data[variable]);

            proto = tf.make_tensor_proto(array);
            summary.value.add(
                tag = tag,
                metadata = summary_metadata,
                tensor = proto
            );



        return summary;
