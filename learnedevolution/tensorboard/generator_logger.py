from .logger import Logger;
from ..utils.random import RandomGenerator;
from .recorder import Recorder;
from tensorflow.core.framework import tensor_pb2
import tensorflow as tf;

PLUGIN_NAME = 'structevol'

class GeneratorLogger(Logger):
    def __init__(self, generator, logdir):
        assert isinstance(generator, RandomGenerator);
        Logger.__init__(self, generator, logdir);

    def add_current(self, tag, step):
        proto = self._instance._current.proto;
        tensor = tensor_pb2.TensorProto();
        tensor.tensor_content = proto.SerializeToString();

        summary = tf.Summary();
        summary_metadata = tf.SummaryMetadata(
            display_name=None,
            summary_description=None,
            plugin_data=tf.SummaryMetadata.PluginData(
                plugin_name=PLUGIN_NAME
            )
        )

        summary.value.add(
            tag = tag,
            metadata = summary_metadata,
            tensor = tensor
        )

        self._writer.add_summary(summary, step);
        self._writer.flush();
