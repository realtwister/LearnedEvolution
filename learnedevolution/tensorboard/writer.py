import tensorflow as tf;
import numpy as np;

import logging;
log = logging.getLogger(__name__)

class Writer(object):
    _writers = {};
    def __new__(cls,  logdir):
        if logdir in Writer._writers:
            log.debug('Reloading writer');
            self = Writer._writers[logdir];
        else:
            log.debug('Creating writer for "{}"'.format(logdir));
            self = super().__new__(cls);
            self.writer = tf.summary.FileWriter(logdir);
            Writer._writers[logdir] = self;

        return self;

    @staticmethod
    def scalar_proto(tag, value):
        return tf.summary.Summary.Value(
            tag=tag,
            simple_value=value);

    @staticmethod
    def histogram_proto(tag, values, step = None, bins = 1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        return tf.Summary.Value(tag=tag, histo=hist);

    @staticmethod
    def summary_proto(protos):
        assert isinstance(protos, list), 'Protos should be a list';
        return tf.Summary(value=protos);

    def add_scalar(self, tag, value, step):
        scalar = self.scalar_proto(tag, value);
        summary = self.summary_proto([scalar]);
        self.add_summary(summary, step);

    def add_histogram(self, tag, value, step):
        histogram = self.histogram_proto(tag, value);
        summary = self.summary_proto([histogram]);
        self.add_summary(summary, step);

    def add_summary(self, summary, step):
        log.debug('Adding summary to {}'.format(self._writer.get_logdir()));
        self._writer.add_summary(summary, step);

    def flush(self):
        log.debug('Flushing {}'.format(self._writer.get_logdir()));
        self._writer.flush();

    @property
    def writer(self):
        return self._writer;

    @writer.setter
    def writer(self, writer):
        log.info('Logging to {}'.format(writer.get_logdir()));
        self._writer = writer;
