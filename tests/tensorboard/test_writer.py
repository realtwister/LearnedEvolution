import tensorflow as tf;

import learnedevolution as lev;
from learnedevolution.tensorboard.writer import Writer;

def proto_test(proto, tag):
    assert isinstance(proto, tf.summary.Summary.Value);
    assert proto.tag == tag;

def test_caching():
    w1 = Writer('/tmp/thesis/writer_test');
    w2 = Writer('/tmp/thesis/writer_test');
    assert w1 is w2;

def test_scalar_proto():
    scalar = Writer.scalar_proto('length',10);
    proto_test(scalar, 'length')
    assert scalar.simple_value == 10;

def test_histogram_proto():
    dat = [10,20]
    histogram = Writer.histogram_proto('length',dat);
    proto_test(histogram, 'length');
    assert isinstance(histogram.histo, tf.HistogramProto);
    hist = histogram.histo;
    assert hist.max == max(dat);
    assert hist.min == min(dat);
    assert hist.num == len(dat);
    assert hist.sum == sum(dat);
