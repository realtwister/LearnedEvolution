import numpy as np;
import matplotlib.pyplot as plt;

def path(d, **kwargs):
    assert isinstance(d, np.ndarray);
    assert len(d.shape)==2;
    assert d.shape[1] ==2;
    d = d.T;
    std_kwargs = dict(
        angles = 'xy',
        scale_units='xy',
        scale=1
    );
    std_kwargs.update(kwargs);
    plt.quiver(*d[:,:-1],
               *(d[:,1:]-d[:,:-1]),
              **std_kwargs)

def __attr_path(attr):
    def fn(objs, coords=[0,1], **kwargs):
        d = np.array([getattr(obj, attr)[coords] for obj in objs]);
        path(d)
    return fn;

mean_path = __attr_path('mean');
