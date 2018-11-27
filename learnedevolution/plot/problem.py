import numpy as np;
import matplotlib.pyplot as plt;

def optimum(problem, fmt = 'ro', coords=[0,1], **kwargs):
    std_kwargs = dict(
    )
    std_kwargs.update(kwargs);
    plt.plot(*problem.optimum[coords], fmt, **std_kwargs)
