import numpy as np
from collections import namedtuple

config = dict(
    evaluator=dict(
        seed = 1001,
        N_episodes = 100,
        summarizer = lambda pop: (np.mean(pop.fitness), pop.mean, pop.covariance),
    )

)
