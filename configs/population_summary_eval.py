import numpy as np
from collections import namedtuple

config = dict(
    evaluator=dict(
        seed = 1001,
        N_episodes = 100,
        summarizer = lambda pop: dict(mean_fitness=np.mean(pop.fitness), mean = pop.mean, covariance = pop.covariance),
    )

)
