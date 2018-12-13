import numpy as np
from collections import namedtuple

config = dict(
    threshold = 1e-20,
    evaluator=dict(
        seed = 1001,
        N_episodes = 1000,
        summarizer = lambda pop: dict(mean_fitness=np.mean(pop.fitness), mean = pop.mean, covariance = pop.covariance),
        name = "big_evaluation"
    ),

)
