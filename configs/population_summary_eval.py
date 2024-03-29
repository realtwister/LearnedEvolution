import numpy as np
from collections import namedtuple

config = dict(
    algorithm = dict(
        convergence_criterion = dict(
                  type = "CovarianceConvergence"
                  ),
    ),
    threshold = 1e-20,
    evaluator=dict(
        seed = 1001,
        N_episodes = 1000,
        summarizer = lambda pop, algo, problem: dict(mean_fitness=np.mean(pop.fitness), max_fitness=np.max(pop.fitness), reward = list(algo._mean_targets.keys())[0].reward, evaluations=problem.evaluations),
        name = "big_evaluation_2"
    ),

)
