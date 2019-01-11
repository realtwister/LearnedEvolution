import numpy as np
from collections import namedtuple

config = dict(
    should_learn = False,
    algorithm = dict(
        convergence_criterion = dict(
                  type = "PaperConvergence",
                  max_iter = 1000
                  ),
    ),
    evaluator=dict(
        seed = 1001,
        N_episodes = 1000,
        summarizer = lambda pop, algo, problem: dict(mean_fitness=np.mean(pop.fitness), max_fitness=np.max(pop.fitness), reward = list(algo._mean_targets.keys())[0].reward, evaluations=problem.evaluations),
        name = "evaluate"
    ),

)
