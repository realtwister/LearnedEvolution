import numpy as np
# GIT HASH: master ba9a07ef4aac41940ecd4b8f809233a201848218
config = dict(
        dimension = 2,
        population_size = 100,
        algorithm = dict(
            mean_function = dict(
                type = "RLMean",
                reward_function = dict(
                        type = "DifferentialReward"
                )
                ),
            covariance_function = dict(
                type = "AMaLGaMCovariance"
                ),
            convergence_criterion = dict(
                type = "CovarianceConvergence"
                ),
            ),
        problem_suite = dict(
            clss = [
                    ['RotateProblem', 'TranslateProblem','Ellipsoid']
                ],
            ),
        benchmark = dict(
            seed = 1000,
            should_save = lambda i: set(str(i)[1:])==set('0'),
            N_episodes = 100000,
            ),
        threshold = 1e-20,
        evaluator=dict(
            seed = 1001,
            N_episodes = 1000,
            summarizer = lambda pop: dict(mean_fitness=np.mean(pop.fitness), mean = pop.mean, covariance = pop.covariance),
            name = "big_evaluation"
        ),
        )
