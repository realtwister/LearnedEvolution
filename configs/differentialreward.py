import numpy as np
config = dict(
    dimension = 2,
    population_size = 100,
    algorithm = dict(
        mean_function =dict(
            type = "RLMean"
        ),
        covariance_function = dict(
            type = "AMaLGaMCovariance"
        ),
        convergence_criterion = dict(
            type = "TimeConvergence",
            max_iter = 200,
        )
    ),
    problem_suite = dict(
        clss=[
            ["RotateProblem", "TranslateProblem", "Rosenbrock"]
        ]
    ),
    benchmark = dict(
        should_save = lambda i: set(str(i)[1:]) == set('0'),
        seed = 1000
    ),

)
