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
        should_save = lambda i: i/(10**np.floor(np.log10(i))) in [1,5],
        seed = 1000
    ),

)
