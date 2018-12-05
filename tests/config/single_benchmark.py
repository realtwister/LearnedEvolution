import numpy as np
config = dict(
    dimension = 2,
    population_size = 100,
    algorithm = dict(
        mean_function =dict(
            type = "RLMean",
            reward_function = dict(
                type = "DifferentialReward"
            )
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
        logdir = "/tmp/thesis/single_benchmarks/differentialReward_TimeConv",
        seed = 1000
    ),
    evaluator = dict(
        restoredir = "/tmp/thesis/test/savedir2/10000",
        logdir = "/tmp/thesis/test/savedir2/evaluations",
        seed = 1001,
        N_episodes = 100
    )

)


from learnedevolution.benchmark import Benchmark
benchmark = Benchmark.from_config(config, 'benchmark')
benchmark.run()
