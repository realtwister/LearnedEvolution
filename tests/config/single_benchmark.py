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
            type = "CovarianceConvergence",
            threshold = 1e-20
        )
    ),
    problem_suite = dict(
        clss=[
            ["RotateProblem", "TranslateProblem", "Rosenbrock"]
        ]
    ),
    benchmark = dict(
        should_save = lambda i: i/(10**np.floor(np.log10(i))) in [1,5],
        savedir = "/tmp/thesis/test/savedir2",
        restoredir = "/tmp/thesis/test/savedir/1000",
        seed = 1000
    )

)


from learnedevolution import Benchmark
benchmark = Benchmark.from_config(config, 'benchmark')
benchmark.run();

# algorithm =
# suite = ProblemSuite.from_config(config, "problem_suite")
# import time
# start = time.clock();
# N=1000
# for i,problem in enumerate(suite.iter(N)):
#     if config['benchmark']['should_save'](i):
#         print(i);
#     algorithm.maximize(problem.fitness)
#
# print(N/(time.clock()-start))
