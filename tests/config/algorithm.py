config = dict(
    dimension= 2,
    population_size = 100,
    mean_function =dict(
        type = "RLMean"
    ),
    covariance_function = dict(
        type = "AMaLGaMCovariance"
    ),
    convergence_criterion = dict(
        type = "AMaLGaMConvergence"
    )
)

from learnedevolution.algorithm import Algorithm;
algo = Algorithm.from_config(config);

from learnedevolution.problems import *


problems = [
    Sphere
];
suite = ProblemSuite(problems, dimension= config['dimension']);
for problem in suite.iter(100):
    mean, covariance = algo.maximize(problem.fitness)
    print(mean);
