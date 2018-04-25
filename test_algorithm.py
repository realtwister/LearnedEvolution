import learnedevolution as lev;
import numpy as np;

dimension = 2;

problem_generator = lev.problems.Sphere.generator(dimension=dimension);


mean_targets = {lev.targets.mean.MaximumLikelihoodMean():1};
covariance_targets ={lev.targets.covariance.ConstantCovariance():1};
algo = lev.Algorithm(dimension, mean_targets, covariance_targets);

for problem in problem_generator.iter(1):
    algo.reset();
    for algo._iteration in range(100):
        converged = algo._step(problem.fitness);
        mean, covariance = algo._mean, algo._covariance;
        if algo._iteration > 0:
            diff = min( problem.fitness(prev_mean[np.newaxis,:])-problem.fitness(mean[np.newaxis,:]),
            problem.fitness(problem.optimum[np.newaxis,:])-problem.fitness(mean[np.newaxis,:]));
            assert diff< 1, "Algorithm should be strictly decreasing but diff was {}".format(diff);
        prev_mean = mean;
        if converged:
            break;

for problem in problem_generator.iter(100):
    mean, covariance = algo.maximize(problem.fitness, 200);
    assert np.linalg.norm(mean-problem.optimum)<1;
