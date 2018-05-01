import tensorflow as tf;
import logging;

from learnedevolution.tensorboard.algorithm_logger import AlgorithmLogger;
from learnedevolution.tensorboard.generator_logger import GeneratorLogger;
import learnedevolution as lev;

logging.basicConfig(level="DEBUG");

dimension = 2;
logdir = "/tmp/thesis/logger/10";

mean_targets = {lev.targets.mean.MaximumLikelihoodMean():1};
covariance_targets ={lev.targets.covariance.ConstantCovariance():1};
algo = lev.Algorithm(dimension, mean_targets, covariance_targets);

log1 = AlgorithmLogger(algo, logdir = '/tmp/thesis/logger');
log = AlgorithmLogger(algo, logdir = logdir);
assert log1.writer == log.writer;


problem_generator = lev.problems.Sphere.generator(dimension=dimension);
gen_log = GeneratorLogger(problem_generator, logdir);

#log.watch_scalar('_mean_fitness', 'before_reset', once_in=1, tag="mean_fitness");
#log.watch_histogram('_evaluated_fitness', 'before_reset', once_in=1, tag="evaluated_fitness");



for problem in problem_generator.iter(1):
    log.recorder.record();
    gen_log.add_current('problem', algo.current_step+1);
    mean, covariance = algo.maximize(problem.fitness, 100);
