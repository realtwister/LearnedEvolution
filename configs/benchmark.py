from learnedevolution.algorithm import Algorithm;

from learnedevolution.convergence.time_convergence import TimeConvergence;

from learnedevolution.targets.mean import BaselinePPOMean;
from learnedevolution.rewards.delta_reward import DeltaReward;
from learnedevolution.rewards.fitness_reward import FitnessReward;

from learnedevolution.targets.covariance import SphericalCovariance;

from learnedevolution.tensorboard.algorithm_logger import AlgorithmLogger;
from learnedevolution.tensorboard.generator_logger import GeneratorLogger;

from learnedevolution.problems import *;

class BenchmarkConfig:
    parameters = dict(
        population_size = 100,
        dimension = 2,
        N_train = 200,
        N_test = 10,
        N_epoch = 100,
        seed_test = 1000,
        seed_train = 1001,

    );

    def initiate_algorithm(self):
        convergence = TimeConvergence(40);

        self._ppo_mean = ppo_mean = BaselinePPOMean(self.parameters['dimension'],
            population_size = self.parameters['population_size'],
            rewards = {DeltaReward():1},
            convergence_criteria=[convergence]);

        mean_targets = {
            ppo_mean:1,
        }

        # initiate covariance target
        covariance_targets = {
            SphericalCovariance(alpha=0.1):1
        }

        self._algorithm = algorithm = Algorithm(
            self.parameters['dimension'],
            mean_targets,
            covariance_targets,
            [convergence],
            population_size = self.parameters['population_size'],

        );

        return algorithm;

    def initiate_problem_generator(self):
        problems = [
            RotateProblem(TranslateProblem(Sphere)),
        ];
        self._train_suite = train_suite = ProblemSuite(problems, dimension= self.parameters['dimension']);
        self._test_suite = test_suite = train_suite.copy()
        return train_suite, test_suite;

    def initiate_logging(self, logdir):
        algo = AlgorithmLogger(self._algorithm, logdir);

        self._ppo_mean._agent.set_logdir(logdir+"/agent");
        mean = algo.create_child(self._ppo_mean);
        mean.recorder.watch('_current_reward','reward');

        gen_log = GeneratorLogger(self._test_suite, logdir);
        return algo, gen_log

if __name__ == "__main__":
    from learnedevolution.benchmark.benchmark import Benchmark;
    b = Benchmark("./benchmark.py", "/tmp/thesis/single_benchmarks/benchmark_7", progress=True);
    b.run();
