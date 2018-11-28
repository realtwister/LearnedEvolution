from learnedevolution.algorithm import Algorithm;

from learnedevolution.convergence.time_convergence import TimeConvergence;
from learnedevolution.convergence.convergence_criterion import ConvergenceCriterion;
from learnedevolution.convergence.amalgam_convergence import AMaLGaMConvergence;
from learnedevolution.convergence.covariance_convergence import CovarianceConvergence;
from learnedevolution.convergence.combined_reward import CombinedDifferentialReward;

from learnedevolution.targets.mean import BaselinePPOMean, MaximumLikelihoodMean;
from learnedevolution.rewards.differential_reward import DifferentialReward;
from learnedevolution.rewards.trace_differential_reward import TraceDifferentialReward;
from learnedevolution.rewards.divergence_penalty import DivergencePenalty;
from learnedevolution.rewards.normalized_fitness_reward import NormalizedFitnessReward, DecayingMinimum, WindowMinimum, InitialMinimum, DelayedMaximum;
from learnedevolution.rewards.fitness_reward import FitnessReward;
from learnedevolution.rewards.lagging_differential import LaggingDifferentialReward;

from learnedevolution.targets.covariance import AMaLGaMCovariance,DiagonalCovariance, AdaptiveCovariance, AdaptiveCovarianceSelect, AdaptiveCovarianceNew, AdaptiveCovarianceEig;

from learnedevolution.tensorboard.algorithm_logger import AlgorithmLogger;
from learnedevolution.tensorboard.generator_logger import GeneratorLogger;

from learnedevolution.problems import *;

dimension = 10;

class BenchmarkConfig:
    parameters = dict(
        population_size = 100,
        dimension = 2,
        N_train = 1000,
        N_test = 100,
        N_epoch = 25,
        seed_test = 1000,
        seed_train = 1001,

    );

    def initiate_algorithm(self,logdir="/tmp/thesis/tmp"):
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
                type = "CovarianceConvergence",
                threshold = 1e-20
            )
        )

        from learnedevolution.algorithm import Algorithm;
        self._algorithm = algorithm = Algorithm.from_config(config);

        return algorithm;

    def initiate_problem_generator(self):
        problems = [
            RotateProblem(TranslateProblem(Rosenbrock)),
            #RotateProblem(TranslateProblem)
        ];
        self._train_suite = train_suite = ProblemSuite(problems, dimension= self.parameters['dimension']);
        self._test_suite = test_suite = train_suite.copy()
        return train_suite, test_suite;

    def initiate_logging(self, logdir):
        algo = AlgorithmLogger(self._algorithm, logdir);

        # self._ppo_mean._agent.set_logdir(logdir+"/agent");
        # mean = algo.create_child(self._ppo_mean);
        # mean.recorder.watch('_current_reward','reward');

        gen_log = GeneratorLogger(self._test_suite, logdir);
        return algo, gen_log

if __name__=="__main__":
    config = BenchmarkConfig();
    print(config.initiate_algorithm());
    print(config.initiate_problem_generator());
    print(config.initiate_logging('/tmp/thesis/automated_benchmark/0'));
