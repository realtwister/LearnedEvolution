from learnedevolution.algorithm import Algorithm;

from learnedevolution.convergence.time_convergence import TimeConvergence;
from learnedevolution.convergence.convergence_criterion import ConvergenceCriterion;
from learnedevolution.convergence.combined_reward import CombinedDifferentialReward;

from learnedevolution.targets.mean import BaselinePPOMean;
from learnedevolution.rewards.differential_reward import DifferentialReward;
from learnedevolution.rewards.trace_differential_reward import TraceDifferentialReward;
from learnedevolution.rewards.divergence_penalty import DivergencePenalty;
from learnedevolution.rewards.normalized_fitness_reward import NormalizedFitnessReward, DecayingMinimum, WindowMinimum, InitialMinimum, DelayedMaximum;
from learnedevolution.rewards.fitness_reward import FitnessReward;
from learnedevolution.rewards.lagging_differential import LaggingDifferentialReward;

from learnedevolution.targets.covariance import ConstantCovariance;
from learnedevolution.targets.covariance import DiagonalCovariance, AdaptiveCovariance;

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
        # Convergence criterion
        if False:
            self._convergence = convergence = ConvergenceCriterion(gamma=0.02, max_streak=10);
        else:
            self._convergence = convergence = TimeConvergence();

        #self._convergence = convergence = CombinedDifferentialReward();

        # initiate mean target
        minima = [];
        if True:
            minima += [DecayingMinimum(.92)];
        if True:
            minima += [WindowMinimum(10)];
        if True:
            minima +=[InitialMinimum()];

        if len(minima) == 0:
            minima +=[InitialMinimum()];
            self.parameters['N_epoch'] = 0;

        maxima = [
            DelayedMaximum()
        ]
        normalized_fitness = NormalizedFitnessReward(minima,maxima)

        rewards = [
            DifferentialReward(),
            normalized_fitness,
        ]

        self._ppo_mean = ppo_mean = BaselinePPOMean(self.parameters['dimension'],
            population_size = self.parameters['population_size'],
            rewards = {rewards[0]:1},
            convergence_criteria=[convergence]);

        mean_targets = {
            ppo_mean:1
        }

        # initiate covariance target
        covariances = [
            DiagonalCovariance(0.2,[0.5,1.5]),
            ConstantCovariance()
        ]
        covariance_targets = {
            AdaptiveCovariance():1
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
        self._problem_suite = suite = ProblemSuite(problems, dimension= self.parameters['dimension']);
        return suite;

    def initiate_logging(self, logdir):
        algo = AlgorithmLogger(self._algorithm, logdir);

        self._ppo_mean._agent.set_logdir(logdir+"/agent");
        mean = algo.create_child(self._ppo_mean);
        mean.recorder.watch('_current_reward','reward');

        gen_log = GeneratorLogger(self._problem_suite, logdir);
        return algo, gen_log


if __name__=="__main__":
    config = BenchmarkConfig();
    print(config.initiate_algorithm());
    print(config.initiate_problem_generator());
    print(config.initiate_logging('/tmp/thesis/automated_benchmark/0'));
