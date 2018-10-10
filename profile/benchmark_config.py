from learnedevolution.algorithm import Algorithm;

from learnedevolution.convergence.time_convergence import TimeConvergence;

from learnedevolution.targets.mean import BaselinePPOMean;
from learnedevolution.rewards.differential_reward import DifferentialReward;

from learnedevolution.targets.covariance import AdaptiveCovarianceNew;

from learnedevolution.tensorboard.algorithm_logger import AlgorithmLogger;
from learnedevolution.tensorboard.generator_logger import GeneratorLogger;

from learnedevolution.problems import *;

class BenchmarkConfig:
    parameters = dict(
        population_size = 30,
        dimension = 1,
        N_train = 200,
        N_test = 100,
        N_epoch = 1,
        seed_test = 1000,
        seed_train = 1001,

    );

    def initiate_algorithm(self):
        self._convergence = convergence = TimeConvergence(100);

        self._ppo_mean = ppo_mean = BaselinePPOMean(self.parameters['dimension'],
            population_size = self.parameters['population_size'],
            rewards = {DifferentialReward():1},
            convergence_criteria=[convergence]);

        mean_targets = {
            ppo_mean:1,
        }

        covariance_targets = {
            AdaptiveCovarianceNew():1
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
            TranslateProblem(Sphere),
            #RotateProblem(TranslateProblem)
        ];
        self._train_suite = train_suite = ProblemSuite(problems, dimension= self.parameters['dimension']);
        self._test_suite = test_suite = train_suite.copy()
        return train_suite, test_suite;

    def initiate_logging(self, logdir):
        algo = AlgorithmLogger(self._algorithm, logdir);

        self._ppo_mean._agent.set_logdir(logdir+"/agent");
        mean = algo.create_child(self._ppo_mean);
        mean.recorder.watch('_current_reward','reward');
        mean.recorder.watch('_action','action');
        mean.recorder.watch('_current_state', 'state');

        gen_log = GeneratorLogger(self._test_suite, logdir);
        return algo, gen_log
