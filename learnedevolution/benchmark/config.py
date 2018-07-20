from learnedevolution.algorithm import Algorithm;

from learnedevolution.convergence.time_convergence import TimeConvergence;

from learnedevolution.targets.mean import BaselinePPOMean;
from learnedevolution.rewards.differential_reward import DifferentialReward;

from learnedevolution.targets.covariance import ConstantCovariance;

from learnedevolution.tensorboard.algorithm_logger import AlgorithmLogger;
from learnedevolution.tensorboard.generator_logger import GeneratorLogger;

from learnedevolution.problems import *;

class BenchmarkConfig:
    parameters = dict(
        population_size = 100,
        dimension = 2,
        N_train = 200,
        N_test = 10,
        N_epoch = 1,
        seed_test = 1000,
        seed_train = 1001,

    );

    def initiate_algorithm(self):
        # Convergence criterion
        self._convergence = convergence = TimeConvergence(100);

        # initiate mean target
        mean_rewards = {
            DifferentialReward():1
        };

        self._ppo_mean = ppo_mean = BaselinePPOMean(self.parameters['dimension'],
            population_size = 100,
            rewards = mean_rewards,
            convergence_criteria=[convergence]);

        mean_targets = {
            ppo_mean:1
        }

        # initiate covariance target
        covariance_targets = {
            ConstantCovariance():1
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
