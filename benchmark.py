import tensorflow as tf;
import logging;
import os;
from learnedevolution.problems import *;

from learnedevolution.tensorboard.algorithm_logger import AlgorithmLogger;
from learnedevolution.tensorboard.generator_logger import GeneratorLogger;
import learnedevolution as lev;

from learnedevolution.rewards.differential_reward import DifferentialReward;
from learnedevolution.rewards.divergence_penalty import DivergencePenalty;
from learnedevolution.rewards.normalized_fitness_reward import NormalizedFitnessReward, DecayingMinimum, WindowMinimum, InitialMinimum, LaggingMaximum;

from learnedevolution.convergence.convergence_criterion import ConvergenceCriterion;
from learnedevolution.convergence.time_convergence import TimeConvergence;

# Algo setting
dimension = 2;
population_size = 100;
seed = 1000;

# Logdir
base_path = '/tmp/thesis/logger/';
if True:
    i = 0;
    while os.path.exists(base_path+str(i)):
        i += 1;
    logdir = base_path+str(i)
else:
    logdir = base_path+"new_convergence"

if os.path.exists(logdir):
    raise Exception("Pad bestaat al");

# Convergence criteria
convergence_criterion = ConvergenceCriterion(reward_per_step=0.5, gamma=0.02);
time_convergence = TimeConvergence();

convergence_criteria = [time_convergence];

minima = [
    DecayingMinimum(0.95),
    WindowMinimum(20),
    InitialMinimum()
];
maxima = [
    LaggingMaximum()
]
normalized_fitness = NormalizedFitnessReward(minima,maxima)

rewards = {
    normalized_fitness:1,
    DifferentialReward():0,
    DivergencePenalty():0};

ppo_mean = lev.targets.mean.BaselinePPOMean(dimension, population_size, rewards, convergence_criteria, logdir = logdir+"/agent");

mean_targets = {
    ppo_mean:1,
    #lev.targets.mean.TensorforcePPOMean(dimension, population_size):1,
    };
diag_covariance = lev.targets.covariance.DiagonalCovariance(0.2, [1,2])
covariance_targets ={
    #lev.targets.covariance.ConstantCovariance():1,
    #lev.targets.covariance.AdhocCovariance():1,
    diag_covariance:1,
    };

algo = lev.Algorithm(dimension, mean_targets, covariance_targets, convergence_criteria, population_size=population_size);


log = AlgorithmLogger(algo, logdir = logdir);
covariance_log = log.create_child(diag_covariance);
covariance_log.recorder.watch('variance','variance')

mean_log = log.create_child(ppo_mean);
mean_log.recorder.watch('_current_reward','reward');

reward_log = mean_log.create_child(normalized_fitness);
reward_log.recorder.watch('_minimum', 'minimum')
reward_log.recorder.watch('_maximum', 'maximum');

convergence_log = log.create_child(algo._convergence_criteria[0]);
convergence_log.watch_scalar('epsilon', 'after_reset', once_in=10, tag="epsilon");

problems = [
    Sphere
];

# Problem suite

suite = ProblemSuite(problems, dimension= dimension);

gen_log = GeneratorLogger(suite, logdir);

# Settings
N_test = 1;
N_train = 100;
epochs = 100;

#Seeding
algo.seed(seed);
suite.seed(seed);


# Algorithm

test_suite_state = suite.get_state();
test_algo_state = algo._random_state.get_state();
train_suite_state = None;
train_algo_state = None;


for i in range(epochs):
    # Test algo
    suite.set_state(test_suite_state);
    algo._random_state.set_state(test_algo_state);
    print("Starting test....")
    for problem in suite.iter(N_test):
        algo._steps -=1;
        temp_suite_state = suite.get_state();
        temp_algo_state = algo._random_state.get_state();
        log.record(suffix="deterministic");
        gen_log.add_current('problem', algo.current_step+1);
        mean, covariance = algo.maximize(problem.fitness, 100, True);
        suite.set_state(temp_suite_state);
        algo._random_state.set_state(temp_algo_state);
        algo._steps -=1;
        log.record();
        mean, covariance = algo.maximize(problem.fitness, 100);
    print("Test ended");

    # Train algo
    if train_suite_state is not None:
        suite.set_state(train_suite_state);
        algo._random_state.set_state(train_algo_state);

    for problem in suite.iter(N_train):
        mean, covariance = algo.maximize(problem.fitness, 100);
    train_suite_state = suite.get_state();
    train_algo_state = algo._random_state.get_state();
