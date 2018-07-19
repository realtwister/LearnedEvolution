import tensorflow as tf;
import logging;
import os;
from tqdm import tqdm;
import numpy as np;
from learnedevolution.problems import *;

from learnedevolution.tensorboard.algorithm_logger import AlgorithmLogger;
from learnedevolution.tensorboard.generator_logger import GeneratorLogger;
import learnedevolution as lev;

from learnedevolution.rewards.differential_reward import DifferentialReward;
from learnedevolution.rewards.trace_differential_reward import TraceDifferentialReward;
from learnedevolution.rewards.divergence_penalty import DivergencePenalty;
from learnedevolution.rewards.normalized_fitness_reward import NormalizedFitnessReward, DecayingMinimum, WindowMinimum, InitialMinimum, LaggingMaximum;

from learnedevolution.convergence.convergence_criterion import ConvergenceCriterion;
from learnedevolution.convergence.time_convergence import TimeConvergence;

#logging.basicConfig(level="DEBUG");

# Algo setting
dimension = 2;
population_size = 100;
seed = 1000;

# Logdir
base_path = '/tmp/thesis/benchmark2/';
if False:
    i = 0;
    while os.path.exists(base_path+str(i)):
        i += 1;
    logdir = base_path+str(i)
else:
    logdir = base_path+"new_differential_4"

if os.path.exists(logdir):
    raise Exception("Pad bestaat al");

# Convergence criteria
convergence_criterion = ConvergenceCriterion(reward_per_step=0.5, gamma=0.02, max_streak=20);
time_convergence = TimeConvergence(100);

convergence_criteria = [convergence_criterion];
#convergence_criteria = [time_convergence];

minima = [
    DecayingMinimum(.92),
    WindowMinimum(10),
    InitialMinimum()
];
maxima = [
    LaggingMaximum()
]
normalized_fitness = NormalizedFitnessReward(minima,maxima)

rewards = {
    normalized_fitness:0,
    DifferentialReward():1,
    TraceDifferentialReward():0,
    DivergencePenalty():0};

ppo_mean = lev.targets.mean.BaselinePPOMean(dimension, population_size, rewards, convergence_criteria, logdir = logdir+"/agent");

mean_targets = {
    ppo_mean:1,
    #lev.targets.mean.TensorforcePPOMean(dimension, population_size):1,
    };
diag_covariance = lev.targets.covariance.DiagonalCovariance(0.5, [0.5,2])
covariance_targets ={
    #lev.targets.covariance.ConstantCovariance(1):1,
    #lev.targets.covariance.AdhocCovariance():1,
    diag_covariance:1,
    };

algo = lev.Algorithm(dimension, mean_targets, covariance_targets, convergence_criteria, population_size=population_size);


log = AlgorithmLogger(algo, logdir = logdir);
#covariance_log = log.create_child(diag_covariance);
#covariance_log.recorder.watch('variance','variance')

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
N_test = 10;
N_train = 200;
epochs = 100;
T = 100;

#Seeding
algo.seed(seed);
suite.seed(seed);

def calculate_fitness(recorder):
    fitness = np.array(recorder._data['_evaluated_fitness']);
    fitness= fitness.mean(axis=1);
    fitness = np.concatenate([fitness, np.ones(T-len(fitness))*fitness[-1]])
    return np.median(fitness);



# Algorithm

test_suite_state = suite.get_state();
test_algo_state = algo._random_state.get_state();
train_suite_state = None;
train_algo_state = None;

print("Logging to: {}".format(logdir))
for i in tqdm(range(epochs), desc="Epochs"):
    # Test algo
    suite.set_state(test_suite_state);
    algo._random_state.set_state(test_algo_state);
    ppo_mean.learning = False;
    update_entropy = 0;
    for i, problem in tqdm(enumerate(suite.iter(N_test)), desc="Testing", leave=False, total=N_test):
        algo._steps -=1;
        temp_suite_state = suite.get_state();
        temp_algo_state = algo._random_state.get_state();
        log.record(suffix="BENCHMARK.deterministic."+str(i));
        gen_log.add_current('problem', algo.current_step+1);
        mean, covariance = algo.maximize(problem.fitness, T, True);
        deterministic_fitness = calculate_fitness(log.recorder);
        suite.set_state(temp_suite_state);
        algo._random_state.set_state(temp_algo_state);
        algo._steps -=1;
        log.record(suffix="BENCHMARK.explorative."+str(i));
        mean, covariance = algo.maximize(problem.fitness, T);
        if deterministic_fitness-calculate_fitness(log.recorder)>0:
            update_entropy+=1;
        else:
            update_entropy-=1;
    if update_entropy == N_test:
        print("Multiplying")
        ppo_mean._agent._policy.add_std_offset(-1);
    print(update_entropy)
    ppo_mean.learning = True;

    # Train algo
    if train_suite_state is not None:
        suite.set_state(train_suite_state);
        algo._random_state.set_state(train_algo_state);

    for problem in tqdm(suite.iter(N_train), desc="Training", leave=False, total=N_train):
        mean, covariance = algo.maximize(problem.fitness, T);
    train_suite_state = suite.get_state();
    train_algo_state = algo._random_state.get_state();
