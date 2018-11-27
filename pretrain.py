import random;
import numpy as np;
import os;
from itertools import chain;

from learnedevolution.algorithm import Algorithm;
from learnedevolution.targets.mean import *;
from learnedevolution.targets.covariance import *;
from learnedevolution.convergence import CovarianceConvergence;
from learnedevolution.rewards import *;
from learnedevolution.problems import *;
from learnedevolution.utils.signals import TimedCallback;
config = dict(
    dimension = 5,
    population_size = 100,
    pretrainer = dict(
        batch_size = 128,
        episodes_per_epoch = 100,
        repetition_per_epoch = 1,
        epochs = 100,
        save_dir = "/tmp/thesis/rosenbrock/pretrainer/unordered",
        threads = 6,
    )
)

# Setup the algorithm
convergence = CovarianceConvergence();
source_target = MaximumLikelihoodMean();
destination = BaselinePPOMean(
    dimension = config['dimension'],
    population_size = config['population_size'],
    rewards = {DifferentialReward():1},
    convergence_criteria=[]
)

covariance_target = AMaLGaMCovariance();

algorithm = Algorithm(
    dimension = config['dimension'],
    mean_targets = {source_target:1},
    covariance_targets = {covariance_target:1},
    convergence_criteria = [convergence],
    population_size = config['population_size']
)

# Initialize problems
problemset = [
    RotateProblem(TranslateProblem(Rosenbrock))
];
problem_suite = ProblemSuite(problemset, dimension=config['dimension']);


def collect_episode(source, fitness):
    # initiate data buffers
    populations = [];

    # Setup data collection
    def add_population(*args,**kwargs):
        populations.append(source._population_obj);
    timed_callback = TimedCallback('after_generate_population',
    sender = source,
    fns = add_population
    );
    timed_callback.connect();

    # Run episode
    source.maximize(fitness);

    #Disconnect
    timed_callback.disconnect();
    return populations;

def process_episode(destination_state, history, log_std_const = -1):
    ## init state and action buffer
    states = [];
    actions = [];
    log_stds =[];

    # alias and reset state
    destination_state.reset();

    # Main loop
    for i, population in enumerate(history):
        if i > 0:
            action = destination_state.invert(population.mean);
            action_log_norm = np.log(np.linalg.norm(action));
            actions.append(action);
            log_stds.append([action_log_norm+log_std_const]*population.dimension)
        if i < len(history)-1:
            states.append(destination_state.encode(population));

    return (states, actions, log_stds);

def parse_episode(destination_state, history, log_std_const = 0):
    # init state and action buffer
    states = [];
    actions = [];
    log_stds =[];

    # alias and reset state
    destination_state.reset();

    # Main loop
    for i, population in enumerate(history):
        if i > 0:
            action = destination_state.invert(population.mean);
            action_log_norm = np.log(np.linalg.norm(action));
            actions.append(action);
            log_stds.append([action_log_norm+log_std_const]*population.dimension)
        if i < len(history)-1:
            observations.append(destination_state.encode(population));
        states.append(state.encode(population));
        actions.append(state.invert(population.mean));

    return (states, actions, log_stds);


import tensorflow as tf;
def setup_pretrainer(destination):
    with tf.variable_scope('pretrainer'):
        tf_observations = destination._agent._policy._observations;
        distribution = destination._agent._policy.pd;

        mean = distribution.mean;
        tf_true_mean = tf.placeholder(tf.float32, shape=(None, config['dimension']));

        log_std = distribution.logstd;
        tf_true_log_std = tf.placeholder(tf.float32, shape=(None, config['dimension']));

        # calculate loss
        quadratic_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf_true_mean-mean),axis=1)))**2;

        quadratic_loss += tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf_true_log_std-log_std),axis=1)))**2;

        optimizer = tf.train.AdamOptimizer();
        var_list = destination._agent._policy.get_trainable_variables();
        minimize_op = optimizer.minimize(quadratic_loss, var_list = [var_list]);
        print('got here');
    return dict(actions = tf_true_mean, observations=tf_observations, log_stds = tf_true_log_std), minimize_op, quadratic_loss;


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


from copy import deepcopy;
def worker(i, problem, source, destination):
    source = deepcopy(source);
    destination_state = deepcopy(destination._state);
    episode = collect_episode(source, problem.fitness);
    return process_episode(destination_state, episode);

from multiprocessing import Pool

#setup_value_trainer(destination);

placeholders, minimize_op, loss = setup_pretrainer(destination)

sess = destination._agent._session;
sess.run(tf.initializers.global_variables())
for i_epoch in range(config['pretrainer']['epochs']):
    def epoch_worker(args):
        (i,problem) = args;
        return worker(i,problem,algorithm,destination)
    print("epoch {}: collecting..".format(i_epoch), end="\r");
    with Pool(config['pretrainer']['threads']) as p:
        states,actions, log_stds = zip(*p.map(epoch_worker,enumerate(problem_suite.iter(config['pretrainer']['episodes_per_epoch']))));
    epoch_states = list(chain.from_iterable(states));
    epoch_actions = list(chain.from_iterable(actions));
    epoch_log_stds = list(chain.from_iterable(log_stds));
    print("epoch {}: training..".format(i_epoch), end="\r");
    idx = np.arange(len(epoch_states));
    n_trainings = 0;
    current_loss = sess.run(loss,{
        placeholders['observations']:np.array(epoch_states),
        placeholders['actions']:np.array(epoch_actions),
        placeholders['log_stds']:np.array(epoch_log_stds)
    });
    print("epoch {}: Loss before training: {}".format(i_epoch, current_loss));
    for _ in range(config['pretrainer']['repetition_per_epoch']):
        random.shuffle(idx);
        for batch_idx in batch(idx,config['pretrainer']['batch_size']):
            n_trainings += 1;
            sess.run(minimize_op, {
                placeholders['observations']:np.array(epoch_states)[batch_idx],
                placeholders['actions']:np.array(epoch_actions)[batch_idx],
                placeholders['log_stds']:np.array(epoch_log_stds)[batch_idx]
            });
    current_loss = sess.run(loss,{
        placeholders['observations']:np.array(epoch_states),
        placeholders['actions']:np.array(epoch_actions),
        placeholders['log_stds']:np.array(epoch_log_stds)
    });
    print("epoch {}: minimize was called {} times with total_loss {}".format(i_epoch, n_trainings, current_loss));


if not os.path.exists(config['pretrainer']['save_dir']):
    os.makedirs(config['pretrainer']['save_dir'])
save_path = os.path.join(config['pretrainer']['save_dir'],'mean_target.ckpt');
if os.path.isfile(save_path):
    os.remove(save_path);
print('saving..')
destination.save(save_path);
print("saved_to"+save_path);
