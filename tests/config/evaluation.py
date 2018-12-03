import numpy as np

from collections import namedtuple

Population = namedtuple("Population", ['mean_fitness','mean', 'covariance'])

config = dict(
    dimension = 2,
    population_size = 100,
    algorithm = dict(
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
    ),
    problem_suite = dict(
        clss=[
            ["RotateProblem", "TranslateProblem", "Rosenbrock"]
        ]
    ),
    evaluator = dict(
        algorithm = dict(
            mean_function =dict(
                type = "RLMean"
            ),
            covariance_function = dict(
                type = "AMaLGaMCovariance"
            ),
            convergence_criterion = dict(
                type = "TimeConvergence",
                max_iter = 200
            )
        ),
        restoredir = "/tmp/thesis/single_benchmarks/differentialReward_TimeConv/10000",
        logdir = "/tmp/thesis/single_benchmarks/differentialReward_TimeConv/evaluations/10000",
        seed = 1001,
        N_episodes = 100,
        summarizer = lambda pop: Population(np.mean(pop.fitness), pop.mean, pop.covariance),
    )
)


from learnedevolution import Benchmark, Evaluator
#benchmark = Benchmark.from_config(config, 'benchmark')
#benchmark.run()

evaluator = Evaluator.from_config(config, 'evaluator')
histories = evaluator.run()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
# 1 history fitness plot
plt.figure();
data = dict(
    fitness =[],
)
for i in range(len(histories)):
    history = histories[i]
    mean_fitness = -np.array([population.mean_fitness for population in history])
    data['fitness'] += [mean_fitness];
    plt.semilogy(mean_fitness, alpha = 0.1, color = 'k')

def plot_time_mean(fitness):
    max_T = np.max([len(f) for f in fitness]);
    transpose_fitness = [];
    for t in range(max_T):
        transpose_fitness.append([])
        for f in fitness:
            if t <len(f):
                transpose_fitness[t].append(f[t]);

    mean_fitness = [np.mean(f) for f in transpose_fitness];
    plt.semilogy(mean_fitness)

def precision_hits(fitness, precisions, ts = None):
    if ts is None:
        ts = list(np.arange(len(fitness)).astype(float))
    ps = sorted(precisions)[::-1]
    hits = []
    i = 0
    for t, f in zip(ts, fitness):
        while True:
            if i>=len(ps):
                break
            if f < ps[i]:
                hits.append(t)
                i += 1
            else:
                break
        if i>=len(ps):
            break
    return hits, ps[:len(hits)]


def plot_precision_mean(fitness, num_bins=100):
    ts = [i for f in fitness for i in range(len(f)) ]
    fs = [f for ff in fitness for f in ff]
    fs,ts = zip(*sorted(zip(fs,ts), key=lambda pair: -pair[0]))
    N = len(fs)
    bin_size = np.ceil(N/num_bins).astype(int)
    xs = [];
    ys = [];
    for i in range(num_bins):
        xs.append(np.mean(ts[i*bin_size: (i+1)*bin_size]))
        ys.append(np.mean(fs[i*bin_size: (i+1)*bin_size]))

    plt.semilogy(xs,ys)

def plot_precision_hits (fitness, num_bins = 100 ):
    max_precision = 0
    min_precision = float('inf')
    for f in fitness:
        max_precision = max(max_precision, np.min(f))
        min_precision = min(min_precision, np.max(f))

    precisions = np.logspace(np.log10(min_precision), np.log10(max_precision), num_bins)
    data = pd.DataFrame(columns=['time','precision'])
    for f in fitness:
        hits,ps = precision_hits(f, precisions)
        plt.semilogy(hits,ps)
        data = data.append([dict(time=t, precision=p) for t,p in zip(hits,ps)])
    plt.figure()
    ax = sns.scatterplot(x= 'precision', y='time', data=data, alpha= 0.1)
    ax.set( xscale="log")
    ax = sns.lineplot(x= 'precision', y='time', data=data, ax=ax, ci='sd')
    ax.set( xscale="log")









plt.figure();
plt.yscale('log')
plot_time_mean(data['fitness'])
plot_precision_hits(data['fitness'], num_bins=10)

plt.show();
