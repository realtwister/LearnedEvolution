import os
import pickle
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_step(evaldir, step):
    assert os.path.exists(evaldir)
    path = os.path.join(evaldir, str(step)+".pkl")
    assert os.path.exists(path)
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_pickle(path):
    assert os.path.exists(path)
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def get_from_data(data, statistic):
    assert statistic in data[0][0], "Data should contain \"{}\"".format(statistic)
    result = [];
    for history in data:
        result.append([])
        for population in history:
            result[-1].append(population[statistic])
    return result

def calculate_first_hit(series, targets, descending = True):
    targets = sorted(set(targets), reverse=descending)
    hits = []
    i_target = 0
    for i, y in enumerate(series):
        while i_target < len(targets):
            if targets[i_target] > y:
                hits.append(i)
                i_target += 1
            else:
                break
        if i_target >= len(targets):
            break
    return hits, targets[:len(hits)]


def plot_step_first_hit(histories, num_bins = 10, min_val=None, max_val = None, quantile = 0.20, epsilon = 1e-30):
    histories = [-np.array(history) for history in histories]
    max_vals, min_vals = reduce(lambda res,x:(res[0]+[np.max(x)], res[1]+[np.min(x)]),
                                histories,
                                ([],[]))
    if max_val is None:
        max_val = np.quantile(max_vals,quantile)
    if min_val is None:
        min_val = np.quantile(min_vals, 1-quantile)
        min_val = max(epsilon, min_val)
    targets = np.logspace(np.log10(min_val), np.log10(max_val), num_bins)
    first_hits, targets = zip(*[calculate_first_hit(history,targets) for history in histories]);

    flatten = lambda l:reduce(lambda res,x: res+x,l,[])
    data = dict(
        target = flatten(targets),
        first_hit = flatten(first_hits),
    )
    pd.DataFrame(data=data)
    ax = sns.lineplot(x='target', y='first_hit', data=data, ci = "sd")
    return ax

def plot_steps_first_hit(evaldir, steps, **kwargs):
    for step in steps:
        data = load_step(evaldir, step)
        mean_fitness = get_from_data(data,'mean_fitness')
        plot_step_first_hit(mean_fitness, **kwargs)
    plt.legend(steps)
    ax = plt.gca()
    ax.set_xscale("log", nonposx='clip')
    xlim = plt.xlim
    plt.xlim(plt.xlim()[::-1])
