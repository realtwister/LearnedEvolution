config = dict(
)

from learnedevolution.targets.mean.RL_mean import RLMean;
mean = RLMean.from_config(config);
print(mean._observation_space.population_size)
