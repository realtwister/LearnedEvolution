from .maximum_likelihood_mean import MaximumLikelihoodMean;
from .baseline_ppo_mean import BaselinePPOMean;
from .RL_mean import RLMean
def mean_classes():
    from .maximum_likelihood_mean import MaximumLikelihoodMean;
    from .baseline_ppo_mean import BaselinePPOMean;
    from .RL_mean import RLMean
    return locals();
