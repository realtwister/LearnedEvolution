from .maximum_likelihood_mean import MaximumLikelihoodMean;
from .RL_mean import RLMean
def mean_classes():
    from .maximum_likelihood_mean import MaximumLikelihoodMean;
    from .RL_mean import RLMean
    from .cmaes_mean import CMAESMean
    return locals();
