config = dict(
    mean_function =dict(
        type = "MaximumLikelihoodMean"
    ),
    covariance_function = dict(
        type = "AMaLGaMCovariance"
    ),
    convergence_criterion = dict(
        type = "AMaLGaMConvergence"
    )
)

from learnedevolution.algorithm import Algorithm;
alg = Algorithm.from_config(config);
