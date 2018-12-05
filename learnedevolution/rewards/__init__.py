def rewards_classes():
    from .differential_reward import DifferentialReward
    from .normalized_fitness_reward import NormalizedFitnessReward
    return locals()
