config = dict(
    population_size = 100,
    dimension = 2,
    benchmark = dict(
        type = "Benchmark",
        N_train = 200,
        N_test = 100,
        N_epoch = 1,
        seed_test = 1000,
        seed_train = 1001,
    ),
    algorithm = dict(
        population_size = 10,
        mean_function = dict(
            type = "BaselinePPOMean",
            reward = dict(
                type = "DifferentialReward"
            ),
        ),
        covariance_function = dict(
            type = "AMaLGaMCovariance",
        ),
        convergence = dict(
            type = "ConvergenceCriterion",
            gamma = 0.02,
            max_streak = 10,
        )
    ),
    problem_suite = dict(
        problems = [
            ["Rotate", "Translate","Sphere"]
        ],
    ),
    logger=dict(
        dir = "/tmp/thesis/test/config/1",
        overwrite = True,
    )
)
