config = dict(
        dimension = 2,
        population_size = 100,
        algorithm = dict(
            mean_function = dict(
                type = "RLMean",
                reward_function = <<VARIABLE:reward_function>>,
                ),
            covariance_function = dict(
                type = "AMaLGaMCovariance",
                ),
            convergence_criterion = dict(
                type = "CovarianceConvergence",
                ),
            ),
        problem_suite = dict(
            clss = [
                ['RotateProblem','TranslateProblem','Rosenbrock']
                ],
            ),
        benchmark = dict(
            N_episodes = 1000,
            should_save = lambda i: set(str(i)[1:]) == set('0'),
            seed = 1000,
            ),
        )
