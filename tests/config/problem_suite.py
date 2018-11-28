config = dict(
    dimension = 2,
    problem_suite = dict(
        clss = [
            ["RotateProblem","TranslateProblem","Sphere"]
        ]
    )

);


from learnedevolution.problems.suite import ProblemSuite;

suite = ProblemSuite.from_config(config, "problem_suite")

print(suite.generate());
