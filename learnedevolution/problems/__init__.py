from .sphere import Sphere
from .rosenbrock import Rosenbrock
from .suite import ProblemSuite
from .translated import TranslateProblem
from .rotated import RotateProblem
from .fitness_scaled import FitnessScaleProblem
from .discretized import DiscretizeProblem

def problem_classes():
    from .sphere import Sphere;
    from .rosenbrock import Rosenbrock;
    from .suite import ProblemSuite;
    from .translated import TranslateProblem;
    from .rotated import RotateProblem;
    from .fitness_scaled import FitnessScaleProblem;
    from .discretized import DiscretizeProblem;
    return locals()
