from .time_convergence import TimeConvergence;
from .covariance_convergence import CovarianceConvergence;
from .amalgam_convergence import AMaLGaMConvergence;
def convergence_classes():
    from .time_convergence import TimeConvergence;
    from .covariance_convergence import CovarianceConvergence;
    from .amalgam_convergence import AMaLGaMConvergence;
    from .paper_convergence import PaperConvergence
    return locals();
