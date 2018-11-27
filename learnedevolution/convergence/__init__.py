def convergence_classes():
    from .time_convergence import TimeConvergence;
    from .covariance_convergence import CovarianceConvergence;
    from .amalgam_convergence import AMaLGaMConvergence;
    return locals();
