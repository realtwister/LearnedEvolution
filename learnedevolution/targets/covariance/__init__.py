def covariance_classes():
    from .cmaes_covariance import CMAESCovariance;
    from .amalgam_covariance import AMaLGaMCovariance;
    return locals();
