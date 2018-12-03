from .constant_covariance import ConstantCovariance;
from .covariance_target import CovarianceTarget
from .adhoc_covariance import AdhocCovariance;
from .diagonal_covariance import DiagonalCovariance;
from .adaptive_covariance import AdaptiveCovariance;
from .spherical_covariance import SphericalCovariance;
from .adaptive_covariance_select import AdaptiveCovarianceSelect;
from .adaptive_covariance_new import AdaptiveCovarianceNew;
from .adaptive_covariance_eig import AdaptiveCovarianceEig;
from .nes_covariance import NESCovariance;
from .cmaes_covariance import CMAESCovariance;
from .amalgam_covariance import AMaLGaMCovariance;
def covariance_classes():
    from .constant_covariance import ConstantCovariance;
    from .covariance_target import CovarianceTarget
    from .adhoc_covariance import AdhocCovariance;
    from .diagonal_covariance import DiagonalCovariance;
    from .adaptive_covariance import AdaptiveCovariance;
    from .spherical_covariance import SphericalCovariance;
    from .adaptive_covariance_select import AdaptiveCovarianceSelect;
    from .adaptive_covariance_new import AdaptiveCovarianceNew;
    from .adaptive_covariance_eig import AdaptiveCovarianceEig;
    from .nes_covariance import NESCovariance;
    from .cmaes_covariance import CMAESCovariance;
    from .amalgam_covariance import AMaLGaMCovariance;
    return locals();
