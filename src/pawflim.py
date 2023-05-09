import numpy as np
from binlets import binlets
from scipy import stats


def phasor(R0, Rn, *, mask=None):
    """Compute phasor by normalizing with zeroth harmonic.

    Parameters
    ----------
    R0, Rn : array_like
        Harmonics (0, n)
    mask : array_like
        Only compute where mask == True.
        If mask is None, it is computed as R0 > 0.
    """
    R0 = R0.real
    if mask is None:
        mask = R0 > 0
    return np.divide(
        Rn,
        R0,
        where=mask,
        out=np.zeros(np.broadcast(R0, Rn).shape, complex),
    )


def phasor_covariance(R0, R1, R2, *, mask=None):
    """Compute phasor covariance matrix.

    Parameters
    ----------
    R0, R1, R2 : array_like
        Harmonics (0, n, 2n)
    mask : array_like
        Only compute where mask == True.
        If mask is None, it is computed as R0 > 0.

    References
    ----------
    Silberberg, M., & Grecco, H. E. (2017). pawFLIM: reducing bias
    and uncertainty to enable lower photon count in FLIM experiments.
    Methods and applications in fluorescence, 5(2), 024016.
    """
    R0 = R0.real
    if mask is None:
        mask = R0 > 0
    r1 = phasor(R0, R1, mask=mask)
    r2 = phasor(R0, R2, mask=mask)
    shape = np.broadcast(R0, r1, r2).shape
    cov = np.empty(shape + (2, 2))
    cov[..., 0, 0] = 1 + r2.real - 2 * r1.real**2
    cov[..., 1, 1] = 1 - r2.real - 2 * r1.imag**2
    cov[..., 0, 1] = cov[..., 1, 0] = r2.imag - 2 * r1.real * r1.imag
    cov = np.divide(cov, 2 * R0[..., None, None], out=cov, where=mask[..., None, None])
    return cov


def _inverse(x):
    """Calculates the inverse for a 2x2 matrix.
    
    Much faster than np.linalg.inv.
    """
    y = np.empty_like(x)
    y[..., 0, 0] = x[..., 1, 1]
    y[..., 1, 1] = x[..., 0, 0]
    y[..., 0, 1] = -x[..., 0, 1]
    y[..., 1, 0] = -x[..., 1, 0]
    y /= np.linalg.det(x)[..., None, None]
    return y


def _threshold_from_sigma(n_sigmas: float) -> float:
    """Threshold value for a bivariate normal distribution
    with the same significance level as N sigmas
    in a unidimensional normal distribuion.

    In other words, n_sigmas=1 returns a threshold for 68.3%.
    """
    p_value = stats.chi.cdf(n_sigmas, df=1)
    return stats.chi2.ppf(p_value, df=2)


def _distance(X, Y):
    """Distance between two phasors given their covariance."""
    diff = phasor(*X[:2]) - phasor(*Y[:2])
    cov = phasor_covariance(*X) + phasor_covariance(*Y)
    inv_cov = _inverse(cov)
    diff = np.atleast_1d(diff).view(float).reshape(diff.shape + (2,))
    distance = (diff[..., None, :] @ (inv_cov @ diff[..., None]))[..., 0, 0]
    return distance


def pawflim(data: np.ndarray, *, n_sigmas: float, levels: int | None = None):
    """pawFLIM denoising.

    Parameters
    ----------
    data : np.ndarray[complex]
        Fourier coefficients (i.e., not normalized phasor),
        where data[i] corresponds to harmonics [0, n, 2n].
        The n-th phasor is calculated as data[1] / data[0].
    n_sigmas : float
        Significance level to test the difference between two phasors,
        given in terms of the equivalent 1D standard deviations.
        n_sigmas=2 corresponds to ~95% (or 5%) significance.
    levels : int
        Number of levels for the wavelet decomposition.
        Controls the maximum averaging area, which has a length of 2**level.

    References
    ----------
    Silberberg, M., & Grecco, H. E. (2017). pawFLIM: reducing bias
    and uncertainty to enable lower photon count in FLIM experiments.
    Methods and applications in fluorescence, 5(2), 024016.
    """
    threshold = _threshold_from_sigma(n_sigmas)

    def test_at_threshold(X, Y):
        return _distance(X, Y) < threshold

    return binlets(data, levels=levels, test=test_at_threshold, linear=False)
