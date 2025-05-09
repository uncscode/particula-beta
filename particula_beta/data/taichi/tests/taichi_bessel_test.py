"""Numerical accuracy tests for Taichi-based complex-argument Bessel
functions implemented in `particula_beta.data.taichi.taichi_bessel`.

The kernels claim ≲1 × 10⁻¹² accuracy for |z| ≤ 20. We therefore sample
real-positive z up to 50 (to stress the series) and assert 1 × 10⁻9
agreement with SciPy’s reference implementation.
"""

import numpy as np
import pytest
from scipy.special import jv, yv

from particula_beta.data.taichi.taichi_bessel import (
    bessel_jv_batch,
    bessel_yv_batch,
)

RTOL = 1e-9
ATOL = 1e-9
SAMPLE_SIZE = 1_000
RNG = np.random.default_rng(0)


def _random_nu_z(sample_size: int):
    """Generate matching ν and z arrays over a large positive range."""
    nu = RNG.uniform(0.0, 10.0, size=sample_size)

    # Mix purely-random z with some near-integer ν cases to hit SciPy fallback
    z = RNG.uniform(0.1, 50.0, size=sample_size)
    # inject a batch whose order is an integer to exercise the alt-path
    nu[:10] = np.arange(0, 10)  # exact integers
    return nu, z


@pytest.mark.parametrize("func_pair", [
    (bessel_jv_batch, jv, "J"),
    (bessel_yv_batch, yv, "Y"),
])
def test_bessel_func_accuracy(func_pair):
    """Compare Taichi kernels against SciPy for a batch of random inputs."""
    taichi_func, scipy_func, tag = func_pair
    nu, z = _random_nu_z(SAMPLE_SIZE)

    taichi_out = taichi_func(nu, z)
    scipy_out = scipy_func(nu, z)

    np.testing.assert_allclose(
        taichi_out.real, scipy_out.real, rtol=RTOL, atol=ATOL,
        err_msg=f"Real part mismatch for Bessel {tag}"
    )
    np.testing.assert_allclose(
        taichi_out.imag, scipy_out.imag, rtol=RTOL, atol=ATOL,
        err_msg=f"Imag part mismatch for Bessel {tag}"
    )
