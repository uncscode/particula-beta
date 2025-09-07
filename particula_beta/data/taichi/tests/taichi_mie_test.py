"""Numerical validation of Taichi-accelerated Mie efficiencies.

We run the public `compute_mie_efficiencies` against the reference
`PyMieScatt.AutoMieQ` implementation and assert numerical agreement for the
three efficiencies that are returned (Qext, Qsca, Qabs).  A mix of particles
both inside and outside the Rayleigh regime (x ≤ 0.05) is included.
"""

import numpy as np
import PyMieScatt as ps  # reference implementation
import pytest

from particula_beta.data.taichi.taichi_mie import (  # noqa: E402
    compute_mie_efficiencies,
)

# ---------------------------------------------------------------------------
#  Helper – vectorised reference via PyMieScatt
# ---------------------------------------------------------------------------


def _reference_pms(
    m: complex | np.ndarray, wavelength: float, diameter: np.ndarray
) -> np.ndarray:
    """Return (N,3) array of (Qext,Qsca,Qabs) via PyMieScatt."""
    if np.isscalar(m):
        m_arr = np.full(diameter.size, m, dtype=np.complex128)
    else:
        m_arr = np.asarray(m, dtype=np.complex128)
    ref = np.empty((diameter.size, 3), dtype=np.float64)
    for i, (m_i, d_i) in enumerate(zip(m_arr, diameter, strict=True)):
        qext, qsca, qabs, *_ = ps.AutoMieQ(
            m=m_i,
            wavelength=wavelength,
            diameter=d_i,
            nMedium=1.0,
        )
        ref[i] = qext, qsca, qabs
    return ref


# ---------------------------------------------------------------------------
#  Parametrised test cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m_particle",
    [1.33 + 0.0j,
     1.5 + 0.01j,
     1.1 + 0.0j,
     2.0 + 0.0j,
     2.0 + 1.0j,
     1.5 + 1.0j,
     ],  # non-absorbing and weakly absorbing
)
def test_mie_efficiencies_against_pymiescatt(m_particle: complex) -> None:
    """Ensure Taichi and PyMieScatt agree within tight tolerances."""
    wavelength_nm = 550.0  # visible green
    # two Rayleigh-scale diameters + three Mie-scale diameters
    diam_nm = np.array([1.0, 5.0, 100.0, 200.0, 400.0, 1000.0, 10_000, 100_000], dtype=float)

    qs_taichi = compute_mie_efficiencies(m_particle, wavelength_nm, diam_nm)
    qs_ref = _reference_pms(m_particle, wavelength_nm, diam_nm)

    # ——— numeric tolerance ——————————————————————————
    # Rayleigh analytic part should be exact to ~1e-12; full Mie part looser.
    np.testing.assert_allclose(qs_taichi, qs_ref, rtol=5e-4, atol=1e-6)
