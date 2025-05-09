"""Numerical validation of Taichi-accelerated Mie efficiencies.

We run the public `compute_mie_efficiencies` against the reference
`PyMieScatt.AutoMieQ` implementation and assert numerical agreement for the
three efficiencies that are returned (Qext, Qsca, Qabs).  A mix of particles
both inside and outside the Rayleigh regime (x ≤ 0.05) is included.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import PyMieScatt as ps  # reference implementation
except ModuleNotFoundError:  # pragma: no cover
    ps = None  # allow test discovery even if dependency missing

pytestmark = pytest.mark.skipif(
    ps is None, reason="PyMieScatt is required for reference values"
)

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
    [1.33 + 0.0j, 1.5 + 0.01j],  # non-absorbing and weakly absorbing
)
def test_mie_efficiencies_against_pymiescatt(m_particle: complex) -> None:
    """Ensure Taichi and PyMieScatt agree within tight tolerances."""
    wavelength_nm = 550.0  # visible green
    # two Rayleigh-scale diameters + three Mie-scale diameters
    diam_nm = np.array([1.0, 5.0, 100.0, 200.0, 400.0], dtype=float)

    qs_taichi = compute_mie_efficiencies(m_particle, wavelength_nm, diam_nm)
    qs_ref = _reference_pms(m_particle, wavelength_nm, diam_nm)

    # ——— numeric tolerance ——————————————————————————
    # Rayleigh analytic part should be exact to ~1e-12; full Mie part looser.
    np.testing.assert_allclose(qs_taichi, qs_ref, rtol=5e-4, atol=1e-6)
