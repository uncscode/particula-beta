"""Simplified, *batch‑aware* Mie efficiencies using Taichi.

This module accelerates the **summation** step of the Lorenz‑Mie series on
GPU/CPU via Taichi.  The heavy per‑particle coefficient generation still runs
on the CPU (SciPy’s Bessel functions), but all particles are processed in a
single Taichi kernel.  Arrays of complex refractive indices and diameters are
therefore handled efficiently.

The public façade exposes **`mie_q_batch`** – a convenience function that:
    1. Normalises inputs to the host medium.
    2. Pre‑computes Mie coefficients (*aₙ*, *bₙ*) for every particle.
    3. Launches the Taichi kernel **`mie_q`** that fills three output arrays
       (Qext, Qsca, Qabs).

Everything is fully type‑hinted, Black‑formatted (≤ 79 columns), and follows
Google‑style docstrings.
"""

import math
from typing import Tuple

import numpy as np
import taichi as ti
from scipy.special import jv, yv

# ---------------------------------------------------------------------------
#  Taichi initialisation (GPU if available)
ti.init(arch=ti.cpu)
# ---------------------------------------------------------------------------
#  Analytic Rayleigh regime (x ≲ 0.05)
# ---------------------------------------------------------------------------


def _rayleigh_q(
    m_re: float, m_im: float, x: float
) -> Tuple[float, float, float]:
    """Return (Qext, Qsca, Qabs) from Rayleigh scattering theory."""
    # (m² − 1)/(m² + 2)
    m2_re = m_re * m_re - m_im * m_im
    m2_im = 2.0 * m_re * m_im

    num_re = m2_re - 1.0
    num_im = m2_im
    den_re = m2_re + 2.0
    den_im = m2_im

    den_mag2 = den_re * den_re + den_im * den_im
    ll_re = (num_re * den_re + num_im * den_im) / den_mag2
    ll_im = (num_im * den_re - num_re * den_im) / den_mag2
    ll_mag2 = ll_re * ll_re + ll_im * ll_im

    qsca = (8.0 / 3.0) * ll_mag2 * x**4
    qabs = 4.0 * x * ll_im
    qext = qsca + qabs
    return qext, qsca, qabs


# ---------------------------------------------------------------------------
#  CPU helper – Mie coefficients for a *single* particle
# ---------------------------------------------------------------------------


def _mie_ab(
    m: complex, x: float
) -> Tuple[np.ndarray, np.ndarray]:  # noqa: D401
    """Return arrays *(aₙ, bₙ)* up to Bohren–Huffman n<sub>max</sub>."""
    mx = m * x
    nmax = int(round(2 + x + 4 * x ** (1.0 / 3.0)))
    nmx = int(round(max(nmax, abs(mx)) + 16))

    n = np.arange(1, nmax + 1)
    nu = n + 0.5

    # Riccati–Bessel ψ and ζ
    sx = math.sqrt(0.5 * math.pi * x)
    px = sx * jv(nu, x)
    p1x = np.concatenate(([math.sin(x)], px[:-1]))

    chx = -sx * yv(nu, x)
    ch1x = np.concatenate(([math.cos(x)], chx[:-1]))

    gsx = px - 1j * chx
    gs1x = p1x - 1j * ch1x

    # Downward recurrence for Dₙ(mx)
    dn = np.zeros(nmx, dtype=complex)
    for k in range(nmx - 1, 1, -1):
        dn[k - 1] = (k / mx) - 1.0 / (dn[k] + k / mx)

    d = dn[1 : nmax + 1]

    an = ((d / m + n / x) * px - p1x) / ((d / m + n / x) * gsx - gs1x)
    bn = ((m * d + n / x) * px - p1x) / ((m * d + n / x) * gsx - gs1x)
    return an, bn


# ---------------------------------------------------------------------------
#  CPU helper – prepare *batch* coefficient tensors
# ---------------------------------------------------------------------------


def _prepare_coefficients(
    m_arr: np.ndarray,
    x_arr: np.ndarray,
    rayleigh_cutoff: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Pre‑compute padded coefficient arrays for all particles."""
    n_particles = len(x_arr)
    nmax_list = []
    an_r_list, an_i_list, bn_r_list, bn_i_list = [], [], [], []

    for m, x in zip(m_arr, x_arr):
        if x <= rayleigh_cutoff or x == 0.0:
            nmax_list.append(0)
            an_r_list.append([0.0])
            an_i_list.append([0.0])
            bn_r_list.append([0.0])
            bn_i_list.append([0.0])
        else:
            an, bn = _mie_ab(m, x)
            nmax_list.append(an.size)
            an_r_list.append(an.real)
            an_i_list.append(an.imag)
            bn_r_list.append(bn.real)
            bn_i_list.append(bn.imag)

    nmax_max = max(nmax_list)

    def _pad(lst: list[np.ndarray]) -> np.ndarray:
        return np.stack(
            [
                np.pad(a, (0, nmax_max - a.size), constant_values=0.0)
                for a in lst
            ],
            axis=0,
        )

    return (
        np.asarray(nmax_list, dtype=np.int32),
        _pad(an_r_list),
        _pad(an_i_list),
        _pad(bn_r_list),
        _pad(bn_i_list),
    )


# ---------------------------------------------------------------------------
#  Taichi kernel – *vectorised* Mie Q for all particles
# ---------------------------------------------------------------------------
@ti.kernel  # noqa: D401 – single sentence not needed
def mie_q(  # pylint: disable=too-many-arguments
    n_particles: ti.i32,
    nmax_max: ti.i32,
    rayleigh_cutoff: ti.f64,
    x_arr: ti.types.ndarray(dtype=ti.f64, ndim=1),
    m_re: ti.types.ndarray(dtype=ti.f64, ndim=1),
    m_im: ti.types.ndarray(dtype=ti.f64, ndim=1),
    nmax_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
    an_r: ti.types.ndarray(dtype=ti.f64, ndim=2),
    an_i: ti.types.ndarray(dtype=ti.f64, ndim=2),
    bn_r: ti.types.ndarray(dtype=ti.f64, ndim=2),
    bn_i: ti.types.ndarray(dtype=ti.f64, ndim=2),
    qs_out: ti.types.ndarray(dtype=ti.f64, ndim=2),  # shape (N, 3)
):
    """Populate ``qs_out`` with Qext, Qsca, Qabs for every particle."""
    for p in range(n_particles):
        x = x_arr[p]
        if x == 0.0:
            qs_out[p, 0] = 0.0
            qs_out[p, 1] = 0.0
            qs_out[p, 2] = 0.0
            continue

        if x <= rayleigh_cutoff:
            # ---- analytic Rayleigh ----------------------------------------
            m_re_p = m_re[p]
            m_im_p = m_im[p]
            # (m² − 1)/(m² + 2)
            m2_re = m_re_p * m_re_p - m_im_p * m_im_p
            m2_im = 2.0 * m_re_p * m_im_p
            num_re = m2_re - 1.0
            num_im = m2_im
            den_re = m2_re + 2.0
            den_im = m2_im
            den_mag2 = den_re * den_re + den_im * den_im
            ll_re = (num_re * den_re + num_im * den_im) / den_mag2
            ll_im = (num_im * den_re - num_re * den_im) / den_mag2
            ll_mag2 = ll_re * ll_re + ll_im * ll_im
            qsca = (8.0 / 3.0) * ll_mag2 * x**4
            qabs = 4.0 * x * ll_im
            qs_out[p, 0] = qsca + qabs  # Qext
            qs_out[p, 1] = qsca  # Qsca
            qs_out[p, 2] = qabs  # Qabs
        else:
            # ---- full Lorenz‑Mie summation -------------------------------
            qext = 0.0
            qsca = 0.0
            for j in range(nmax_arr[p]):
                n = j + 1
                n1 = 2 * n + 1
                ar = an_r[p, j]
                ai = an_i[p, j]
                br = bn_r[p, j]
                bi = bn_i[p, j]
                qext += n1 * (ar + br)
                qsca += n1 * (ar * ar + ai * ai + br * br + bi * bi)
            x2 = x * x
            qext *= 2.0 / x2
            qsca *= 2.0 / x2
            qs_out[p, 0] = qext
            qs_out[p, 1] = qsca
            qs_out[p, 2] = qext - qsca  # Qabs


# ---------------------------------------------------------------------------
#  Public wrapper – NumPy in, NumPy out
# ---------------------------------------------------------------------------


def mie_q_batch(  # noqa: C901 – high cyclomatic, but user‑facing wrapper
    m: np.ndarray | complex,
    wavelength: float,
    diameter: np.ndarray,
    n_medium: float = 1.0,
    rayleigh_cutoff: float = 0.05,
) -> np.ndarray:
    """Vectorised Mie efficiencies.

    Args:
        m: Array of complex refractive indices (shape *N*) or a single value.
        wavelength: Vacuum wavelength (same units as *diameter*).
        diameter: Diameters (shape *N*).
        n_medium: Real refractive index of surrounding medium.
        rayleigh_cutoff: *x* below which Rayleigh approximation is used.

    Returns:
        ``np.ndarray`` of shape *(N, 3)* where columns are
        *(Qext, Qsca, Qabs).*  Order matches **row‑wise** with *m/diameter*
        inputs.
    """
    diameter = np.asarray(diameter, dtype=float)
    n_particles = diameter.size

    # Broadcast scalar m to array if necessary
    if np.isscalar(m):
        m_arr = np.full(n_particles, m, dtype=np.complex128)
    else:
        m_arr = np.asarray(m, dtype=np.complex128)
        if m_arr.size != n_particles:
            raise ValueError(
                "m and diameter must have same length if m is array."
            )

    # Medium normalisation
    m_arr /= n_medium
    wavelength_eff = wavelength / n_medium
    x_arr = math.pi * diameter / wavelength_eff

    # Pre‑compute coefficients on CPU
    (
        nmax_arr,
        an_r_np,
        an_i_np,
        bn_r_np,
        bn_i_np,
    ) = _prepare_coefficients(m_arr, x_arr, rayleigh_cutoff)

    nmax_max = an_r_np.shape[1]

    # --- Taichi device arrays ---------------------------------------------
    x_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
    m_re_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
    m_im_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
    nmax_ti = ti.ndarray(dtype=ti.i32, shape=n_particles)

    an_r_ti = ti.ndarray(dtype=ti.f64, shape=(n_particles, nmax_max))
    an_i_ti = ti.ndarray(dtype=ti.f64, shape=(n_particles, nmax_max))
    bn_r_ti = ti.ndarray(dtype=ti.f64, shape=(n_particles, nmax_max))
    bn_i_ti = ti.ndarray(dtype=ti.f64, shape=(n_particles, nmax_max))

    qs_out_ti = ti.ndarray(dtype=ti.f64, shape=(n_particles, 3))

    # Copy data to device
    x_ti.from_numpy(x_arr)
    m_re_ti.from_numpy(m_arr.real)
    m_im_ti.from_numpy(m_arr.imag)
    nmax_ti.from_numpy(nmax_arr)
    an_r_ti.from_numpy(an_r_np)
    an_i_ti.from_numpy(an_i_np)
    bn_r_ti.from_numpy(bn_r_np)
    bn_i_ti.from_numpy(bn_i_np)

    # Kernel launch
    mie_q(
        n_particles,
        nmax_max,
        rayleigh_cutoff,
        x_ti,
        m_re_ti,
        m_im_ti,
        nmax_ti,
        an_r_ti,
        an_i_ti,
        bn_r_ti,
        bn_i_ti,
        qs_out_ti,
    )

    return qs_out_ti.to_numpy()


# ---------------------------------------------------------------------------
#  Self‑test / example
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    wl_nm = 550.0
    diam_nm = np.linspace(50.0, 500.0, 16)
    m_particle = 1.5 + 0.01j

    q = mie_q_batch(m_particle, wl_nm, diam_nm)
    print("Columns: Qext, Qsca, Qabs")
    np.set_printoptions(precision=4, suppress=True)
    print(q)
