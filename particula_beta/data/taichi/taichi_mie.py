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
ti.init(arch=ti.cpu, default_fp=ti.f64)  # use 64-bit floats everywhere
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


def _compute_mie_coefficients(
    refractive_index_array: np.ndarray,
    size_parameter_array: np.ndarray,
    rayleigh_cutoff: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return padded Mie-series coefficient tensors for every particle."""
    # --- pre-allocate ----------------------------------------------------
    particle_count = size_parameter_array.size

    # Bohren–Huffman estimate (upper-bound; identical to _mie_ab length)
    estimated_max_order = np.rint(2 + size_parameter_array + 4 * np.power(size_parameter_array, 1.0 / 3.0)).astype(
        np.int32
    )
    max_order_array = np.where(
        (size_parameter_array <= rayleigh_cutoff) | (size_parameter_array == 0.0), 0, estimated_max_order
    )

    global_max_order = int(max_order_array.max(initial=0))

    coeff_a_real = np.zeros((particle_count, global_max_order), dtype=np.float64)
    coeff_a_imag = np.zeros_like(coeff_a_real)
    coeff_b_real = np.zeros_like(coeff_a_real)
    coeff_b_imag = np.zeros_like(coeff_a_real)

    # --- fill coefficients in-place -------------------------------------
    for idx in range(particle_count):
        particle_max_order = max_order_array[idx]
        if particle_max_order == 0:
            continue  # Rayleigh or zero-size – keep zeros
        an, bn = _mie_ab(refractive_index_array[idx], size_parameter_array[idx])

        # Actual length (defensive – should equal particle_max_order)
        actual_order = an.size
        max_order_array[idx] = actual_order

        coeff_a_real[idx, :actual_order] = an.real
        coeff_a_imag[idx, :actual_order] = an.imag
        coeff_b_real[idx, :actual_order] = bn.real
        coeff_b_imag[idx, :actual_order] = bn.imag

    return max_order_array, coeff_a_real, coeff_a_imag, coeff_b_real, coeff_b_imag


# ---------------------------------------------------------------------------
#  Taichi kernel – *vectorised* Mie Q for all particles
# ---------------------------------------------------------------------------
@ti.kernel  # noqa: D401 – single sentence not needed
def _compute_mie_q_kernel(  # pylint: disable=too-many-arguments
    particle_count: ti.i32,
    global_max_order: ti.i32,
    rayleigh_cutoff: ti.f64,
    size_parameter_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    refractive_index_real: ti.types.ndarray(dtype=ti.f64, ndim=1),
    refractive_index_imag: ti.types.ndarray(dtype=ti.f64, ndim=1),
    max_order_array: ti.types.ndarray(dtype=ti.i32, ndim=1),
    coeff_a_real: ti.types.ndarray(dtype=ti.f64, ndim=2),
    coeff_a_imag: ti.types.ndarray(dtype=ti.f64, ndim=2),
    coeff_b_real: ti.types.ndarray(dtype=ti.f64, ndim=2),
    coeff_b_imag: ti.types.ndarray(dtype=ti.f64, ndim=2),
    efficiencies_out: ti.types.ndarray(dtype=ti.f64, ndim=2),  # shape (N, 3)
):
    """Populate ``efficiencies_out`` with Qext, Qsca, Qabs for every particle."""
    for p in range(particle_count):
        size_param = size_parameter_array[p]
        if size_param == 0.0:
            efficiencies_out[p, 0] = 0.0
            efficiencies_out[p, 1] = 0.0
            efficiencies_out[p, 2] = 0.0
            continue

        if size_param <= rayleigh_cutoff:
            # Rayleigh handled on CPU; kernel just leaves zeros
            efficiencies_out[p, 0] = 0.0
            efficiencies_out[p, 1] = 0.0
            efficiencies_out[p, 2] = 0.0
        else:
            # ---- full Lorenz‑Mie summation -------------------------------
            qext = 0.0
            qsca = 0.0
            for j in range(max_order_array[p]):
                n = j + 1
                n1 = 2 * n + 1
                ar = coeff_a_real[p, j]
                ai = coeff_a_imag[p, j]
                br = coeff_b_real[p, j]
                bi = coeff_b_imag[p, j]
                qext += n1 * (ar + br)
                qsca += n1 * (ar * ar + ai * ai + br * br + bi * bi)
            x2 = size_param * size_param
            qext *= 2.0 / x2
            qsca *= 2.0 / x2
            efficiencies_out[p, 0] = qext
            efficiencies_out[p, 1] = qsca
            efficiencies_out[p, 2] = qext - qsca  # Qabs


# ---------------------------------------------------------------------------
#  Public wrapper – NumPy in, NumPy out
# ---------------------------------------------------------------------------


def compute_mie_efficiencies(  # noqa: C901 – high cyclomatic, but user‑facing wrapper
    m: np.ndarray | complex,
    wavelength: float,
    diameter: np.ndarray,
    n_medium: float = 1.0,
    rayleigh_cutoff: float = 0.05,
) -> np.ndarray:
    """Compute Mie efficiencies (Qext, Qsca, Qabs) for a particle batch.

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
    particle_count = diameter.size

    # Broadcast scalar m to array if necessary
    if np.isscalar(m):
        refractive_index_array = np.full(particle_count, m, dtype=np.complex128)
    else:
        refractive_index_array = np.asarray(m, dtype=np.complex128)
        if refractive_index_array.size != particle_count:
            raise ValueError(
                "m and diameter must have same length if m is array."
            )

    # Medium normalisation
    refractive_index_array /= n_medium
    effective_wavelength = wavelength / n_medium
    size_parameter_array = math.pi * diameter / effective_wavelength

    small_mask = size_parameter_array <= rayleigh_cutoff

    # Pre‑compute coefficients on CPU
    (
        max_order_array,
        coeff_a_real_np,
        coeff_a_imag_np,
        coeff_b_real_np,
        coeff_b_imag_np,
    ) = _compute_mie_coefficients(refractive_index_array, size_parameter_array, rayleigh_cutoff)

    global_max_order = coeff_a_real_np.shape[1]

    # --- Taichi device arrays ---------------------------------------------
    size_parameter_ti = ti.ndarray(dtype=ti.f64, shape=particle_count)
    refr_index_real_ti = ti.ndarray(dtype=ti.f64, shape=particle_count)
    refr_index_imag_ti = ti.ndarray(dtype=ti.f64, shape=particle_count)
    max_order_ti = ti.ndarray(dtype=ti.i32, shape=particle_count)

    coeff_a_real_ti = ti.ndarray(dtype=ti.f64, shape=(particle_count, global_max_order))
    coeff_a_imag_ti = ti.ndarray(dtype=ti.f64, shape=(particle_count, global_max_order))
    coeff_b_real_ti = ti.ndarray(dtype=ti.f64, shape=(particle_count, global_max_order))
    coeff_b_imag_ti = ti.ndarray(dtype=ti.f64, shape=(particle_count, global_max_order))

    efficiencies_ti = ti.ndarray(dtype=ti.f64, shape=(particle_count, 3))

    # Copy data to device
    size_parameter_ti.from_numpy(size_parameter_array)
    refr_index_real_ti.from_numpy(refractive_index_array.real)
    refr_index_imag_ti.from_numpy(refractive_index_array.imag)
    max_order_ti.from_numpy(max_order_array)
    coeff_a_real_ti.from_numpy(coeff_a_real_np)
    coeff_a_imag_ti.from_numpy(coeff_a_imag_np)
    coeff_b_real_ti.from_numpy(coeff_b_real_np)
    coeff_b_imag_ti.from_numpy(coeff_b_imag_np)

    # Kernel launch
    _compute_mie_q_kernel(
        particle_count,
        global_max_order,
        rayleigh_cutoff,
        size_parameter_ti,
        refr_index_real_ti,
        refr_index_imag_ti,
        max_order_ti,
        coeff_a_real_ti,
        coeff_a_imag_ti,
        coeff_b_real_ti,
        coeff_b_imag_ti,
        efficiencies_ti,
    )

    efficiencies = efficiencies_ti.to_numpy()

    # -- analytical Rayleigh regime handled on CPU -------------------------
    if np.any(small_mask):
        m_small = refractive_index_array[small_mask]
        x_small = size_parameter_array[small_mask]

        # vectorized via short loop (N << total)
        rayleigh_vals = np.empty((x_small.size, 3), dtype=np.float64)
        for i, (m_val, x_val) in enumerate(zip(m_small, x_small, strict=True)):
            rayleigh_vals[i] = _rayleigh_q(m_val.real, m_val.imag, x_val)

        efficiencies[small_mask] = rayleigh_vals

    return efficiencies


# ---------------------------------------------------------------------------
#  Self‑test / example
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    wl_nm = 550.0
    diam_nm = np.linspace(50.0, 500.0, 16)
    m_particle = 1.5 + 0.01j

    q = compute_mie_efficiencies(m_particle, wl_nm, diam_nm)
    print("Columns: Qext, Qsca, Qabs")
    np.set_printoptions(precision=4, suppress=True)
    print(q)
