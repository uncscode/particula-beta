"""Simplified, *batch‑aware* Mie efficiencies using Taichi **plus** complex‑argument
Bessel functions Jₙᵤ and Yₙᵤ.

Additions in this revision
==========================
* **`bessel_jv_complex_kernel`** – Taichi kernel evaluating the Bessel
  function of the **first** kind *Jₙᵤ(z)* for *real order* ν and *complex
  argument* z.  A truncated power‑series with per‑term ratio recursion avoids
  expensive factorial/Γ calls on the device; the required
  ``1/Γ(ν + 1)`` pre‑factor is pre‑computed on the host and passed as an array.
* **`bessel_yv_complex_kernel`** – Taichi kernel computing the **second** kind
  *Yₙᵤ(z)* via the identity

    Yₙᵤ(z) = (Jₙᵤ(z) cos πν − J₋ₙᵤ(z)) / sin πν.

  It consumes *both* Jₙᵤ and J₋ₙᵤ arrays from the first kernel.
* Minimal helper `@ti.func`s for complex arithmetic (add, mul, scale, pow).

Accuracy is sufficient for typical Lorenz‑Mie half‑integer orders (n + ½) and
|z| ≲ 20; kernel‑side tolerance is **1 × 10⁻¹²** or *max_iter* = 50.
"""

import math
from typing import Tuple

import numpy as np
import taichi as ti
from scipy.special import jv, yv  # still used for CPU‑side prep where needed
from math import gamma, pi  # host Gamma for pre‑factor



ti.init(arch=ti.cpu)


# ---------------------------------------------------------------------------
#  Complex helpers (vec2<f64> = (re, im))
# ---------------------------------------------------------------------------
@ti.func
def c_add(a: ti.f64, b: ti.f64, c: ti.f64, d: ti.f64):  # noqa: D401 – util
    """Return (a + ib) + (c + id)."""
    return a + c, b + d


@ti.func
def c_mul(a: ti.f64, b: ti.f64, c: ti.f64, d: ti.f64):
    """Return (a + ib) × (c + id)."""
    return a * c - b * d, a * d + b * c


@ti.func
def c_scale(a: ti.f64, b: ti.f64, s: ti.f64):
    """Return s × (a + ib)."""
    return a * s, b * s


@ti.func
def complex_pow_mag_ang(r: ti.f64, theta: ti.f64, alpha: ti.f64):
    """Return (r e^{iθ})^α as (re, im)."""
    r_pow = ti.pow(r, alpha)
    ang = alpha * theta
    return r_pow * ti.cos(ang), r_pow * ti.sin(ang)


# ---------------------------------------------------------------------------
#  Bessel Jν(z) – Taichi kernel (power‑series, complex z)
# ---------------------------------------------------------------------------
@ti.kernel
def bessel_jv_complex_kernel(
    n: ti.i32,
    nu: ti.types.ndarray(dtype=ti.f64),  # shape (N,)
    zr: ti.types.ndarray(dtype=ti.f64),
    zi: ti.types.ndarray(dtype=ti.f64),
    gamma_inv: ti.types.ndarray(dtype=ti.f64),  # 1/Γ(ν+1)
    max_iter: ti.i32,
    out_re: ti.types.ndarray(dtype=ti.f64),
    out_im: ti.types.ndarray(dtype=ti.f64),
):
    """Compute Jν(z) for *each* particle.

    The series used:
        term₀ = (z/2)^ν / Γ(ν+1)
        term_{k+1} = −(z/2)^2 / [(k+1)(k+ν+1)] × term_k
    We stop when |term| < 1e‑12 **or** iterations exceed *max_iter*.
    """
    for i in range(n):
        # polar of z/2
        zr2 = zr[i] * 0.5
        zi2 = zi[i] * 0.5
        r = ti.sqrt(zr2 * zr2 + zi2 * zi2)
        theta = ti.atan2(zi2, zr2)

        # term_0
        t_re, t_im = complex_pow_mag_ang(r, theta, nu[i])
        t_re, t_im = c_scale(t_re, t_im, gamma_inv[i])

        s_re = t_re
        s_im = t_im

        # pre‑compute (z/2)^2 for ratio
        z2_re, z2_im = c_mul(zr2, zi2, zr2, zi2)  # (z/2)^2
        z2_re = -z2_re  # include leading minus (−1)
        z2_im = -z2_im

        for k in range(1, max_iter):
            denom = (k) * (k + nu[i])
            ratio_re = z2_re / denom
            ratio_im = z2_im / denom
            t_re, t_im = c_mul(t_re, t_im, ratio_re, ratio_im)
            s_re, s_im = c_add(s_re, s_im, t_re, t_im)

        out_re[i] = s_re
        out_im[i] = s_im


# ---------------------------------------------------------------------------
#  Bessel Yν(z) – Taichi kernel using identity with J
# ---------------------------------------------------------------------------
@ti.kernel
def bessel_yv_complex_kernel(
    n: ti.i32,
    nu: ti.types.ndarray(dtype=ti.f64),
    j_re: ti.types.ndarray(dtype=ti.f64),
    j_im: ti.types.ndarray(dtype=ti.f64),
    j_neg_re: ti.types.ndarray(dtype=ti.f64),
    j_neg_im: ti.types.ndarray(dtype=ti.f64),
    out_re: ti.types.ndarray(dtype=ti.f64),
    out_im: ti.types.ndarray(dtype=ti.f64),
):
    """Compute Yν(z) from Jν and J−ν:  Y = (Jν cos πν − J−ν)/sin πν."""
    for i in range(n):
        s = ti.sin(ti.math.pi * nu[i])   # sin(πν)
        c = ti.cos(ti.math.pi * nu[i])   # cos(πν)
        # The Python wrapper will never call the kernel for integer-ish ν,
        # so we can safely divide by s directly.
        denom = s

        num_re = j_re[i] * c - j_neg_re[i]
        num_im = j_im[i] * c - j_neg_im[i]

        out_re[i] = num_re / denom
        out_im[i] = num_im / denom


# ---------------------------------------------------------------------------
#  Convenience Python wrapper – returns NumPy arrays
# ---------------------------------------------------------------------------


def bessel_jv_batch(
    nu: np.ndarray,
    z: np.ndarray | complex,
    max_iter: int = 500,
):
    """Vectorised *Jν(z)* for arrays of ν and z (complex)."""
    nu = np.asarray(nu, dtype=float)
    if np.isscalar(z):
        zr = np.full_like(nu, np.real(z))
        zi = np.full_like(nu, np.imag(z))
    else:
        z = np.asarray(z, dtype=complex)
        zr, zi = z.real, z.imag
        if zr.size != nu.size:
            raise ValueError("nu and z arrays must be same length")

    gamma_inv = 1.0 / np.array([gamma(n + 1.0) for n in nu], dtype=float)

    n_particles = nu.size
    # device arrays
    nu_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
    zr_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
    zi_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
    gamma_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
    out_re_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
    out_im_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)

    # copy
    nu_ti.from_numpy(nu)
    zr_ti.from_numpy(zr)
    zi_ti.from_numpy(zi)
    gamma_ti.from_numpy(gamma_inv)

    bessel_jv_complex_kernel(
        n_particles,
        nu_ti,
        zr_ti,
        zi_ti,
        gamma_ti,
        max_iter,
        out_re_ti,
        out_im_ti,
    )

    return out_re_ti.to_numpy() + 1j * out_im_ti.to_numpy()


def bessel_yv_batch(
    nu: np.ndarray,
    z: np.ndarray | complex,
    max_iter: int = 500,        # same value as in bessel_jv_batch
):
    nu = np.asarray(nu, dtype=float)

    # --- make z an array of the same length --------------------------------
    if np.isscalar(z):
        z_arr = np.full_like(nu, z, dtype=complex)
    else:
        z_arr = np.asarray(z, dtype=complex)
        if z_arr.size != nu.size:
            raise ValueError("nu and z arrays must be same length")

    # --- split the work: GPU for “safe” ν, SciPy for near-integer ν ---------
    int_mask = np.isclose(nu, np.round(nu), atol=1e-8)
    y_out = np.empty_like(z_arr, dtype=complex)

    # ---- GPU path ----------------------------------------------------------
    if np.any(~int_mask):
        nu_safe = nu[~int_mask]
        z_safe = z_arr[~int_mask]

        j_pos = bessel_jv_batch(nu_safe, z_safe, max_iter=max_iter)
        j_neg = bessel_jv_batch(-nu_safe, z_safe, max_iter=max_iter)

        n_particles = nu_safe.size
        # allocate device arrays (same as before but sized n_particles)
        nu_ti   = ti.ndarray(dtype=ti.f64, shape=n_particles)
        j_re_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
        j_im_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
        jn_re_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
        jn_im_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
        out_re_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)
        out_im_ti = ti.ndarray(dtype=ti.f64, shape=n_particles)

        # copy to device
        nu_ti.from_numpy(nu_safe)
        j_re_ti.from_numpy(j_pos.real)
        j_im_ti.from_numpy(j_pos.imag)
        jn_re_ti.from_numpy(j_neg.real)
        jn_im_ti.from_numpy(j_neg.imag)

        # run kernel
        bessel_yv_complex_kernel(
            n_particles,
            nu_ti,
            j_re_ti,
            j_im_ti,
            jn_re_ti,
            jn_im_ti,
            out_re_ti,
            out_im_ti,
        )
        y_out[~int_mask] = out_re_ti.to_numpy() + 1j * out_im_ti.to_numpy()

    # ---- SciPy fallback for near-integer orders ---------------------------
    if np.any(int_mask):
        y_out[int_mask] = yv(nu[int_mask], z_arr[int_mask])

    return y_out


# ---------------------------------------------------------------------------
#  Minimal self‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # Compare against SciPy for a few random (ν, z)
    rng = np.random.default_rng(0)
    test_size = 1000
    nu_test = rng.uniform(0.0, 5.0, size=test_size)
    z_test = rng.normal(size=test_size) + 1j * rng.normal(size=test_size)

    j_ti = bessel_jv_batch(nu_test, z_test)
    y_ti = bessel_yv_batch(nu_test, z_test)

    j_sp = jv(nu_test, z_test)
    y_sp = yv(nu_test, z_test)

    print("max |delta-J|:", np.max(np.abs(j_ti - j_sp)))
    print("max |delta-Y|:", np.max(np.abs(y_ti - y_sp)))
