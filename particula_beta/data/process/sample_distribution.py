"""
Sample from a tabulated probability density function (PDF).
This function allows you to draw random samples from a 1-D PDF that is

"""

import numpy as np
from numpy.typing import NDArray
from typing import Union, Sequence


def get_samples_from_tabulated_pdf(
    distribution_bins: NDArray[np.float64],
    distribution_pdf: NDArray[np.float64],
    number_of_samples: int,
    random_generator: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Draw random samples from a 1-D PDF that is supplied as tabulated values.

    The function assumes `x` is strictly increasing and `pdf` ≥ 0 for all
    points.  If `pdf` is not already normalised, it is normalised internally.

    Args:
        x: 1-D array of the abscissa values (e.g. particle radii in metres).
        pdf: 1-D array of the *unnormalised* probability-density values
            evaluated at each `x`.  Must be the same length as `x`.
        n_samples: Number of random variates to return.
        rng: Optional `np.random.Generator` for reproducibility.

    Returns:
        1-D NumPy array of length `n_samples` containing samples drawn from the
        distribution represented by (`x`, `pdf`).

    Example:
        ```python
        # Mixture of two lognormals, tabulated on a log-spaced grid
        import numpy as np
        from scipy.stats import lognorm

        x_grid = np.geomspace(1e-9, 5e-7, 10000)
        modes   = np.array([5e-8, 1e-7])
        gsds    = np.array([1.5,  2.0])
        weights = np.array([1e9, 5e9])

        pdf_vals = sum(
            w * lognorm(s=np.log(gsd), scale=mode).pdf(x_grid)
            for w, gsd, mode in zip(weights, gsds, modes)
        )

        radii = sample_from_tabulated_pdf(x_grid, pdf_vals, 10_000)
        ```
    """
    rng = (
        np.random.default_rng()
        if random_generator is None
        else random_generator
    )
    distribution_bins = np.asarray(distribution_bins, dtype=np.float64)
    distribution_pdf = np.asarray(distribution_pdf, dtype=np.float64)

    if (
        distribution_bins.ndim != 1
        or distribution_pdf.ndim != 1
        or distribution_bins.size != distribution_pdf.size
    ):
        raise ValueError(
            "`x` and `pdf` must be 1-D arrays of the same length."
        )

    if np.any(np.diff(distribution_bins) <= 0):
        raise ValueError("`x` must be strictly increasing.")

    if np.any(distribution_pdf < 0):
        raise ValueError("`pdf` contains negative values.")

    # --- Build the (normalised) CDF ----------------------------------------
    # Trapezoidal rule → cumulative integral
    cdf = np.cumsum(
        (distribution_pdf[:-1] + distribution_pdf[1:])
        / 2
        * np.diff(distribution_bins),
        dtype=np.float64,
    )
    cdf = np.hstack([0.0, cdf])  # prepend CDF( x[0] ) = 0
    cdf /= cdf[-1]  # normalise so CDF(x[-1]) = 1

    # --- Inverse-transform sampling ----------------------------------------
    u = rng.random(number_of_samples)
    return np.interp(u, cdf, distribution_bins)
