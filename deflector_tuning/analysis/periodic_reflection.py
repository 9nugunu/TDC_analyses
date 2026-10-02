"""Empirical complex reflection recurrence and held-out sequence prediction.

The projective map describes a sequence of exported reflection coefficients.
Fitting it does not identify a de-embedded cell matrix or validate matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class MobiusFit:
    """Projective matrix and diagnostics of the homogeneous complex fit.

    ``normalized_determinant`` is ``abs(det(M)) / norm(M, 'fro')**2``.
    ``condition_sigma1_over_sigma3`` describes the three constrained directions.
    For three transitions the design is 3 by 4: its right null dimension is
    implicit and the three returned singular values are not a null-gap test.
    """

    matrix: NDArray[np.complex128]
    singular_values: tuple[float, ...]
    design_rank: int
    condition_sigma1_over_sigma3: float
    normalized_determinant: float
    transition_count: int

    @property
    def sigma4(self) -> float | None:
        """Return the fourth singular value only for an overdetermined fit."""
        return self.singular_values[3] if len(self.singular_values) == 4 else None


@dataclass(frozen=True)
class ReflectionPrediction:
    """Fit and held-out-only arrays, all in increasing sequence order."""

    fit: MobiusFit
    training_state_count: int
    observed: NDArray[np.complex128]
    predicted: NDArray[np.complex128]
    phase_error_deg: NDArray[np.float64]
    relative_complex_error: NDArray[np.float64]


def _complex_vector(values: ArrayLike, name: str) -> NDArray[np.complex128]:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional sequence")
    return array


def _normalized_matrix(matrix: ArrayLike) -> tuple[NDArray[np.complex128], float]:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.shape != (2, 2) or not np.isfinite(array).all():
        raise ValueError("Map matrix must be finite and have shape (2, 2)")
    # First remove large/small projective scale without squaring it.
    scale = np.max(np.abs(array))
    if scale == 0:
        raise ValueError("Map matrix is singular")
    array = array / scale
    array = array / np.linalg.norm(array)
    determinant = float(abs(np.linalg.det(array)))
    if determinant <= 1e-12:
        raise ValueError("Map matrix is singular or numerically noninvertible")
    return array, determinant


def fit_mobius_transition(current: ArrayLike, following: ArrayLike) -> MobiusFit:
    """Fit ``following = (a * current + b) / (c * current + d)``.

    Parameters
    ----------
    current, following : array_like of complex, shape (n,)
        Paired finite states with at least three transitions. Repeated inputs
        or a constant-output singular map cannot identify an invertible map.

    Returns
    -------
    MobiusFit
        Unit-Frobenius-norm projective matrix and design diagnostics. Three
        transitions give an exactly determined fit, not validation evidence.

    Notes
    -----
    Rows are ``[z, 1, -z*z_next, -z_next]``. Complex SVD uses the conjugate
    of the final row of the full ``Vh`` so the underdetermined null vector is
    retained. With more than three transitions, the fourth singular value
    reports the residual direction separately from ``sigma1 / sigma3``.
    """
    z = _complex_vector(current, "current")
    next_z = _complex_vector(following, "following")
    if z.shape != next_z.shape or len(z) < 3:
        raise ValueError("At least three paired transitions are required")
    design = np.column_stack((z, np.ones(len(z)), -z * next_z, -next_z))
    if not np.isfinite(design).all():
        raise ValueError("Transition design contains non-finite products")
    _, singular_values, vh = np.linalg.svd(design, full_matrices=True)
    tolerance = max(design.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank < 3:
        raise ValueError(f"Transition design rank {rank} cannot identify an invertible map")
    matrix, determinant = _normalized_matrix(vh[-1].conj().reshape(2, 2))
    return MobiusFit(
        matrix=matrix,
        singular_values=tuple(float(value) for value in singular_values),
        design_rank=rank,
        condition_sigma1_over_sigma3=float(singular_values[0] / singular_values[2]),
        normalized_determinant=determinant,
        transition_count=len(z),
    )


def apply_mobius(
    matrix: ArrayLike, values: ArrayLike, *, denominator_tolerance: float = 1e-12
) -> NDArray[np.complex128]:
    """Apply an invertible projective map with a scale-independent pole guard.

    The denominator threshold is relative to the norm of the bottom matrix
    row times the norm of ``[z, 1]``. The returned shape matches ``values``.
    """
    if not np.isfinite(denominator_tolerance) or denominator_tolerance <= 0:
        raise ValueError("Denominator tolerance must be finite and positive")
    normalized, _ = _normalized_matrix(matrix)
    z = np.asarray(values, dtype=np.complex128)
    if not np.isfinite(z).all():
        raise ValueError("Map inputs must be finite")
    a, b, c, d = normalized.ravel()
    denominator = c * z + d
    threshold = denominator_tolerance * np.linalg.norm(normalized[1]) * np.hypot(np.abs(z), 1)
    if np.any(np.abs(denominator) <= threshold):
        raise ValueError("Map denominator is zero or too close to a pole")
    predicted = np.asarray((a * z + b) / denominator, dtype=np.complex128)
    if not np.isfinite(predicted).all():
        raise ValueError("Map prediction is non-finite")
    return predicted


def propagate_mobius(matrix: ArrayLike, initial: complex, steps: int) -> NDArray[np.complex128]:
    """Predict successive states in open loop, excluding the initial state."""
    count = index(steps)
    if count < 0:
        raise ValueError("Propagation steps must be nonnegative")
    predicted = np.empty(count, dtype=np.complex128)
    state = complex(initial)
    for step in range(count):
        state = complex(apply_mobius(matrix, state))
        predicted[step] = state
    return predicted


def reflection_errors(
    predicted: ArrayLike, observed: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return signed circular phase error and pointwise relative complex error.

    Phase error is ``arg(predicted / observed)`` in degrees, wrapped to
    ``[-180, 180)``. Relative error is ``abs(predicted-observed)/abs(observed)``.
    Zero reflection has undefined phase and is rejected instead of hidden.
    """
    expected = _complex_vector(observed, "observed")
    estimated = _complex_vector(predicted, "predicted")
    if expected.shape != estimated.shape:
        raise ValueError("Prediction and observation shapes must match")
    if np.any(np.abs(expected) == 0) or np.any(np.abs(estimated) == 0):
        raise ValueError("Phase and relative-error inputs must be nonzero")
    phase = (np.rad2deg(np.angle(estimated) - np.angle(expected)) + 180) % 360 - 180
    relative = np.abs(estimated - expected) / np.abs(expected)
    return phase, relative


def evaluate_held_out_sequence(
    states: ArrayLike, *, training_transition_count: int = 3
) -> ReflectionPrediction:
    """Fit the initial transitions and predict every later state without reset.

    Parameters
    ----------
    states : array_like of complex, shape (n,)
        Ordered observations at equal increments of the sequence coordinate.
    training_transition_count : int, default 3
        Initial transitions used to fit the map. Exactly three consumes four
        states; all returned error arrays exclude these training states.
    """
    values = _complex_vector(states, "states")
    count = index(training_transition_count)
    if count < 3 or len(values) <= count + 1:
        raise ValueError("Sequence requires at least three training transitions and one held-out state")
    fit = fit_mobius_transition(values[:count], values[1:count + 1])
    observed = values[count + 1:].copy()
    predicted = propagate_mobius(fit.matrix, values[count], len(observed))
    phase, relative = reflection_errors(predicted, observed)
    return ReflectionPrediction(fit, count + 1, observed, predicted, phase, relative)
