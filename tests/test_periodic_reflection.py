"""Synthetic checks for complex projective fitting and held-out prediction."""

import numpy as np
import pytest

from deflector_tuning.analysis.periodic_reflection import (
    apply_mobius,
    evaluate_held_out_sequence,
    fit_mobius_transition,
    propagate_mobius,
    reflection_errors,
)


MATRIX = np.array([[0.81 + 0.23j, 0.17 - 0.09j],
                   [-0.14 + 0.08j, 1.02 - 0.16j]])


def _known_sequence(initial=0.7 - 0.4j, count=9):
    values = [initial]
    for _ in range(count - 1):
        z = values[-1]
        values.append((MATRIX[0, 0] * z + MATRIX[0, 1]) /
                      (MATRIX[1, 0] * z + MATRIX[1, 1]))
    return np.asarray(values)


def test_complex_svd_recovers_known_map_and_implicit_null_dimension():
    z = np.array([0.2 + 0.8j, -0.6 + 0.1j, 0.4 - 0.5j])
    target = (MATRIX[0, 0] * z + MATRIX[0, 1]) / (MATRIX[1, 0] * z + MATRIX[1, 1])
    fit = fit_mobius_transition(z, target)
    probe = np.array([0.9 + 0.2j, -0.7 - 0.1j])
    expected = (MATRIX[0, 0] * probe + MATRIX[0, 1]) / (MATRIX[1, 0] * probe + MATRIX[1, 1])
    np.testing.assert_allclose(apply_mobius(fit.matrix, probe), expected, atol=2e-14)
    assert fit.design_rank == 3
    assert len(fit.singular_values) == 3  # The fourth zero is implicit, not sigma_3.
    assert fit.sigma4 is None
    assert fit.condition_sigma1_over_sigma3 == pytest.approx(fit.singular_values[0] / fit.singular_values[2])
    assert fit.normalized_determinant == pytest.approx(abs(np.linalg.det(MATRIX)) / np.linalg.norm(MATRIX)**2)


def test_overdetermined_fit_reports_fourth_singular_value_separately():
    sequence = _known_sequence()
    fit = fit_mobius_transition(sequence[:-1], sequence[1:])
    assert len(fit.singular_values) == 4
    assert fit.design_rank == 3
    assert fit.sigma4 < 1e-14
    np.testing.assert_allclose(apply_mobius(fit.matrix, sequence[:-1]), sequence[1:], atol=2e-13)


def test_held_out_propagation_recovers_truth_using_first_four_states_only():
    sequence = _known_sequence()
    result = evaluate_held_out_sequence(sequence)
    assert result.training_state_count == 4
    assert result.fit.transition_count == 3
    np.testing.assert_allclose(result.observed, sequence[4:])
    np.testing.assert_allclose(result.predicted, sequence[4:], atol=2e-13)
    np.testing.assert_allclose(result.phase_error_deg, 0, atol=1e-10)
    np.testing.assert_allclose(result.relative_complex_error, 0, atol=1e-12)


def test_held_out_observations_do_not_reset_open_loop_predictions():
    sequence = _known_sequence()
    baseline = evaluate_held_out_sequence(sequence)
    sequence[4:] += np.linspace(0.05, 0.4, len(sequence) - 4) * (1 + 0.5j)
    changed = evaluate_held_out_sequence(sequence)
    np.testing.assert_array_equal(changed.predicted, baseline.predicted)
    assert len(changed.phase_error_deg) == len(sequence) - 4
    assert np.max(changed.relative_complex_error) > 0.05


def test_inverse_map_and_phase_wrap_preserve_complex_convention():
    z = np.exp(1j * np.deg2rad([179.0, -179.0, 45.0]))
    transformed = apply_mobius(MATRIX, z)
    np.testing.assert_allclose(apply_mobius(np.linalg.inv(MATRIX), transformed), z, atol=1e-14)
    observed = np.exp(1j * np.deg2rad([179.0, -179.0]))
    predicted = np.exp(1j * np.deg2rad([-179.0, 179.0]))
    phase_error, complex_error = reflection_errors(predicted, observed)
    np.testing.assert_allclose(phase_error, [2.0, -2.0], atol=1e-12)
    np.testing.assert_allclose(complex_error, 2 * np.sin(np.deg2rad(1.0)))


def test_projective_scaling_does_not_change_map_or_pole_detection():
    z = np.array([0.3 + 0.7j, -0.2j])
    np.testing.assert_allclose(apply_mobius(MATRIX * (1e-20 + 2e-20j), z), apply_mobius(MATRIX, z))


@pytest.mark.parametrize("current,target", [
    ([1j, 1j, 1j], [1, 1, 1]),
    ([1, 2, 3], [0.4j, 0.4j, 0.4j]),
])
def test_unidentifiable_repeated_inputs_or_singular_map_are_rejected(current, target):
    with pytest.raises(ValueError, match="rank|singular"):
        fit_mobius_transition(current, target)


def test_singular_matrix_is_rejected():
    with pytest.raises(ValueError, match="singular"):
        apply_mobius(np.array([[1, 2], [2, 4]], dtype=complex), 0.5)


@pytest.mark.parametrize("scale", [1, 1e-20, 1e20j])
def test_denominator_near_zero_is_rejected_before_propagation(scale):
    matrix = scale * np.array([[1, 0], [1, -1]], dtype=complex)
    with pytest.raises(ValueError, match="denominator"):
        propagate_mobius(matrix, 1 + 1e-14j, 2)


@pytest.mark.parametrize("states", [[1, 2, 3, 4], [1, 2, 3, 4, np.nan]])
def test_sequence_requires_finite_held_out_states(states):
    with pytest.raises(ValueError):
        evaluate_held_out_sequence(states)


def test_zero_observation_has_no_defined_phase_or_relative_error():
    with pytest.raises(ValueError, match="nonzero"):
        reflection_errors([1j], [0j])
