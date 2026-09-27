from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deflector_tuning.dispersion import (
    DoubleChainParameters,
    predict_double_chain_frequencies,
    track_nearest_mode_branch,
)


def test_double_chain_prediction_reproduces_the_four_band_edges() -> None:
    parameters = DoubleChainParameters.from_band_edges(
        lower_f0_ghz=3.10,
        lower_fpi_ghz=2.80,
        upper_f0_ghz=3.20,
        upper_fpi_ghz=5.00,
    )

    prediction = predict_double_chain_frequencies(
        np.array([0.0, 180.0]),
        parameters,
    )

    np.testing.assert_allclose(prediction["lower_freq_GHz"], [3.10, 2.80], atol=1e-12)
    np.testing.assert_allclose(prediction["upper_freq_GHz"], [3.20, 5.00], atol=1e-12)


def test_branch_tracking_follows_the_nearest_candidate_across_a_mode_index_switch() -> None:
    dispersion = pd.DataFrame(
        {
            "mode_index": [3, 3, 3, 5, 5, 5],
            "phase_deg": [0.0, 90.0, 180.0, 0.0, 90.0, 180.0],
            "freq_GHz": [3.20, 4.02, 4.10, 4.30, 4.35, 4.80],
        }
    )

    tracked = track_nearest_mode_branch(
        dispersion,
        phase_deg=np.array([0.0, 90.0, 180.0]),
        predicted_freq_ghz=np.array([3.21, 4.00, 4.79]),
        candidate_mode_indices=(3, 5),
    )

    assert tracked["observed_mode_index"].tolist() == [3, 3, 5]
    np.testing.assert_allclose(tracked["observed_freq_GHz"], [3.20, 4.02, 4.80])


def test_double_chain_parameters_reject_nonpositive_band_edges() -> None:
    with pytest.raises(ValueError, match="positive finite frequencies"):
        DoubleChainParameters.from_band_edges(
            lower_f0_ghz=3.10,
            lower_fpi_ghz=0.0,
            upper_f0_ghz=3.20,
            upper_fpi_ghz=5.00,
        )
