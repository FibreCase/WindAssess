"""Tests for radar QC filter functions."""

import pytest
import pandas as pd
import numpy as np

from src.radar.qc_filter import (
    init_qc_flags,
    angle_diff,
    range_test,
    turbulence_correlation_test,
    vertical_speed_correlation_test,
    vertical_direction_correlation_test,
    flatline_test,
    spike_test,
    missing_test,
    apply_qc,
    radar_run_qc,
    HEIGHTS,
)


class TestInitQcFlags:
    """Tests for init_qc_flags function."""

    def test_creates_qc_columns(self):
        """Test that QC columns are created for all heights."""
        df = pd.DataFrame({"Wind Speed40m": [1.0, 2.0]})
        result = init_qc_flags(df)

        for h in HEIGHTS:
            assert f"QC_{h}m" in result.columns

    def test_qc_columns_initialized_to_zero(self):
        """Test that QC columns are initialized to 0."""
        df = pd.DataFrame({"Wind Speed40m": [1.0, 2.0]})
        result = init_qc_flags(df)

        for h in HEIGHTS:
            assert (result[f"QC_{h}m"] == 0).all()

    def test_does_not_modify_original(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({"Wind Speed40m": [1.0, 2.0]})
        original_len = len(df.columns)
        _ = init_qc_flags(df)
        assert len(df.columns) == original_len


class TestAngleDiff:
    """Tests for angle_diff function."""

    def test_simple_difference(self):
        """Test simple angular difference."""
        assert angle_diff(90, 0) == 90
        assert angle_diff(0, 90) == 90

    def test_wraparound(self):
        """Test wrap-around at 360 degrees."""
        assert angle_diff(350, 10) == 20
        assert angle_diff(10, 350) == 20

    def test_opposite_directions(self):
        """Test opposite directions."""
        assert angle_diff(0, 180) == 180
        assert angle_diff(180, 0) == 180

    def test_same_angle(self):
        """Test same angle."""
        assert angle_diff(45, 45) == 0

    def test_array_input(self):
        """Test with array inputs."""
        a = np.array([0, 90, 180])
        b = np.array([90, 180, 270])
        result = angle_diff(a, b)
        assert np.allclose(result, [90, 90, 90])


class TestRangeTest:
    """Tests for range_test function."""

    def create_test_df(self, **kwargs):
        """Helper to create test DataFrame with specified values."""
        data = {
            "Wind Speed40m": [kwargs.get("ws", 10.0)],
            "Wind Direction40m": [kwargs.get("wd", 180.0)],
            "Vertical Wind Speed40m": [kwargs.get("vw", 0.0)],
            "Wind Speed Std40m": [kwargs.get("std", 2.0)],
        }
        return pd.DataFrame(data)

    def test_valid_values_not_flagged(self):
        """Test that valid values are not flagged."""
        df = self.create_test_df()
        df = init_qc_flags(df)
        result = range_test(df)
        assert result["QC_40m"].iloc[0] == 0

    def test_negative_wind_speed_flagged(self):
        """Test that negative wind speed is flagged."""
        df = self.create_test_df(ws=-1.0)
        df = init_qc_flags(df)
        result = range_test(df)
        assert result["QC_40m"].iloc[0] == 1

    def test_wind_speed_over_75_flagged(self):
        """Test that wind speed > 75 is flagged."""
        df = self.create_test_df(ws=76.0)
        result = range_test(df)
        assert result["QC_40m"].iloc[0] == 1

    def test_invalid_wind_direction_flagged(self):
        """Test that invalid wind direction is flagged."""
        df = self.create_test_df(wd=360.0)
        result = range_test(df)
        assert result["QC_40m"].iloc[0] == 1

    def test_invalid_vertical_speed_flagged(self):
        """Test that vertical wind speed outside range is flagged."""
        df = self.create_test_df(vw=11.0)
        result = range_test(df)
        assert result["QC_40m"].iloc[0] == 1

    def test_invalid_std_flagged(self):
        """Test that std > 20 is flagged."""
        df = self.create_test_df(std=21.0)
        result = range_test(df)
        assert result["QC_40m"].iloc[0] == 1


class TestTurbulenceCorrelationTest:
    """Tests for turbulence_correlation_test function."""

    def test_normal_ti_not_flagged(self):
        """Test normal turbulence intensity is not flagged."""
        df = pd.DataFrame({
            "Wind Speed40m": [10.0],
            "Wind Speed Std40m": [2.0],  # TI = 0.2
        })
        df = init_qc_flags(df)
        result = turbulence_correlation_test(df)
        assert result["QC_40m"].iloc[0] == 0

    def test_high_ti_flagged(self):
        """Test high turbulence intensity (>1) is flagged."""
        df = pd.DataFrame({
            "Wind Speed40m": [5.0],
            "Wind Speed Std40m": [10.0],  # TI = 2.0
        })
        df = init_qc_flags(df)
        result = turbulence_correlation_test(df)
        assert result["QC_40m"].iloc[0] == 2


class TestVerticalSpeedCorrelationTest:
    """Tests for vertical_speed_correlation_test function."""

    def test_small_difference_not_flagged(self):
        """Test small vertical difference is not flagged."""
        df = pd.DataFrame({
            "Wind Speed40m": [10.0],
            "Wind Speed45m": [11.0],
        })
        df = init_qc_flags(df)
        result = vertical_speed_correlation_test(df)
        assert result["QC_40m"].iloc[0] == 0
        assert result["QC_45m"].iloc[0] == 0

    def test_large_difference_flagged(self):
        """Test large vertical difference (>15) is flagged."""
        df = pd.DataFrame({
            "Wind Speed40m": [10.0],
            "Wind Speed45m": [30.0],  # diff = 20
        })
        df = init_qc_flags(df)
        result = vertical_speed_correlation_test(df)
        assert result["QC_40m"].iloc[0] == 2
        assert result["QC_45m"].iloc[0] == 2


class TestVerticalDirectionCorrelationTest:
    """Tests for vertical_direction_correlation_test function."""

    def test_small_direction_diff_not_flagged(self):
        """Test small direction difference is not flagged."""
        df = pd.DataFrame({
            "Wind Direction40m": [180.0],
            "Wind Direction45m": [190.0],
        })
        df = init_qc_flags(df)
        result = vertical_direction_correlation_test(df)
        assert result["QC_40m"].iloc[0] == 0

    def test_large_direction_diff_flagged(self):
        """Test large direction difference (>120) is flagged."""
        df = pd.DataFrame({
            "Wind Direction40m": [0.0],
            "Wind Direction45m": [180.0],
        })
        df = init_qc_flags(df)
        result = vertical_direction_correlation_test(df)
        assert result["QC_40m"].iloc[0] == 2


class TestFlatlineTest:
    """Tests for flatline_test function."""

    def test_variable_series_not_flagged(self):
        """Test variable series is not flagged."""
        df = pd.DataFrame({
            "Wind Speed40m": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        })
        df = init_qc_flags(df)
        result = flatline_test(df, window=6)
        assert result["QC_40m"].iloc[-1] == 0

    def test_constant_series_flagged(self):
        """Test constant series is flagged after window."""
        df = pd.DataFrame({
            "Wind Speed40m": [10.0] * 10,
        })
        df = init_qc_flags(df)
        result = flatline_test(df, window=6)
        # Last rows should be flagged
        assert result["QC_40m"].iloc[-1] == 3


class TestSpikeTest:
    """Tests for spike_test function."""

    def test_normal_change_not_flagged(self):
        """Test normal change is not flagged."""
        df = pd.DataFrame({
            "Wind Speed40m": [10.0, 11.0, 12.0],
        })
        df = init_qc_flags(df)
        result = spike_test(df)
        assert result["QC_40m"].all() == 0

    def test_spike_flagged(self):
        """Test sudden spike (>20 m/s change) is flagged."""
        df = pd.DataFrame({
            "Wind Speed40m": [10.0, 35.0],  # change = 25
        })
        df = init_qc_flags(df)
        result = spike_test(df)
        assert result["QC_40m"].iloc[1] == 3


class TestMissingTest:
    """Tests for missing_test function."""

    def test_complete_data_not_flagged(self):
        """Test complete data is not flagged."""
        df = pd.DataFrame({
            "Wind Speed40m": [10.0],
            "Wind Direction40m": [180.0],
        })
        df = init_qc_flags(df)
        result = missing_test(df)
        assert result["QC_40m"].iloc[0] == 0

    def test_missing_speed_flagged(self):
        """Test missing wind speed is flagged."""
        df = pd.DataFrame({
            "Wind Speed40m": [np.nan],
            "Wind Direction40m": [180.0],
        })
        df = init_qc_flags(df)
        result = missing_test(df)
        assert result["QC_40m"].iloc[0] == 5

    def test_missing_direction_flagged(self):
        """Test missing wind direction is flagged."""
        df = pd.DataFrame({
            "Wind Speed40m": [10.0],
            "Wind Direction40m": [np.nan],
        })
        df = init_qc_flags(df)
        result = missing_test(df)
        assert result["QC_40m"].iloc[0] == 5


class TestApplyQc:
    """Tests for apply_qc function."""

    def test_flagged_values_set_to_nan(self):
        """Test that flagged values are set to NaN."""
        df = pd.DataFrame({
            "Wind Speed40m": [10.0, 20.0],
            "Wind Direction40m": [180.0, 90.0],
            "QC_40m": [0, 1],
        })
        result = apply_qc(df)
        assert np.isnan(result["Wind Speed40m"].iloc[1])
        assert np.isnan(result["Wind Direction40m"].iloc[1])

    def test_unflagged_values_preserved(self):
        """Test that unflagged values are preserved."""
        df = pd.DataFrame({
            "Wind Speed40m": [10.0, 20.0],
            "QC_40m": [0, 0],
        })
        result = apply_qc(df)
        assert result["Wind Speed40m"].iloc[0] == 10.0
        assert result["Wind Speed40m"].iloc[1] == 20.0


class TestRadarRunQc:
    """Integration tests for radar_run_qc function."""

    def test_full_pipeline_returns_dataframe(self):
        """Test that full pipeline returns a DataFrame."""
        df = pd.DataFrame({
            "Wind Speed40m": [10.0, 11.0, 12.0],
            "Wind Direction40m": [180.0, 185.0, 190.0],
            "Wind Speed Std40m": [1.0, 1.5, 2.0],
            "Vertical Wind Speed40m": [0.5, 0.3, 0.2],
        })
        result = radar_run_qc(df)
        assert isinstance(result, pd.DataFrame)

    def test_qc_columns_present(self):
        """Test that all QC columns are present in result."""
        df = pd.DataFrame({
            "Wind Speed40m": [10.0],
            "Wind Direction40m": [180.0],
        })
        result = radar_run_qc(df)
        for h in HEIGHTS:
            assert f"QC_{h}m" in result.columns
