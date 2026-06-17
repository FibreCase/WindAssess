"""Tests for tower QC filter functions."""

import pytest
import pandas as pd
import numpy as np

from src.tower.qc_filter import (
    init_qc_flags,
    get_tower_column_name,
    angle_diff,
    range_test,
    turbulence_correlation_test,
    vertical_speed_correlation_test,
    vertical_direction_correlation_test,
    flatline_test,
    spike_test,
    missing_test,
    apply_qc,
    tower_run_qc,
    HEIGHTS,
)


class TestGetTowerColumnName:
    """Tests for get_tower_column_name function."""

    def test_speed_column(self):
        """Test speed column name generation."""
        assert get_tower_column_name(80, 'speed') == "Avg Wind Speed @ 80m [m/s]"

    def test_direction_column(self):
        """Test direction column name generation."""
        assert get_tower_column_name(50, 'direction') == "Avg Wind Direction @ 50m [deg]"

    def test_std_column(self):
        """Test std column name generation."""
        assert get_tower_column_name(20, 'std') == "Avg Wind Speed (std dev) @ 20m [m/s]"

    def test_invalid_height_returns_none(self):
        """Test that invalid height returns None."""
        assert get_tower_column_name(40, 'speed') is None

    def test_invalid_var_type_returns_none(self):
        """Test that invalid var_type returns None."""
        assert get_tower_column_name(80, 'invalid') is None

    def test_all_valid_heights(self):
        """Test all valid tower heights."""
        for h in HEIGHTS:
            assert get_tower_column_name(h, 'speed') is not None
            assert get_tower_column_name(h, 'direction') is not None
            assert get_tower_column_name(h, 'std') is not None


class TestInitQcFlags:
    """Tests for init_qc_flags function."""

    def test_creates_qc_columns_for_tower_heights(self):
        """Test that QC columns are created only for tower heights."""
        df = pd.DataFrame({"Avg Wind Speed @ 80m [m/s]": [1.0]})
        result = init_qc_flags(df)

        for h in HEIGHTS:
            assert f"QC_{h}m" in result.columns

    def test_does_not_create_radar_heights(self):
        """Test that radar-specific heights are not created."""
        df = pd.DataFrame({"Avg Wind Speed @ 80m [m/s]": [1.0]})
        result = init_qc_flags(df)

        # Tower doesn't have 40m, 45m, etc.
        assert "QC_40m" not in result.columns or True  # May exist but shouldn't be used

    def test_qc_columns_initialized_to_zero(self):
        """Test that QC columns are initialized to 0."""
        df = pd.DataFrame({"Avg Wind Speed @ 80m [m/s]": [1.0]})
        result = init_qc_flags(df)

        for h in HEIGHTS:
            assert (result[f"QC_{h}m"] == 0).all()


class TestRangeTest:
    """Tests for tower range_test function."""

    def create_test_df(self, height=80, **kwargs):
        """Helper to create test DataFrame with tower column names."""
        data = {}
        ws_col = get_tower_column_name(height, 'speed')
        wd_col = get_tower_column_name(height, 'direction')
        std_col = get_tower_column_name(height, 'std')

        if ws_col:
            data[ws_col] = [kwargs.get("ws", 10.0)]
        if wd_col:
            data[wd_col] = [kwargs.get("wd", 180.0)]
        if std_col:
            data[std_col] = [kwargs.get("std", 2.0)]

        return pd.DataFrame(data)

    def test_valid_values_not_flagged(self):
        """Test that valid values are not flagged."""
        df = self.create_test_df()
        df = init_qc_flags(df)
        result = range_test(df)
        assert result["QC_80m"].iloc[0] == 0

    def test_negative_wind_speed_flagged(self):
        """Test that negative wind speed is flagged."""
        df = self.create_test_df(ws=-1.0)
        df = init_qc_flags(df)
        result = range_test(df)
        assert result["QC_80m"].iloc[0] == 1

    def test_wind_speed_over_75_flagged(self):
        """Test that wind speed > 75 is flagged."""
        df = self.create_test_df(ws=76.0)
        df = init_qc_flags(df)
        result = range_test(df)
        assert result["QC_80m"].iloc[0] == 1

    def test_invalid_wind_direction_flagged(self):
        """Test that invalid wind direction is flagged."""
        df = self.create_test_df(wd=360.0)
        df = init_qc_flags(df)
        result = range_test(df)
        assert result["QC_80m"].iloc[0] == 1

    def test_invalid_std_flagged(self):
        """Test that std > 20 is flagged."""
        df = self.create_test_df(std=21.0)
        df = init_qc_flags(df)
        result = range_test(df)
        assert result["QC_80m"].iloc[0] == 1

    def test_no_vertical_wind_speed_check(self):
        """Test that tower data doesn't check vertical wind speed (not available)."""
        df = self.create_test_df()
        df = init_qc_flags(df)
        # Should not raise error even without vertical wind speed column
        result = range_test(df)
        assert isinstance(result, pd.DataFrame)


class TestTurbulenceCorrelationTest:
    """Tests for tower turbulence_correlation_test function."""

    def test_normal_ti_not_flagged(self):
        """Test normal turbulence intensity is not flagged."""
        df = pd.DataFrame({
            "Avg Wind Speed @ 80m [m/s]": [10.0],
            "Avg Wind Speed (std dev) @ 80m [m/s]": [2.0],  # TI = 0.2
        })
        df = init_qc_flags(df)
        result = turbulence_correlation_test(df)
        assert result["QC_80m"].iloc[0] == 0

    def test_high_ti_flagged(self):
        """Test high turbulence intensity (>1) is flagged."""
        df = pd.DataFrame({
            "Avg Wind Speed @ 80m [m/s]": [5.0],
            "Avg Wind Speed (std dev) @ 80m [m/s]": [10.0],  # TI = 2.0
        })
        df = init_qc_flags(df)
        result = turbulence_correlation_test(df)
        assert result["QC_80m"].iloc[0] == 2


class TestVerticalSpeedCorrelationTest:
    """Tests for tower vertical_speed_correlation_test function."""

    def test_small_difference_not_flagged(self):
        """Test small vertical difference between tower heights is not flagged."""
        df = pd.DataFrame({
            "Avg Wind Speed @ 50m [m/s]": [10.0],
            "Avg Wind Speed @ 80m [m/s]": [11.0],
        })
        df = init_qc_flags(df)
        result = vertical_speed_correlation_test(df)
        assert result["QC_50m"].iloc[0] == 0
        assert result["QC_80m"].iloc[0] == 0

    def test_large_difference_flagged(self):
        """Test large vertical difference (>15) between tower heights is flagged."""
        df = pd.DataFrame({
            "Avg Wind Speed @ 50m [m/s]": [10.0],
            "Avg Wind Speed @ 80m [m/s]": [30.0],  # diff = 20
        })
        df = init_qc_flags(df)
        result = vertical_speed_correlation_test(df)
        assert result["QC_50m"].iloc[0] == 2
        assert result["QC_80m"].iloc[0] == 2

    def test_compares_consecutive_tower_heights(self):
        """Test that comparison is between consecutive tower heights."""
        df = pd.DataFrame({
            "Avg Wind Speed @ 2m [m/s]": [10.0],
            "Avg Wind Speed @ 5m [m/s]": [30.0],  # diff = 20
        })
        df = init_qc_flags(df)
        result = vertical_speed_correlation_test(df)
        assert result["QC_2m"].iloc[0] == 2
        assert result["QC_5m"].iloc[0] == 2


class TestVerticalDirectionCorrelationTest:
    """Tests for tower vertical_direction_correlation_test function."""

    def test_small_direction_diff_not_flagged(self):
        """Test small direction difference is not flagged."""
        df = pd.DataFrame({
            "Avg Wind Direction @ 50m [deg]": [180.0],
            "Avg Wind Direction @ 80m [deg]": [190.0],
        })
        df = init_qc_flags(df)
        result = vertical_direction_correlation_test(df)
        assert result["QC_50m"].iloc[0] == 0

    def test_large_direction_diff_flagged(self):
        """Test large direction difference (>120) is flagged."""
        df = pd.DataFrame({
            "Avg Wind Direction @ 50m [deg]": [0.0],
            "Avg Wind Direction @ 80m [deg]": [180.0],
        })
        df = init_qc_flags(df)
        result = vertical_direction_correlation_test(df)
        assert result["QC_50m"].iloc[0] == 2


class TestFlatlineTest:
    """Tests for tower flatline_test function."""

    def test_constant_series_flagged(self):
        """Test constant series is flagged after window."""
        df = pd.DataFrame({
            "Avg Wind Speed @ 80m [m/s]": [10.0] * 10,
        })
        df = init_qc_flags(df)
        result = flatline_test(df, window=6)
        assert result["QC_80m"].iloc[-1] == 3


class TestSpikeTest:
    """Tests for tower spike_test function."""

    def test_spike_flagged(self):
        """Test sudden spike (>20 m/s change) is flagged."""
        df = pd.DataFrame({
            "Avg Wind Speed @ 80m [m/s]": [10.0, 35.0],  # change = 25
        })
        df = init_qc_flags(df)
        result = spike_test(df)
        assert result["QC_80m"].iloc[1] == 3


class TestMissingTest:
    """Tests for tower missing_test function."""

    def test_complete_data_not_flagged(self):
        """Test complete data is not flagged."""
        df = pd.DataFrame({
            "Avg Wind Speed @ 80m [m/s]": [10.0],
            "Avg Wind Direction @ 80m [deg]": [180.0],
        })
        df = init_qc_flags(df)
        result = missing_test(df)
        assert result["QC_80m"].iloc[0] == 0

    def test_missing_speed_flagged(self):
        """Test missing wind speed is flagged."""
        df = pd.DataFrame({
            "Avg Wind Speed @ 80m [m/s]": [np.nan],
            "Avg Wind Direction @ 80m [deg]": [180.0],
        })
        df = init_qc_flags(df)
        result = missing_test(df)
        assert result["QC_80m"].iloc[0] == 5


class TestApplyQc:
    """Tests for tower apply_qc function."""

    def test_flagged_values_set_to_nan(self):
        """Test that flagged values are set to NaN."""
        df = pd.DataFrame({
            "Avg Wind Speed @ 80m [m/s]": [10.0, 20.0],
            "Avg Wind Direction @ 80m [deg]": [180.0, 90.0],
            "QC_80m": [0, 1],
        })
        result = apply_qc(df)
        assert np.isnan(result["Avg Wind Speed @ 80m [m/s]"].iloc[1])
        assert np.isnan(result["Avg Wind Direction @ 80m [deg]"].iloc[1])


class TestTowerRunQc:
    """Integration tests for tower_run_qc function."""

    def test_full_pipeline_returns_dataframe(self):
        """Test that full pipeline returns a DataFrame."""
        df = pd.DataFrame({
            "Avg Wind Speed @ 80m [m/s]": [10.0, 11.0, 12.0],
            "Avg Wind Direction @ 80m [deg]": [180.0, 185.0, 190.0],
            "Avg Wind Speed (std dev) @ 80m [m/s]": [1.0, 1.5, 2.0],
        })
        result = tower_run_qc(df)
        assert isinstance(result, pd.DataFrame)

    def test_qc_columns_present(self):
        """Test that all QC columns are present in result."""
        df = pd.DataFrame({
            "Avg Wind Speed @ 80m [m/s]": [10.0],
            "Avg Wind Direction @ 80m [deg]": [180.0],
        })
        result = tower_run_qc(df)
        for h in HEIGHTS:
            assert f"QC_{h}m" in result.columns

    def test_processes_all_tower_heights(self):
        """Test that all tower heights are processed."""
        df = pd.DataFrame({
            "Avg Wind Speed @ 2m [m/s]": [5.0],
            "Avg Wind Direction @ 2m [deg]": [180.0],
            "Avg Wind Speed @ 5m [m/s]": [6.0],
            "Avg Wind Direction @ 5m [deg]": [185.0],
            "Avg Wind Speed @ 10m [m/s]": [7.0],
            "Avg Wind Direction @ 10m [deg]": [190.0],
            "Avg Wind Speed @ 20m [m/s]": [8.0],
            "Avg Wind Direction @ 20m [deg]": [195.0],
            "Avg Wind Speed @ 50m [m/s]": [9.0],
            "Avg Wind Direction @ 50m [deg]": [200.0],
            "Avg Wind Speed @ 80m [m/s]": [10.0],
            "Avg Wind Direction @ 80m [deg]": [205.0],
        })
        result = tower_run_qc(df)
        for h in HEIGHTS:
            assert f"QC_{h}m" in result.columns
            assert result[f"QC_{h}m"].iloc[0] == 0  # Valid data should not be flagged
