"""Tests for data import/export functions."""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np

from src.data_file import import_data, export_data


class TestImportData:
    """Tests for the import_data function."""

    def test_import_existing_csv(self):
        """Test importing an existing CSV file."""
        df = import_data("data/radar.csv")
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_import_nonexistent_file(self):
        """Test importing a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            import_data("data/nonexistent.csv")

    def test_import_parses_datetime(self):
        """Test that Time column is parsed as datetime."""
        df = import_data("data/radar.csv")
        assert "Time" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["Time"])

    def test_import_tower_timestamp(self):
        """Test that tower timestamp column is parsed as datetime."""
        df = import_data("data/tower.csv")
        assert "timestamp" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


class TestExportData:
    """Tests for the export_data function."""

    def setup_method(self):
        """Create a temporary directory for test outputs."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary files."""
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_export_dataframe(self):
        """Test exporting a DataFrame to CSV."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        output_path = os.path.join(self.temp_dir, "test_output.csv")
        export_data(df, output_path)
        assert os.path.exists(output_path)

    def test_export_and_reimport(self):
        """Test that exported data can be re-imported correctly."""
        df = pd.DataFrame({"x": [10, 20, 30], "y": ["a", "b", "c"]})
        output_path = os.path.join(self.temp_dir, "roundtrip.csv")
        export_data(df, output_path)

        df_loaded = import_data(output_path)
        assert df_loaded is not None
        assert len(df_loaded) == len(df)
        assert list(df_loaded.columns) == list(df.columns)

    def test_export_without_index(self):
        """Test that exported CSV does not include index column."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        output_path = os.path.join(self.temp_dir, "no_index.csv")
        export_data(df, output_path)

        # Read raw file to check no index column
        with open(output_path, "r") as f:
            header = f.readline().strip()
        assert header == "col1,col2"
