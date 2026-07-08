"""Tests for laser.init.plotting.

These tests exercise the standalone ``choropleth`` helper, which is intended to be
reusable (and eventually portable into laser.generic). Failures here mean the choropleth
helper has regressed in a way that would also break the CLI's population map (which now
delegates to this function) or any downstream caller mapping their own per-region data.
"""

import re

import matplotlib

matplotlib.use("Agg")  # headless backend so tests never try to open a window

import geopandas as gpd
import matplotlib.pyplot as plt
import pytest
from shapely.geometry import Polygon

from laser.core.plotting import choropleth


def _square_regions(values=None):
    """Build a tiny 3-region GeoDataFrame for choropleth tests.

    Args:
        values: Optional list of population values (one per region). Defaults to a
            spread of values so log scaling is meaningful.

    Returns:
        A GeoDataFrame with columns ``name``, ``population``, and ``geometry`` (EPSG:4326).
    """
    if values is None:
        values = [10, 1000, 100000]
    geoms = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(len(values))]
    return gpd.GeoDataFrame(
        {"name": [f"r{i}" for i in range(len(values))], "population": values, "geometry": geoms},
        crs="EPSG:4326",
    )


class TestChoropleth:
    """Test suite for the standalone choropleth helper."""

    def test_plots_existing_column_returns_figure(self):
        """Given a GeoDataFrame with a numeric column, when choropleth() is called with
        that column, then it returns a Matplotlib Figure whose axes drew a polygon
        collection.

        Failure indicates the core column-plotting path broke.
        """
        gdf = _square_regions()

        fig = choropleth(gdf, column="population")

        assert isinstance(fig, plt.Figure)
        # One PatchCollection/collection should have been added for the regions.
        assert len(fig.axes[0].collections) >= 1
        plt.close(fig)

    def test_saves_output_file(self, tmp_path):
        """Given an output_path, when choropleth() is called, then the image file is
        written to that path.

        Failure indicates the save path/dpi handling regressed, so the CLI report would
        be missing its choropleth PNG.
        """
        gdf = _square_regions()
        out = tmp_path / "map.png"

        fig = choropleth(gdf, column="population", output_path=out)

        assert out.exists(), "choropleth() should write to the specified output path."
        assert out.stat().st_size > 0, "choropleth() output should not be empty."
        plt.close(fig)

    def test_plots_user_supplied_values(self):
        """Given an array of values (and no such column), when choropleth() is called with
        values=, then it attaches them under the given column name and plots.

        Failure indicates the user-data path (e.g. mapping per-node model output) broke,
        which is the primary reason this helper is being generalized.
        """
        gdf = _square_regions()

        fig = choropleth(gdf, column="attack_rate", values=[0.1, 0.5, 0.9])

        assert isinstance(fig, plt.Figure)
        # The input must not be mutated; the new column lives only on the internal copy.
        assert "attack_rate" not in gdf.columns
        plt.close(fig)

    def test_values_length_mismatch_raises(self):
        """Given a values array whose length differs from the region count, when
        choropleth() is called, then a ValueError is raised.

        Failure indicates the alignment guard was lost, which would silently misattribute
        values to the wrong regions.
        """
        gdf = _square_regions()  # 3 regions

        values = [1, 2]
        with pytest.raises(ValueError, match=r"values has [0-9]+ entries but data has [0-9]+ regions; they must align one-to-one."):
            choropleth(gdf, column="x", values=values)

    def test_missing_column_raises_keyerror(self):
        """Given a column name not present on the data, when choropleth() is called, then a
        KeyError is raised.

        Failure indicates a typo'd column would produce an opaque error instead of a clear
        one listing the available columns.
        """
        gdf = _square_regions()

        with pytest.raises(KeyError):
            choropleth(gdf, column="does_not_exist")

    def test_requires_column_or_values(self):
        """Given neither a column nor values, when choropleth() is called, then a
        ValueError is raised.

        Failure indicates the function would attempt to plot nothing.
        """
        gdf = _square_regions()

        with pytest.raises(ValueError, match=re.escape("Provide either 'column' (an existing column) or 'values'.")):
            choropleth(gdf)

    def test_reads_from_path(self, tmp_path):
        """Given a path to a GeoPackage, when choropleth() is called, then it reads the
        file and returns a Figure.

        Failure indicates the path-loading branch (used by the CLI, which passes a
        GeoPackage path) broke.
        """
        gpkg = tmp_path / "shapes.gpkg"
        _square_regions().to_file(gpkg, driver="GPKG")

        fig = choropleth(gpkg, column="population", log=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unreadable_path_raises_runtimeerror(self, tmp_path):
        """Given a path that cannot be read as vector data, when choropleth() is called,
        then a RuntimeError is raised.

        Failure indicates read errors would surface as an opaque library exception rather
        than a clear, actionable one.
        """
        bogus = tmp_path / "not_a_geofile.gpkg"
        bogus.write_text("this is not a geopackage")

        with pytest.raises(RuntimeError):
            choropleth(bogus, column="population")

    def test_draws_into_existing_axes(self):
        """Given an existing Axes, when choropleth() is called with ax=, then it draws into
        that Axes and returns its parent Figure (without creating a new one).

        Failure indicates composition into multi-panel figures (a key reuse case) broke.
        """
        gdf = _square_regions()
        fig, ax = plt.subplots()

        returned = choropleth(gdf, column="population", ax=ax)

        assert returned is fig
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_log_does_not_mutate_input(self):
        """Given log=True (which needs a derived column), when choropleth() is called, then
        the caller's GeoDataFrame gains no extra columns.

        Failure indicates the helper mutates its input, which would surprise callers
        reusing the same GeoDataFrame across plots.
        """
        gdf = _square_regions()
        original_columns = set(gdf.columns)

        fig = choropleth(gdf, column="population", log=True)

        assert set(gdf.columns) == original_columns
        plt.close(fig)


if __name__ == "__main__":
    pytest.main()
