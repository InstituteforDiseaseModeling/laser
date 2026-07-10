"""Tests for laser.core.plotting.

These tests exercise the standalone ``choropleth`` helper, which is intended to be
reusable (and eventually portable into laser.generic). Failures here mean the choropleth
helper has regressed in a way that would also break the CLI's population map (which now
delegates to this function) or any downstream caller mapping their own per-region data.
"""

import re

import geopandas as gpd
import matplotlib

# headless backend so tests never try to open a window
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

from laser.core.plotting import animate_choropleth  # noqa: E402
from laser.core.plotting import choropleth  # noqa: E402


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


class TestAnimateChoropleth:
    """Test suite for animate_choropleth."""

    def test_returns_funcanimation_with_correct_frame_count(self):
        """Given a GeoDataFrame and a (nframes, nregions) values array,
        when animate_choropleth() is called with no output_path,
        then it returns a matplotlib FuncAnimation with the expected number of frames.

        Baseline sanity: the animation object round-trips its frame count and can be
        used with any downstream writer (Pillow, JS-HTML, ffmpeg).
        """
        from matplotlib.animation import FuncAnimation

        gdf = _square_regions([10, 20, 30])
        frames = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]], dtype=float)

        anim = animate_choropleth(gdf, frames)

        assert isinstance(anim, FuncAnimation)
        assert anim._save_count == 4, f"Expected 4 frames, got {anim._save_count}"
        plt.close(anim._fig)

    def test_shared_color_scale_from_global_min_max(self):
        """Given frames whose values span [0, 100],
        when animate_choropleth() is called without vmin/vmax,
        then the shared color scale is exactly [0, 100] on every frame.

        Pins the "one shared colorbar" contract that motivates animate_choropleth over
        the subplot-grid pattern — colors must be comparable across frames.
        """
        gdf = _square_regions([10, 20, 30])
        frames = np.array([[0.0, 50.0, 100.0], [25.0, 75.0, 50.0]])

        anim = animate_choropleth(gdf, frames)
        # Advance the animation to draw frames; the underlying PolyCollection carries
        # the color-scale limits (norm.vmin / norm.vmax).
        anim._draw_next_frame(0, blit=False)
        collections = [c for c in anim._fig.axes[0].collections if getattr(c, "norm", None) is not None]
        assert collections, "expected at least one drawn collection with a color norm"
        assert collections[0].norm.vmin == 0.0
        assert collections[0].norm.vmax == 100.0

        plt.close(anim._fig)

    def test_explicit_vmin_vmax_override_global(self):
        """Given ``vmin=0`` and ``vmax=10`` passed explicitly,
        when animate_choropleth() is called on values that span [0, 100],
        then the color scale uses [0, 10] (values above vmax will clip at the top color).

        User-supplied vmin/vmax must beat the auto-computed global range so callers can
        force a domain-meaningful scale (e.g. [0, 1] for a probability).
        """
        gdf = _square_regions([10, 20, 30])
        frames = np.array([[0.0, 50.0, 100.0], [25.0, 75.0, 50.0]])

        anim = animate_choropleth(gdf, frames, vmin=0.0, vmax=10.0)
        anim._draw_next_frame(0, blit=False)
        collections = [c for c in anim._fig.axes[0].collections if getattr(c, "norm", None) is not None]
        assert collections[0].norm.vmin == 0.0
        assert collections[0].norm.vmax == 10.0

        plt.close(anim._fig)

    def test_saves_gif_via_pillow(self, tmp_path):
        """Given ``output_path`` with a ``.gif`` suffix,
        when animate_choropleth() is called,
        then a valid non-empty GIF file is written using Pillow (no external tool).

        This is the recommended path for callers who want portable animated output
        without adding ffmpeg to their dependency chain. If this ever fails we've
        regressed the "lightweight animation" contract.
        """
        gdf = _square_regions([10, 20, 30])
        frames = np.array([[1, 2, 3], [3, 2, 1], [2, 2, 2]], dtype=float)
        out = tmp_path / "test.gif"

        anim = animate_choropleth(gdf, frames, output_path=out, fps=2)

        assert out.exists()
        assert out.stat().st_size > 1000, "expected a non-trivial GIF payload"
        with out.open("rb") as fh:
            magic = fh.read(6)
        assert magic in (b"GIF89a", b"GIF87a"), f"expected GIF magic bytes, got {magic!r}"
        plt.close(anim._fig)

    def test_saves_html_via_jshtml(self, tmp_path):
        """Given ``output_path`` with a ``.html`` suffix,
        when animate_choropleth() is called,
        then a self-contained HTML file with an embedded JS-driven animation is written.

        Pins the notebook-friendly path (no external tool). HTML output is what
        ``anim.to_jshtml()`` produces — a JS animation controller plus base64-encoded
        frame PNGs — playable in any modern browser.
        """
        gdf = _square_regions([10, 20, 30])
        frames = np.array([[1, 2, 3], [3, 2, 1]], dtype=float)
        out = tmp_path / "test.html"

        anim = animate_choropleth(gdf, frames, output_path=out, fps=2)

        text = out.read_text(encoding="utf-8")
        assert "<script" in text, "expected embedded <script> for the JS animation"
        assert "base64" in text, "expected base64-encoded frame PNGs"
        plt.close(anim._fig)

    def test_mp4_without_ffmpeg_raises_helpful_error(self, tmp_path, monkeypatch):
        """Given ``output_path`` with ``.mp4`` and no ffmpeg on PATH,
        when animate_choropleth() is called,
        then it raises ``RuntimeError`` with an install-hint (brew / apt-get / conda),
        NOT a bare matplotlib "MovieWriter ffmpeg unavailable" traceback.

        Guards against a poor first-time experience for users who try MP4 without
        installing ffmpeg — the helper's job is to point at the fix, not to leak a
        matplotlib internal error.
        """
        from matplotlib import animation as mpl_animation

        # Force FFMpegWriter.isAvailable() to return False no matter what's on PATH.
        monkeypatch.setattr(mpl_animation.FFMpegWriter, "isAvailable", staticmethod(lambda: False))

        gdf = _square_regions([10, 20, 30])
        frames = np.array([[1, 2, 3], [3, 2, 1]], dtype=float)
        out = tmp_path / "test.mp4"

        with pytest.raises(RuntimeError, match=re.compile(r"ffmpeg.*PATH.*brew.*apt-get", re.DOTALL)):
            animate_choropleth(gdf, frames, output_path=out)

    def test_rejects_mismatched_region_count(self):
        """Given a values-per-frame array whose per-frame length doesn't match the number
        of regions in the GeoDataFrame,
        when animate_choropleth() is called,
        then it raises ``ValueError`` with a clear message.

        Baseline shape-validation: without it the animation would silently misalign
        values to regions.
        """
        gdf = _square_regions([10, 20, 30])  # 3 regions
        frames = np.array([[1, 2], [3, 4]], dtype=float)  # 2 regions per frame — WRONG

        with pytest.raises(ValueError, match=re.escape("must align one-to-one")):
            animate_choropleth(gdf, frames)

    def test_rejects_empty_frames(self):
        """Given a values-per-frame array with zero frames,
        when animate_choropleth() is called,
        then it raises ``ValueError``.

        A zero-frame "animation" has no meaningful vmin/vmax and no frames to write —
        catch it at the input boundary rather than at some downstream matplotlib
        internal.
        """
        gdf = _square_regions([10, 20, 30])
        frames = np.empty((0, 3), dtype=float)

        with pytest.raises(ValueError, match="at least one frame"):
            animate_choropleth(gdf, frames)

    def test_rejects_mismatched_frame_titles(self):
        """Given ``frame_titles`` of length 2 and ``values_by_frame`` of length 3,
        when animate_choropleth() is called,
        then it raises ``ValueError``.

        Otherwise a bad caller would either see a silent title truncation or an
        IndexError deep in the draw callback — neither is helpful.
        """
        gdf = _square_regions([10, 20, 30])
        frames = np.array([[1, 2, 3], [3, 2, 1], [2, 2, 2]], dtype=float)

        with pytest.raises(ValueError, match="frame_titles length"):
            animate_choropleth(gdf, frames, frame_titles=["a", "b"])

    def test_rejects_non_positive_fps(self):
        """Given ``fps <= 0``,
        when animate_choropleth() is called,
        then it raises ``ValueError`` with a message identifying the offending value.

        ``fps <= 0`` produces a nonsensical frame interval (division-by-zero for
        ``fps=0``, negative interval for ``fps<0``) and downstream writers either
        reject it or silently emit garbage. Rejecting at the API boundary keeps the
        error close to the caller's mistake.
        """
        gdf = _square_regions([10, 20, 30])
        frames = np.array([[1, 2, 3], [3, 2, 1]], dtype=float)

        with pytest.raises(ValueError, match=r"fps must be a positive integer, got 0"):
            animate_choropleth(gdf, frames, fps=0)

        with pytest.raises(ValueError, match=r"fps must be a positive integer, got -1"):
            animate_choropleth(gdf, frames, fps=-1)

        with pytest.raises(ValueError, match=r"fps must be a positive integer, got -30"):
            animate_choropleth(gdf, frames, fps=-30)

    def test_accepts_smallest_positive_fps(self):
        """Given ``fps=1`` (the smallest positive value),
        when animate_choropleth() is called,
        then it succeeds — the boundary is strictly ``fps > 0``, not ``fps >= 2``.

        Guards against the check accidentally becoming ``fps < 1`` (which would still
        reject 0 but also reject fractional values like 0.5, and, more importantly,
        might drift to ``fps <= 1`` in a future edit and lock out legitimate slow-
        animation use cases).
        """
        gdf = _square_regions([10, 20, 30])
        frames = np.array([[1, 2, 3], [3, 2, 1]], dtype=float)

        # Should not raise.
        anim = animate_choropleth(gdf, frames, fps=1)
        plt.close(anim._fig)

    def test_one_dimensional_values_treated_as_single_frame(self):
        """Given a 1-D values array (nregions,),
        when animate_choropleth() is called,
        then it produces a single-frame animation (a convenience for callers who want to
        reuse the animation machinery for a one-shot render).
        """
        from matplotlib.animation import FuncAnimation

        gdf = _square_regions([10, 20, 30])
        frames = np.array([1, 2, 3], dtype=float)  # 1-D

        anim = animate_choropleth(gdf, frames)

        assert isinstance(anim, FuncAnimation)
        assert anim._save_count == 1
        plt.close(anim._fig)


if __name__ == "__main__":
    pytest.main()
