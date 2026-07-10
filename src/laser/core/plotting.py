"""Reusable plotting helpers for region-keyed (spatial) data.

Two headline helpers, both operating on a `GeoDataFrame` (or a path to a vector file
readable by `geopandas.read_file`) with an aligned array of per-region values:

- [`choropleth`][laser.core.plotting.choropleth] — a static single-panel map. Values
  come from either a column already present on the GeoDataFrame (pass ``column``) or an
  array of user-supplied values in row order (pass ``values``), so per-node model output
  can be mapped without first materializing it back onto the geometry file. Supports
  log-scaling for wide dynamic ranges, explicit ``vmin`` / ``vmax`` for shared scales
  across multiple calls, and composition into subplot grids via ``ax=``.
- [`animate_choropleth`][laser.core.plotting.animate_choropleth] — the same map, animated
  over a sequence of per-region value frames (shape ``(nframes, nregions)``) with a
  **shared colorbar** across every frame so absolute colors are directly comparable.
  Output format is inferred from the extension of ``output_path``:

  | Extension                       | Writer                                  | External deps |
  | ------------------------------- | --------------------------------------- | ------------- |
  | ``.gif``                        | `matplotlib.animation.PillowWriter`     | Pillow — already a hard dependency of matplotlib, no extra install. |
  | ``.html`` / ``.htm``            | HTML5 + JavaScript via `to_jshtml()`    | None (matplotlib only). |
  | ``.mp4`` / ``.webm`` / ``.mov`` | `matplotlib.animation.FFMpegWriter`     | `ffmpeg` binary on `PATH`. |

  MP4 is the only path that requires a non-Python dependency; the helper checks for the
  binary at save time and raises a `RuntimeError` with `brew` / `apt-get` / `conda`
  install hints if it isn't found. If no ``output_path`` is given, the returned
  `FuncAnimation` can be embedded inline in a Jupyter notebook with
  `IPython.display.HTML(anim.to_jshtml())`.

This module is intentionally dependency-light (matplotlib + geopandas + numpy — all
already required by `laser.core`) so it is straightforward to lift into
`laser.generic` later. See ``examples/choropleth_examples.ipynb`` for runnable
end-to-end demonstrations of every path above.
"""

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# from .logger import logger


def choropleth(
    data: gpd.GeoDataFrame | Path | str,
    column: str | None = None,
    *,
    values: Any = None,
    log: bool = False,
    title: str | None = None,
    legend: bool = True,
    legend_label: str | None = None,
    cmap: str = "viridis",
    edgecolor: str = "black",
    missing_color: str = "lightgrey",
    figsize: tuple[float, float] = (10, 8),
    ax: plt.Axes | None = None,
    layer: str | None = None,
    output_path: Path | str | None = None,
    dpi: int = 200,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """Render a choropleth map of one quantity over administrative regions.

    The quantity to map can come from either an existing column on ``data`` (pass
    ``column``) or an array of values aligned to the rows (pass ``values``) — the latter
    makes the function useful for arbitrary user data such as per-node model output that
    is not stored in the GeoPackage.

    Args:
        data: A GeoDataFrame, or a path to a vector file (e.g. a GeoPackage) that will be
            read with [`geopandas.read_file`][]. The input GeoDataFrame is never mutated;
            a copy is made internally.
        column: Name of the column to plot. Required when ``values`` is not given. When
            ``values`` is given, this is used as the name for the new column (default
            ``"value"``).
        values: Optional array-like of values to plot, one per row of ``data`` (in row
            order). When provided, it is assigned to ``column`` (or ``"value"``) before
            plotting.
        log: If True, plot ``log10(1 + value)`` instead of the raw value. Useful for
            quantities spanning orders of magnitude (e.g. population). The colorbar/legend
            then shows log10 units.
        title: Plot title. Defaults to the column name (with a "(log10)" suffix when
            ``log`` is True).
        legend: Whether to draw the colorbar legend (default True).
        legend_label: Optional label for the colorbar.
        cmap: Matplotlib colormap name (default ``"viridis"``).
        edgecolor: Color for region boundaries (default ``"black"``).
        missing_color: Fill color for regions whose value is missing/NaN
            (default ``"lightgrey"``).
        figsize: Figure size in inches; used only when ``ax`` is not supplied.
        ax: Optional existing Matplotlib Axes to draw into. When given, no new figure is
            created and that Axes' figure is returned.
        layer: Optional layer name passed to [`geopandas.read_file`][] when ``data`` is a
            path to a multi-layer file.
        output_path: If given, the figure is saved to this path (PNG inferred from the
            extension) at ``dpi`` resolution.
        dpi: Resolution for the saved figure (default 200).
        vmin: Lower end of the color scale. Default None (matplotlib autoscales from the
            plotted values). Explicit ``vmin`` / ``vmax`` are what let a caller keep the
            color scale fixed across multiple calls (e.g. from
            [`animate_choropleth`][laser.core.plotting.animate_choropleth] or from a shared
            subplot grid).
        vmax: Upper end of the color scale. Default None.

    Returns:
        The Matplotlib Figure containing the choropleth.

    Raises:
        RuntimeError: If ``data`` is a path that cannot be read.
        ValueError: If neither ``column`` nor ``values`` is given, or if ``values`` does
            not have one entry per row of ``data``.
        KeyError: If ``column`` is given (without ``values``) but is not a column of
            ``data``.

    Example:
        ```python
        # Map an existing column on the GeoPackage, log-scaled.
        choropleth("NGA/2010/shapes.gpkg", column="population", log=True)

        # Map arbitrary per-region model output (one value per row).
        choropleth(gdf, values=attack_rate_by_node, column="attack_rate",
                   legend_label="attack rate", cmap="magma")
        ```
    """
    # Load (without mutating the caller's GeoDataFrame).
    if isinstance(data, gpd.GeoDataFrame):
        gdf = data.copy()
    else:
        try:
            gdf = gpd.read_file(data, layer=layer)
        except Exception as e:
            raise RuntimeError(f"Failed to read vector data from {data}: {e}") from e
    # logger.info(f"choropleth(): {len(gdf)} regions loaded")

    # Resolve the column to plot, attaching user-supplied values if given.
    if values is not None:
        column = column or "value"
        values = np.asarray(values)
        if len(values) != len(gdf):
            raise ValueError(f"values has {len(values)} entries but data has {len(gdf)} regions; " "they must align one-to-one.")
        gdf[column] = values
    elif column is None:
        raise ValueError("Provide either 'column' (an existing column) or 'values'.")
    elif column not in gdf.columns:
        numeric = [c for c in gdf.columns if c != gdf.geometry.name]
        raise KeyError(f"Column '{column}' not found. Available columns: {sorted(numeric)}.")

    # Optionally log-transform for display (log10(1 + value) tolerates zeros).
    if log:
        plot_column = f"_{column}_log10"
        gdf[plot_column] = np.log1p(gdf[column]) / np.log(10)
    else:
        plot_column = column

    # Figure / axes.
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    legend_kwds = {"label": legend_label} if legend_label else None
    gdf.plot(
        column=plot_column,
        ax=ax,
        legend=legend,
        legend_kwds=legend_kwds,
        cmap=cmap,
        edgecolor=edgecolor,
        vmin=vmin,
        vmax=vmax,
        missing_kwds={"color": missing_color, "edgecolor": edgecolor, "label": "No data"},
    )
    ax.set_title(title if title is not None else f"{column}{' (log10)' if log else ''}")
    ax.axis("off")

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        # logger.info(f"choropleth(): figure written to {output_path}")

    return fig


def animate_choropleth(
    data: gpd.GeoDataFrame | Path | str,
    values_by_frame: Any,
    *,
    column: str = "value",
    frame_titles: list[str] | None = None,
    log: bool = False,
    legend: bool = True,
    legend_label: str | None = None,
    cmap: str = "viridis",
    edgecolor: str = "black",
    missing_color: str = "lightgrey",
    figsize: tuple[float, float] = (10, 8),
    vmin: float | None = None,
    vmax: float | None = None,
    fps: int = 5,
    output_path: Path | str | None = None,
    dpi: int = 150,
    layer: str | None = None,
) -> animation.FuncAnimation:
    """Build an animated choropleth over a sequence of per-region value frames.

    Renders one frame per row of ``values_by_frame`` on the same regions. The colorbar is
    shared across all frames (either user-supplied ``vmin`` / ``vmax``, or the global
    min/max across every frame) so raw colors are directly comparable — the whole point
    of an animation vs. a subplot grid.

    File format is inferred from ``output_path``'s extension:

    | Extension                | Writer                                  | External deps |
    | ------------------------ | --------------------------------------- | ------------- |
    | ``.gif``                 | `matplotlib.animation.PillowWriter`     | Pillow (already a hard dep of matplotlib) |
    | ``.html`` / ``.htm``     | HTML5 + JavaScript via `to_jshtml()`    | None (matplotlib only) |
    | ``.mp4`` / ``.webm`` / ``.mov`` | `matplotlib.animation.FFMpegWriter` | `ffmpeg` binary on PATH |

    If ``output_path`` is ``None``, no file is written and the returned ``FuncAnimation``
    can be embedded in a Jupyter notebook with ``IPython.display.HTML(anim.to_jshtml())``.

    Args:
        data: A GeoDataFrame, or a path to a vector file (e.g. a GeoPackage) that will be
            read with [`geopandas.read_file`][]. The input GeoDataFrame is never mutated;
            a copy is made internally for each frame.
        values_by_frame: 2-D array-like of shape ``(nframes, nregions)`` (or a sequence of
            1-D arrays of length ``nregions``). Each row is one animation frame's per-region
            values, in row order matching ``data``.
        column: Name to attach the per-frame values under before plotting (default
            ``"value"``). Only used for the internal column name and as the default panel
            title.
        frame_titles: Optional list of titles, one per frame. Defaults to ``"frame k"``
            for each ``k``. Must have the same length as ``values_by_frame`` if given.
        log: If True, plot ``log10(1 + value)`` instead of the raw value in each frame.
        legend: Whether to draw the shared colorbar (default True).
        legend_label: Optional label for the colorbar.
        cmap: Matplotlib colormap name (default ``"viridis"``).
        edgecolor: Color for region boundaries (default ``"black"``).
        missing_color: Fill color for regions whose value is missing/NaN.
        figsize: Figure size in inches.
        vmin: Lower end of the shared color scale. If ``None``, uses the global minimum
            across ``values_by_frame`` (post-log-transform if ``log=True``).
        vmax: Upper end of the shared color scale. If ``None``, uses the global maximum
            across ``values_by_frame`` (post-log-transform if ``log=True``).
        fps: Frames per second when saving (default 5).
        output_path: Optional destination path; format inferred from the extension. When
            ``None`` no file is written and the caller can save or embed the animation
            themselves.
        dpi: Resolution for the saved animation (default 150).
        layer: Optional layer name passed to [`geopandas.read_file`][] when ``data`` is a
            path to a multi-layer file.

    Returns:
        `matplotlib.animation.FuncAnimation`: The animation object. Save with
        ``anim.save(path, writer=...)`` for custom writers, or embed in a notebook with
        ``IPython.display.HTML(anim.to_jshtml())``.

    Raises:
        RuntimeError: If ``data`` is a path that cannot be read, or if an ``.mp4`` /
            ``.webm`` / ``.mov`` output is requested but ``ffmpeg`` is not on PATH
            (message points at the installation steps).
        ValueError: If ``values_by_frame`` is empty, if any frame's length disagrees with
            the number of regions in ``data``, if ``frame_titles`` length disagrees with
            the number of frames, or if ``fps <= 0``.

    Example:
        ```python
        # GIF (no external tool needed)
        animate_choropleth(gdf, per_frame_values, output_path="prevalence.gif", fps=10)

        # Embedded in a Jupyter notebook (no external tool needed)
        from IPython.display import HTML
        anim = animate_choropleth(gdf, per_frame_values)
        HTML(anim.to_jshtml())

        # MP4 (requires ffmpeg on PATH)
        animate_choropleth(gdf, per_frame_values, output_path="prevalence.mp4", fps=30)
        ```
    """

    # Load data (without mutating the caller's GeoDataFrame).
    if isinstance(data, gpd.GeoDataFrame):
        gdf = data.copy()
    else:
        try:
            gdf = gpd.read_file(data, layer=layer)
        except Exception as e:
            raise RuntimeError(f"Failed to read vector data from {data}: {e}") from e

    # Normalize frames to a (nframes, nregions) 2-D NumPy array.
    frames = np.asarray(values_by_frame)
    if frames.ndim == 1:
        # A caller who hands in a single 1-D vector meant a one-frame animation.
        frames = frames[np.newaxis, :]
    if frames.ndim != 2:
        raise ValueError(f"values_by_frame must be 2-D (nframes, nregions), got shape {frames.shape}")
    nframes, nregions = frames.shape
    if nframes == 0:
        raise ValueError("values_by_frame must contain at least one frame")
    if nregions != len(gdf):
        raise ValueError(
            f"values_by_frame has {nregions} regions per frame but data has {len(gdf)} regions; " "they must align one-to-one."
        )
    if frame_titles is not None and len(frame_titles) != nframes:
        raise ValueError(f"frame_titles length ({len(frame_titles)}) must match number of frames ({nframes})")
    # fps <= 0 would produce a nonsensical frame interval (division-by-zero for fps=0,
    # negative interval for fps<0) and every writer (Pillow / to_jshtml / FFMpeg) would
    # either reject it downstream or silently emit garbage. Reject at the API boundary.
    if fps <= 0:
        raise ValueError(f"fps must be a positive integer, got {fps}")

    # Compute the shared color scale (post-log-transform if requested) so every frame
    # renders on the same numeric range and colors are directly comparable across frames.
    display_frames = np.log1p(frames) / np.log(10) if log else frames
    if vmin is None:
        vmin = float(np.nanmin(display_frames))
    if vmax is None:
        vmax = float(np.nanmax(display_frames))

    # Build the figure once and re-draw into the same Axes each frame — simpler and more
    # reliable than trying to mutate the polygon collection in place across frames.
    # The colorbar is managed *outside* the per-frame draw: if we let each frame add its
    # own colorbar via gdf.plot(legend=True), they stack up (one new colorbar per frame)
    # and dominate the figure. Instead build it once from a ScalarMappable tied to the
    # shared [vmin, vmax] range, and turn the per-frame legend off inside choropleth.
    fig, ax = plt.subplots(figsize=figsize)
    if legend:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm, ax=ax, label=legend_label)

    def draw(idx):
        ax.clear()
        choropleth(
            gdf,
            values=frames[idx],
            column=column,
            log=log,
            title=(frame_titles[idx] if frame_titles is not None else f"frame {idx}"),
            legend=False,  # colorbar is drawn once above the animation, not per-frame
            cmap=cmap,
            edgecolor=edgecolor,
            missing_color=missing_color,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        return ()

    anim = animation.FuncAnimation(fig, draw, frames=nframes, interval=int(1000 / fps), blit=False)

    # Draw the first frame AND mark the animation as rendered so matplotlib's
    # Animation.__del__ doesn't fire the "Animation was deleted without rendering
    # anything" UserWarning when a caller holds the returned object briefly and
    # then lets it get garbage collected (e.g. embedding via `HTML(to_jshtml())`
    # in a notebook, or the test suite closing figures between tests).
    # ``_draw_was_started`` is a private matplotlib attribute but it's the exact
    # flag matplotlib checks in __del__; setting it is the documented escape
    # hatch for helpers that hand ownership of the animation back to the caller.
    draw(0)
    anim._draw_was_started = True  # — see comment above

    if output_path is not None:
        _save_animation(anim, Path(output_path), fps=fps, dpi=dpi)

    return anim


def _save_animation(anim: animation.FuncAnimation, output_path: Path, *, fps: int, dpi: int) -> None:
    """Save a FuncAnimation to disk, choosing the writer from the file extension.

    | Extension                       | Writer                     |
    | ------------------------------- | -------------------------- |
    | ``.gif``                        | `PillowWriter`             |
    | ``.html`` / ``.htm``            | `to_jshtml()` (raw HTML)   |
    | ``.mp4`` / ``.webm`` / ``.mov`` | `FFMpegWriter`             |

    Isolated in its own function so tests can exercise the writer-selection logic
    without spinning up a full animation each time.
    """
    from matplotlib import animation

    ext = output_path.suffix.lower()

    if ext == ".gif":
        anim.save(str(output_path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    elif ext in {".html", ".htm"}:
        # Pure Python; no external writer needed. Embed the JS-driven animation as
        # a standalone HTML document.
        output_path.write_text(anim.to_jshtml(fps=fps), encoding="utf-8")
    elif ext in {".mp4", ".webm", ".mov"}:
        if not animation.FFMpegWriter.isAvailable():
            raise RuntimeError(
                "MP4/WebM/MOV output requires the `ffmpeg` binary on PATH, which was not found. "
                "Install it separately (macOS: `brew install ffmpeg`; Debian/Ubuntu: "
                "`sudo apt-get install ffmpeg`; conda: `conda install -c conda-forge ffmpeg`), "
                "then retry. Or save to `.gif` (no external tool needed) instead."
            )
        anim.save(str(output_path), writer=animation.FFMpegWriter(fps=fps), dpi=dpi)
    else:
        raise ValueError(
            f"Unsupported animation output extension {ext!r}. " "Use .gif (Pillow), .html/.htm (JS animation), or .mp4/.webm/.mov (ffmpeg)."
        )
