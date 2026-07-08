"""Reusable plotting helpers for region-keyed (spatial) data.

This module holds standalone, dependency-light plotting utilities that are not tied to
the laser-init pipeline, so they can be reused directly and are straightforward to lift
into `laser.generic` later. The headline helper is [`choropleth`][laser.init.plotting.choropleth],
which renders a choropleth from either a column already present on a GeoDataFrame or an
arbitrary array of user-supplied values aligned to the rows (e.g. per-node model output).
"""

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

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
            raise ValueError(
                f"values has {len(values)} entries but data has {len(gdf)} regions; "
                "they must align one-to-one."
            )
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
        missing_kwds={"color": missing_color, "edgecolor": edgecolor, "label": "No data"},
    )
    ax.set_title(title if title is not None else f"{column}{' (log10)' if log else ''}")
    ax.axis("off")

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        # logger.info(f"choropleth(): figure written to {output_path}")

    return fig
