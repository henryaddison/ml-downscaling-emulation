import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import metpy.plots.ctables

cp_model_rotated_pole = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
platecarree = ccrs.PlateCarree()

# precip_clevs = [0, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40,
#      50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 750, 1000]
# precip_norm, precip_cmap = metpy.plots.ctables.registry.get_with_boundaries('precipitation', precip_clevs)
precip_clevs = [0, 0.1, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200]
precip_cmap = matplotlib.colors.ListedColormap(
    metpy.plots.ctables.colortables["precipitation"][: len(precip_clevs) - 1],
    "precipitation",
)
precip_norm = matplotlib.colors.BoundaryNorm(precip_clevs, precip_cmap.N)

STYLES = {
    "precip": {"cmap": precip_cmap, "norm": precip_norm},
    "logBlues": {"cmap": "Blues", "norm": matplotlib.colors.LogNorm()},
}


def create_map_fig(grid_spec, width=None, height=None):
    if width is None:
        width = len(grid_spec[0]) * 5.5
    if height is None:
        height = len(grid_spec) * 5.5
    subplot_kw = dict(projection=cp_model_rotated_pole)
    return plt.subplot_mosaic(
        grid_spec,
        figsize=(width, height),
        subplot_kw=subplot_kw,
        constrained_layout=True,
    )


def plot_map(da, ax, title="", style="logBlues", add_colorbar=False, **kwargs):
    if style is not None:
        kwargs = STYLES[style] | kwargs
    da.plot.pcolormesh(ax=ax, add_colorbar=add_colorbar, **kwargs)
    ax.set_title(title)
    ax.coastlines()
    # ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, x_inline=False, y_inline=False)#, xlabel_style=dict(fontsize=24), ylabel_style=dict(fontsize=24))


def qq_plot(
    ax,
    x,
    ys,
    quantiles,
    title="Sample vs Target quantiles",
    xlabel="Target precip (mm day-1)",
    ylabel="Sample precip (mm day-1)",
):
    x_quantiles = x.quantile(quantiles)
    ideal_tr = max(
        x.max().values, *[y[1].max().values for y in ys]
    )  # max(target_quantiles.max().values+10, pred_quantiles.max().values+10)
    ideal_tr = ideal_tr + 0.1 * abs(ideal_tr)
    ideal_bl = (
        x_quantiles.min().values
    )  # max(target_quantiles.max().values+10, pred_quantiles.max().values+10)
    ideal_bl = ideal_bl - 0.1 * abs(
        ideal_bl
    )  # max(target_quantiles.max().values+10, pred_quantiles.max().values+10)
    ax.plot(
        [ideal_bl, ideal_tr],
        [ideal_bl, ideal_tr],
        color="black",
        linestyle="--",
        label="Ideal",
    )
    for (label, y) in ys:
        y_quantiles = y.quantile(quantiles)
        ax.scatter(x_quantiles, y_quantiles, label=label, alpha=0.8, marker="x")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_aspect(aspect=1)
