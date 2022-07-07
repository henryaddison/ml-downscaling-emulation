import cartopy.crs as ccrs
import matplotlib

cp_model_rotated_pole = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
platecarree = ccrs.PlateCarree()


def plot_grid(da, ax, title="", norm=matplotlib.colors.LogNorm(), cmap='Blues', add_colorbar=False, **kwargs):
    da.plot.pcolormesh(ax=ax, norm=norm, add_colorbar=add_colorbar, cmap=cmap, **kwargs)
    ax.set_title(title, fontsize=16)
    ax.coastlines()
    ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, x_inline=False, y_inline=False, xlabel_style=dict(fontsize=12), ylabel_style=dict(fontsize=12))
