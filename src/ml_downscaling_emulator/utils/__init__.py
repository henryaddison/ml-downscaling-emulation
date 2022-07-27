import os

import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import metpy.plots.ctables
import xarray as xr

cp_model_rotated_pole = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
platecarree = ccrs.PlateCarree()

# precip_clevs = [0, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40,
#      50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 750, 1000]
# precip_norm, precip_cmap = metpy.plots.ctables.registry.get_with_boundaries('precipitation', precip_clevs)
precip_clevs = [0, 0.1, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200]
precip_cmap = matplotlib.colors.ListedColormap(metpy.plots.ctables.colortables["precipitation"][:len(precip_clevs)-1], 'precipitation')
precip_norm = matplotlib.colors.BoundaryNorm(precip_clevs, precip_cmap.N)

STYLES = {
    "precip": {
        "cmap": precip_cmap,
        "norm": precip_norm
    },
    "logBlues": {
        "cmap": "Blues",
        "norm": matplotlib.colors.LogNorm()
    }
}

def plot_grid(da, ax, title="", style="logBlues", add_colorbar=False, **kwargs):
    if style is not None:
        kwargs = (STYLES[style] | kwargs)
    da.plot.pcolormesh(ax=ax, add_colorbar=add_colorbar, **kwargs)
    ax.set_title(title, fontsize=16)
    ax.coastlines()
    ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, x_inline=False, y_inline=False, xlabel_style=dict(fontsize=24), ylabel_style=dict(fontsize=24))

def open_samples_ds(run_name, checkpoint_id, dataset_name, split):
    filepaths = os.path.join(os.getenv("DERIVED_DATA"), 'score-sde/workdirs/subvpsde/xarray_cncsnpp_continuous', run_name, f'samples/checkpoint-{checkpoint_id}', dataset_name, split, 'predictions-*.nc')
    return xr.open_mfdataset(filepaths)

def show_samples(ds, timestamps, vmin, vmax):
    num_predictions = len(ds["sample_id"])

    num_plots_per_ts = num_predictions+1 # plot each sample and true target pr

    for (i, ts) in enumerate(timestamps):
        if i % 3 == 0:
            fig = plt.figure(figsize=(40, 5))
            ax = fig.add_axes([0.05, 0.80, 0.9, 0.05])
            cb = matplotlib.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=precip_cmap, norm=precip_norm)
            ax.set_xlabel("Precip (mm day-1)", fontsize=32)
            ax.set_xticks(precip_clevs)
            ax.tick_params(axis='both', which='major', labelsize=32)
            plt.show()

        fig, axes = plt.subplots(1, num_plots_per_ts, figsize=(40,40), constrained_layout=True, subplot_kw={'projection': cp_model_rotated_pole})

        ax = axes[0]
        plot_grid(ds.sel(time=ts)["target_pr"], ax, title=f"Target pr {ts}", cmap=precip_cmap, norm=precip_norm, add_colorbar=False)

        for sample_id in ds["sample_id"].values:
            ax = axes[1+sample_id]
            plot_grid(ds.sel(time=ts, sample_id=sample_id)["pred_pr"], ax, cmap=precip_cmap, norm=precip_norm, add_colorbar=False, title="Sample pr")

        plt.show()

def distribution_figure(target_pr, pred_pr, quantiles, tail_thr, extreme_thr, figtitle):

    fig, axes = plt.subplot_mosaic([["Density"]], figsize=(20, 10), constrained_layout=True)
    ax = axes["Density"]
    _, bins, _ = pred_pr.plot.hist(ax=ax, bins=50, density=True,alpha=0.75, color="red", label="Samples", log=True)
    target_pr.plot.hist(ax=ax, bins=bins, density=True,alpha=1, color="black", histtype="step", label="Target", log=True, linewidth=3, linestyle="-")
    ax.set_title("Log density plot of samples and target precipitation", fontsize=24)
    ax.set_xlabel("Precip (mm day-1)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    text = f"""
    # Timestamps: {pred_pr["time"].count().values}
    # Samples: {pred_pr.count().values}
    # Targets: {target_pr.count().values}
    % Samples == 0: {(((pred_pr == 0).sum()/pred_pr.count()).values*100).round()}
    % Targets == 0: {(((target_pr == 0).sum()/target_pr.count()).values*100).round()}
    % Samples < 1e-5: {(((pred_pr < 1e-5).sum()/pred_pr.count()).values*100).round()}
    % Targets < 1e-5: {(((target_pr < 1e-5).sum()/target_pr.count()).values*100).round()}
    % Samples < 0.1: {(((pred_pr < 0.1).sum()/pred_pr.count()).values*100).round()}
    % Targets < 0.1: {(((target_pr < 0.1).sum()/target_pr.count()).values*100).round()}
    % Samples < 1: {(((pred_pr < 1).sum()/pred_pr.count()).values*100).round()}
    % Targets < 1: {(((target_pr < 1).sum()/target_pr.count()).values*100).round()}
    Sample max: {pred_pr.max().values.round()}
    Target max: {target_pr.max().values.round()}
    """
    ax.text(0.7, 0.5, text, fontsize=16, transform=ax.transAxes)
    ax.legend(fontsize=16)

#     fig, axes = plt.subplot_mosaic([["Head density"], ["Tail density"], ["Extreme tail density"]]], figsize=(20, 20), constrained_layout=True)

#     ax = axes["Head density"]
#     thresholded_target_pr = target_pr.where(target_pr<=tail_thr)
#     thresholded_pred_pr = pred_pr.where(pred_pr<=tail_thr)
#     _, bins, _ = thresholded_pred_pr.plot.hist(ax=ax, bins=50, density=True,alpha=0.75, color="red", label="Sample")
#     thresholded_target_pr.plot.hist(ax=ax, bins=bins, density=True,alpha=1, color="blue", histtype="step", label="Target")
#     ax.set_title("Head of density plot of samples and target precipitation", fontsize=16)
#     ax.set_xlabel("Precip (mm day-1)", fontsize=16)
#     text = f"""
#     # Samples: {thresholded_pred_pr.count().values}
#     # Targets: {thresholded_target_pr.count().values}
#     """
#     ax.text(0.75, 0.50, text, fontsize=16, transform=ax.transAxes)
#     ax.legend()

#     ax = axes["Tail density"]
#     thresholded_target_pr = target_pr.where(target_pr>tail_thr)
#     thresholded_pred_pr = pred_pr.where(pred_pr>tail_thr)
#     _, bins, _ = thresholded_pred_pr.plot.hist(ax=ax, bins=50, density=True,alpha=0.75, color="red", label="Sample")
#     thresholded_target_pr.plot.hist(ax=ax, bins=bins, density=True,alpha=1, color="blue", histtype="step", label="Target")
#     ax.set_title("Tail of density plot of samples and target precipitation", fontsize=16)
#     ax.set_xlabel("Precip (mm day-1)", fontsize=16)
#     ax.tick_params(axis='both', which='major', labelsize=32)
#     text = f"""
#     # Samples: {thresholded_pred_pr.count().values}
#     # Targets: {thresholded_target_pr.count().values}
#     """
#     ax.text(0.75, 0.50, text, fontsize=16, transform=ax.transAxes)
#     ax.legend()

#     ax = axes["Extreme tail density"]
#     thresholded_target_pr = target_pr.where(target_pr>extreme_thr)
#     thresholded_pred_pr = pred_pr.where(pred_pr>extreme_thr)
#     _, bins, _ = thresholded_pred_pr.plot.hist(ax=ax, bins=50, density=True,alpha=0.75, color="red", label="Samples")
#     thresholded_target_pr.plot.hist(ax=ax, bins=bins, density=True,alpha=1, color="blue", histtype="step", label="Target", linewidth=5)
#     ax.set_title("Extreme tail of density plot of samples and target precipitation", fontsize=16)
#     ax.set_xlabel("Precip (mm day-1)", fontsize=16)
#     text = f"""
#     # Sample: {thresholded_pred_pr.count().values}
#     # Target: {thresholded_target_pr.count().values}
#     """
#     ax.text(0.75, 0.5, text, fontsize=16, transform=ax.transAxes)
#     ax.legend()

    fig, axes = plt.subplot_mosaic([["Quantiles"]], figsize=(20, 10), constrained_layout=True)
    ax = axes["Quantiles"]
    target_quantiles = target_pr.quantile(quantiles)
    pred_quantiles = pred_pr.chunk(dict(sample_id=-1)).quantile(quantiles)
    ideal_tr = max(np.max(target_quantiles), np.max(pred_quantiles))
    ax.scatter(target_quantiles, pred_quantiles, label="Computed")
    ax.plot([0,ideal_tr], [0,ideal_tr], color="orange", linestyle="--", label="Ideal")
    ax.set_xlabel("Target pr (mm day-1)", fontsize=16)
    ax.set_ylabel("Sample pr (mm day-1", fontsize=16)
    ax.set_title("Sample quantiles vs Target quantiles (90th to 99.9th centiles)", fontsize=16)
    # ax.set_xticks(target_quantiles, quantiles)
    # ax.set_yticks(pred_quantiles, quantiles)
    text = f"""

    """
    ax.text(0.75, 0.2, text, fontsize=16, transform=ax.transAxes)
    ax.legend()
    ax.set_aspect(aspect=1)

    fig, axes = plt.subplot_mosaic([["Residual"]], figsize=(20, 10), constrained_layout=True)
    ax = axes["Residual"]
    (target_pr - pred_pr).plot.hist(ax=ax, bins=100, density=True, color="brown")
    ax.set_xlabel("Precip (mm day-1)")
    ax.set_title("Density plot of residuals")

    fig.suptitle(figtitle, fontsize=32)

    plt.show()
