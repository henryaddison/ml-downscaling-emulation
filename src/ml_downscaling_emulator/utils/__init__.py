import glob
import os

import cartopy.crs as ccrs
import IPython
import matplotlib
import matplotlib.pyplot as plt
import metpy.plots.ctables
import numpy as np
import pandas as pd
import scipy
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
    samples_filepath_pattern = os.path.join(os.getenv("DERIVED_DATA"), 'score-sde/workdirs/subvpsde/xarray_cncsnpp_continuous', run_name, f'samples/checkpoint-{checkpoint_id}', dataset_name, split, 'predictions-*.nc')
    sample_ds_list = [ xr.open_dataset(sample_filepath) for sample_filepath in glob.glob(samples_filepath_pattern) ]
    # concatenate the samples along a new dimension
    ds = xr.concat(sample_ds_list, dim="sample_id")
    # add a model dimension so can compare data from different ml models
    ds = ds.expand_dims(model=[run_name])
    return ds

def merge_over_runs(runs, dataset_name, split):
    num_samples = 3
    samples_ds = xr.merge([
        open_samples_ds(run_name, checkpoint_id, dataset_name, split).sel(sample_id=range(num_samples)) for run_name, checkpoint_id in runs
    ])
    eval_ds = xr.open_dataset(os.path.join(os.getenv("MOOSE_DERIVED_DATA"), "nc-datasets", dataset_name, f"{split}.nc"))

    return xr.merge([samples_ds, eval_ds], join="inner")

def merge_over_sources(datasets, runs, split):
    xr_datasets = []
    sources = []
    for source, dataset_name in datasets.items():
        xr_datasets.append(merge_over_runs(runs, dataset_name, split))
        sources.append(source)

    return xr.concat(xr_datasets, pd.Index(sources, name='source'))

def prep_eval_data(datasets, runs, split):
    ds = merge_over_sources(datasets, runs, split)

    # convert from kg m-2 s-1 (i.e. mm s-1) to mm day-1
    ds["pred_pr"] = (ds["pred_pr"]*3600*24 ).assign_attrs({"units": "mm day-1"})
    ds["target_pr"] = (ds["target_pr"]*3600*24).assign_attrs({"units": "mm day-1"})

    return ds

def show_samples(ds, timestamps, vmin, vmax):
    num_predictions = len(ds["sample_id"])

    num_plots_per_ts = num_predictions+1 # plot each sample and true target pr

    fig = plt.figure(figsize=(40, 5))
    ax = fig.add_axes([0.05, 0.80, 0.9, 0.05])
    cb = matplotlib.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=precip_cmap, norm=precip_norm)
    ax.set_xlabel("Precip (mm day-1)", fontsize=32)
    ax.set_xticks(precip_clevs)
    ax.tick_params(axis='both', which='major', labelsize=32)
    plt.show()

    for ts in timestamps:
        nrows = len(ds["source"])
        ncols = num_plots_per_ts
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows), constrained_layout=True, subplot_kw={'projection': cp_model_rotated_pole})

        if len(ds["source"]) == 1:
            axes = [axes]

        for source_idx, source in enumerate(ds["source"].values):
            ax = axes[source_idx][0]
            plot_grid(ds.sel(source=source, time=ts)["target_pr"], ax, title=f"{source} Target pr {ts}", cmap=precip_cmap, norm=precip_norm, add_colorbar=False)
            for sample_idx in range(len(ds["sample_id"].values)):
                ax = axes[source_idx][1+sample_idx]
                plot_grid(ds.sel(source=source, time=ts).isel(sample_id=sample_idx)["pred_pr"], ax, cmap=precip_cmap, norm=precip_norm, add_colorbar=False, title=f"{source} Sample pr")

        plt.show()

def distribution_figure(target_pr, pred_pr, quantiles, tail_thr, extreme_thr, figtitle):
    fig, axes = plt.subplot_mosaic([["Density"]], figsize=(20, 10), constrained_layout=True)
    ax = axes["Density"]

    hrange=(min(pred_pr.min().values, target_pr.min().values), max(pred_pr.max().values, target_pr.max().values))
    _, bins, _ = target_pr.plot.hist(ax=ax, bins=50, density=True,alpha=1, label="Target", log=True, range=hrange)
    for source in pred_pr["source"].values:
        for model in pred_pr["model"].values:
            pred_pr.sel(source=source, model=model).plot.hist(ax=ax, bins=bins, density=True,alpha=0.75, histtype="step", label=f"{model} {source} Samples", log=True, range=hrange, linewidth=3, linestyle="-")

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
    for source in pred_pr["source"].values:
        for model in pred_pr["model"].values:
            pred_quantiles = pred_pr.sel(source=source, model=model).chunk(dict(sample_id=-1)).quantile(quantiles)
            ax.scatter(target_quantiles, pred_quantiles, label=f"{model} {source}")

    ideal_tr = max(target_quantiles.max().values+10, pred_quantiles.max().values+10)

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

def plot_mean_bias(ds):
    target_mean = ds['target_pr'].sel(source="CPM").mean(dim="time")
    sample_mean = ds['pred_pr'].mean(dim=["sample_id", "time"])

    vmin = min([da.min().values for da in [sample_mean, target_mean]])
    vmax = max([da.max().values for da in [sample_mean, target_mean]])

    for source in sample_mean["source"].values:
        IPython.display.display_html(f"<h1>{source}</h1>", raw=True)
        for model in sample_mean["model"].values:
            IPython.display.display_html(f"<h2>{model}</h2>", raw=True)

            mean_ds = sample_mean.sel(source=source, model=model)

            bias = mean_ds - target_mean

            fig, axd = plt.subplot_mosaic([["Sample", "Target"]], figsize=(12, 6), subplot_kw=dict(projection=cp_model_rotated_pole), constrained_layout=True)

            ax = axd["Sample"]
            plot_grid(mean_ds, ax, title="Sample mean", norm=None, vmin=vmin, vmax=vmax, add_colorbar=True)

            ax = axd["Target"]
            plot_grid(target_mean, ax, title="Target pr mean", norm=None, vmin=vmin, vmax=vmax, add_colorbar=False)

            plt.show()

            fig, axd = plt.subplot_mosaic([["Bias", "Bias ratio"]], figsize=(12, 6), subplot_kw=dict(projection=cp_model_rotated_pole), constrained_layout=True)

            ax = axd["Bias"]
            plot_grid(bias, ax, title="Bias", norm=None, cmap="BrBG", center=0, add_colorbar=True)

            ax = axd["Bias ratio"]
            plot_grid((bias/target_mean), ax, title="Bias/Target mean", norm=None, cmap="BrBG", center=0, add_colorbar=True)

            plt.show()

def plot_std(ds):
    target_std = ds['target_pr'].sel(source="CPM").std(dim="time")
    sample_std = ds['pred_pr'].std(dim=["sample_id", "time"])

    vmin = min([da.min().values for da in [sample_std, target_std]])
    vmax = max([da.max().values for da in [sample_std, target_std]])

    for source in sample_std["source"].values:
        IPython.display.display_html(f"<h1>{source}</h1>", raw=True)
        for model in sample_std["model"].values:
            IPython.display.display_html(f"<h2>{model}</h2>", raw=True)

            std_ds = sample_std.sel(source=source, model=model)

            fig, axs = plt.subplots(1, 3, figsize=(20, 6), subplot_kw=dict(projection=cp_model_rotated_pole))

            ax = axs[0]
            plot_grid(std_ds, ax, title="Sample std", norm=None, vmin=vmin, vmax=vmax, add_colorbar=True, cmap="viridis")

            ax = axs[1]
            plot_grid(target_std, ax, title="Target pr std", norm=None, vmin=vmin, vmax=vmax, add_colorbar=True, cmap="viridis")

            ax = axs[2]
            plot_grid((std_ds/target_std), ax, title="Sample/Target pr std", norm=None, cmap="BrBG", center=1, add_colorbar=True)

            plt.show()



def psd(batch):
    npix = batch.shape[1]
    fourier = np.fft.fftshift(np.fft.fftn(batch, axes=(1,2)), axes=(1,2))
    amps = np.abs(fourier) ** 2 #/ npix**2
    return amps

def plot_psd(arg):
    plt.figure(figsize=(12,12))
    for label, precip_da in arg.items():
        npix = precip_da["grid_latitude"].size
        fourier_amplitudes = psd(precip_da.values.reshape(-1, npix, npix))

        kfreq = np.fft.fftshift(np.fft.fftfreq(npix)) * npix
        kfreq2D = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
        kbins = np.arange(0.5, npix//2+1, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])

        Abins, _, _ = scipy.stats.binned_statistic(knrm.flatten(), fourier_amplitudes.reshape(-1, npix*npix),
                                                    statistic = "mean",
                                                    bins = kbins)
        mean_Abins = np.mean(Abins, axis=0)

        plt.loglog(kvals, mean_Abins, label=label)

    plt.legend()
    plt.xlabel("$k$")
    plt.ylabel("$P(k)$")
    # plt.tight_layout()

    plt.show()