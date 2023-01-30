import glob
import os

import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import xarray as xr

from ..plotting import (
    cp_model_rotated_pole,
    freq_density_plot,
    precip_clevs,
    precip_norm,
    precip_cmap,
    plot_map as plot_grid,
    qq_plot,
)


def open_samples_ds(
    run_name, human_name, checkpoint_id, dataset_name, split, num_samples
):
    samples_filepath_pattern = os.path.join(
        os.getenv("DERIVED_DATA"),
        "workdirs",
        run_name,
        f"samples/{checkpoint_id}",
        dataset_name,
        split,
        "predictions-*.nc",
    )
    sample_ds_list = [
        xr.open_dataset(sample_filepath)
        for sample_filepath in glob.glob(samples_filepath_pattern)[:num_samples]
    ]
    if len(sample_ds_list) == 0:
        raise RuntimeError(f"{samples_filepath_pattern} has no sample files")
    if len(sample_ds_list) < num_samples:
        raise RuntimeError(
            f"{samples_filepath_pattern} does not have {num_samples} sample files"
        )
    # concatenate the samples along a new dimension
    ds = xr.concat(sample_ds_list, dim="sample_id")
    # add a model dimension so can compare data from different ml models
    ds = ds.expand_dims(model=[human_name])
    return ds


def merge_over_runs(runs, dataset_name, split, samples_per_run):
    samples_ds = xr.merge(
        [
            open_samples_ds(
                run_name,
                human_name,
                checkpoint_id,
                dataset_name,
                split,
                num_samples=samples_per_run,
            ).isel(sample_id=range(samples_per_run))
            for run_name, checkpoint_id, human_name in runs
        ]
    )
    eval_ds = xr.open_dataset(
        os.path.join(
            os.getenv("DERIVED_DATA"),
            "moose",
            "nc-datasets",
            dataset_name,
            f"{split}.nc",
        )
    )

    return xr.merge([samples_ds, eval_ds], join="inner")


def merge_over_sources(datasets, runs, split, samples_per_run):
    xr_datasets = []
    sources = []
    for source, dataset_name in datasets.items():
        xr_datasets.append(
            merge_over_runs(runs, dataset_name, split, samples_per_run=samples_per_run)
        )
        sources.append(source)

    return xr.concat(xr_datasets, pd.Index(sources, name="source"))


def prep_eval_data(datasets, runs, split, samples_per_run=3):
    ds = merge_over_sources(datasets, runs, split, samples_per_run=samples_per_run)

    # convert from kg m-2 s-1 (i.e. mm s-1) to mm day-1
    ds["pred_pr"] = (ds["pred_pr"] * 3600 * 24).assign_attrs({"units": "mm day-1"})
    ds["target_pr"] = (ds["target_pr"] * 3600 * 24).assign_attrs({"units": "mm day-1"})

    return ds


def show_samples(ds, timestamps):
    for ts in timestamps:

        grid_spec = [
            ["Target"]
            + [f"{model} Name"]
            + [
                f"{model} Sample {sample_idx}"
                for sample_idx in range(len(ds["sample_id"]))
            ]
            for model in ds["model"].values
        ]
        fig, axd = plt.subplot_mosaic(
            grid_spec,
            figsize=(12, 10),
            constrained_layout=True,
            subplot_kw={"projection": cp_model_rotated_pole},
        )
        fig.suptitle(f"Precip {ts}")

        ax = axd[f"Target"]
        plot_grid(
            ds.sel(time=ts).isel(model=0)["target_pr"],
            ax,
            title=f"Simulation",
            cmap=precip_cmap,
            norm=precip_norm,
            add_colorbar=False,
        )

        for model in ds["model"].values:
            ax = axd[f"{model} Name"]
            ax.text(x=0, y=0, s=model)
            ax.set_axis_off()
            for sample_idx in range(len(ds["sample_id"].values)):
                ax = axd[f"{model} Sample {sample_idx}"]
                plot_grid(
                    ds.sel(model=model, time=ts).isel(sample_id=sample_idx)["pred_pr"],
                    ax,
                    cmap=precip_cmap,
                    norm=precip_norm,
                    add_colorbar=False,
                    title=f"Sample",
                )

        ax = fig.add_axes([1.05, 0.0, 0.05, 0.95])
        cb = matplotlib.colorbar.Colorbar(ax, cmap=precip_cmap, norm=precip_norm)
        cb.ax.set_yticks(precip_clevs)
        cb.ax.set_yticklabels(precip_clevs)
        cb.ax.tick_params(axis="both", which="major")
        cb.ax.set_ylabel("Precip (mm day-1)")

        plt.show()


def distribution_figure(ds, target_pr, quantiles, figtitle, diagnostics=False):
    fig, axes = plt.subplot_mosaic(
        [["Density", "Quantiles"]], figsize=(16.5, 5.5), constrained_layout=True
    )

    ax = axes["Density"]
    freq_density_plot(ax, ds, target_pr, diagnostics=diagnostics)

    ax = axes["Quantiles"]
    qq_plot(ax, target_pr, ds, quantiles)
    plt.show()


def seasonal_distribution_figure(ds, target_pr, quantiles):
    fig, axes = plt.subplot_mosaic(
        [["Quantiles DJF", "Quantiles MAM", "Quantiles JJA", "Quantiles SON"]],
        figsize=(22, 5.5),
        constrained_layout=True,
    )
    for season, seasonal_ds in ds.groupby("time.season"):
        ax = axes[f"Quantiles {season}"]
        seasonal_target_pr = target_pr.sel(time=(target_pr["time.season"] == season))

        qq_plot(
            ax,
            seasonal_target_pr,
            seasonal_ds,
            quantiles,
            title=f"Sample vs Target {season} quantiles",
        )
    plt.show()


def scatter_plots(ds, target_pr):
    # target_pr = ds.sel(source="CPM")["target_pr"]
    for source in ds["source"].values:
        IPython.display.display_html(f"<h5>{source}</h5>", raw=True)
        pred_pr = ds.sel(source=source)["pred_pr"]
        fig, axd = plt.subplot_mosaic(
            [pred_pr["model"].values], figsize=(22, 5.5), constrained_layout=True
        )
        for model in pred_pr["model"].values:
            tr = max(ds["pred_pr"].max(), ds["target_pr"].max())

            ax = axd[model]

            ax.scatter(
                x=ds.sel(source=source, model=model)["pred_pr"],
                y=ds.sel(source=source, model=model)["target_pr"]
                .values[None, :]
                .repeat(len(ds.sel(source=source, model=model)["sample_id"]), 0),
                alpha=0.05,
            )
            ax.plot(
                [0, tr],
                [0, tr],
                linewidth=1,
                color="black",
                linestyle="--",
                label="Ideal",
            )
            ax.set_title(f"{model}")
            ax.set_aspect(aspect=1)
        plt.show()

    # fig.suptitle(figtitle, fontsize=32)


def compute_gridspec(models, target_name):
    nmodels = len(models)
    ncols = 6
    nrows = nmodels // ncols
    if nmodels % ncols != 0:
        nrows = nrows + 1
    if nrows == 1:
        ncols = nmodels
    bias_gridspec = np.pad(
        models,
        (0, ncols * nrows - nmodels),
        mode="constant",
        constant_values=".",
    ).reshape(-1, ncols)
    return np.stack([np.concatenate([[target_name], row]) for row in bias_gridspec])
    # return bias_gridspec


def plot_mean_bias(ds, target_pr):
    target_mean = target_pr.mean(dim="time")
    sample_mean = ds["pred_pr"].mean(dim=["sample_id", "time"])
    bias = sample_mean - target_mean
    bias_ratio = bias / target_mean

    vmin = min([da.min().values for da in [sample_mean, target_mean]])
    vmax = max([da.max().values for da in [sample_mean, target_mean]])

    bias_ratio_vmax = abs(bias_ratio).max().values

    for source in sample_mean["source"].values:
        IPython.display.display_html(f"<h4>{source}</h4>", raw=True)

        target_name = "$\\mu_{CPM}$"
        grid_spec = compute_gridspec(bias_ratio["model"].values, target_name)
        fig, axd = plt.subplot_mosaic(
            grid_spec,
            figsize=(grid_spec.shape[1] * 5.5, grid_spec.shape[0] * 5.5),
            subplot_kw=dict(projection=cp_model_rotated_pole),
            constrained_layout=True,
        )

        ax = axd[target_name]
        plot_grid(
            target_mean,
            ax,
            title=target_name,
            norm=None,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=True,
        )
        for model in bias_ratio["model"].values:
            ax = axd[model]
            pcm = plot_grid(
                bias_ratio.sel(source=source, model=model),
                ax,
                title=f"{model}",
                norm=None,
                cmap="BrBG",
                vmax=bias_ratio_vmax,
                center=0,
                add_colorbar=False,
            )

        axes = [axd[model] for model in bias_ratio["model"].values]
        fig.colorbar(pcm, ax=axes, location="right")

        plt.show()


def plot_std_bias(ds, target_pr):
    target_std = target_pr.std(dim="time")
    sample_std = ds["pred_pr"].std(dim=["sample_id", "time"])
    std_ratio = sample_std / target_std

    vmin = target_std.min().values
    vmax = target_std.max().values

    std_ratio_vmax = 1 + (abs(1 - std_ratio).max().values)

    for source in sample_std["source"].values:
        IPython.display.display_html(f"<h4>{source}</h4>", raw=True)

        target_name = "$\\sigma_{CPM}$"
        grid_spec = compute_gridspec(std_ratio["model"].values, target_name)
        fig, axd = plt.subplot_mosaic(
            grid_spec,
            figsize=(grid_spec.shape[1] * 5.5, grid_spec.shape[0] * 5.5),
            subplot_kw=dict(projection=cp_model_rotated_pole),
            constrained_layout=True,
        )
        ax = axd[target_name]
        plot_grid(
            target_std,
            ax,
            title=target_name,
            norm=None,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=True,
        )
        for model in std_ratio["model"].values:
            ax = axd[model]
            pcm = plot_grid(
                std_ratio.sel(source=source, model=model),
                ax,
                title=f"{model}",
                norm=None,
                cmap="BrBG",
                vmax=std_ratio_vmax,
                center=1,
            )

        axes = [axd[model] for model in std_ratio["model"].values]
        fig.colorbar(pcm, ax=axes, location="right")
        plt.show()


def psd(batch):
    # npix = batch.shape[1]
    fourier = np.fft.fftshift(np.fft.fftn(batch, axes=(1, 2)), axes=(1, 2))
    amps = np.abs(fourier) ** 2  # / npix**2
    return amps


def plot_psd(arg):
    # fig = plt.figure(figsize=(5.5, 5.5))
    for label, precip_da in arg.items():
        npix = precip_da["grid_latitude"].size
        fourier_amplitudes = psd(precip_da.values.reshape(-1, npix, npix))

        kfreq = np.fft.fftshift(np.fft.fftfreq(npix)) * npix
        kfreq2D = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
        kbins = np.arange(0.5, npix // 2 + 1, 1.0)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])

        Abins, _, _ = scipy.stats.binned_statistic(
            knrm.flatten(),
            fourier_amplitudes.reshape(-1, npix * npix),
            statistic="mean",
            bins=kbins,
        )
        mean_Abins = np.mean(Abins, axis=0)

        plt.loglog(kvals, mean_Abins, label=label)

    # plt.legend()
    plt.legend(ncols=3)
    plt.tight_layout()
    plt.subplots_adjust(right=2, bottom=-0.25)
    plt.xlabel("$k$")
    plt.ylabel("$P(k)$")
    # plt.tight_layout()

    plt.show()


def pys_plot_psd(arg):
    plt.figure(figsize=(12, 12))
    for label, precip_da in arg.items():
        npix = precip_da["grid_latitude"].size
        fourier_amplitudes = psd(precip_da.values.reshape(-1, npix, npix))

        s1 = np.s_[-int(npix / 2) : int(npix / 2)]
        s2 = np.s_[-int(npix / 2) : int(npix / 2)]
        yc, xc = np.ogrid[s1, s2]

        r_grid = np.sqrt(xc * xc + yc * yc).round()

        r_range = np.arange(0, int(npix / 2))
        freq = np.fft.fftfreq(npix) * npix
        freq = freq[r_range]

        pys_result = []
        for r in r_range:
            mask = r_grid == r
            psd_vals = fourier_amplitudes[:, mask]
            pys_result.append(np.mean(psd_vals))

        mean_Abins = np.array(pys_result)

        plt.loglog(freq, mean_Abins, label=label)

    plt.legend()
    plt.xlabel("$k$")
    plt.ylabel("$P(k)$")
    # plt.tight_layout()

    plt.show()
