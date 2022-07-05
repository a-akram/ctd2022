#!/usr/bin/env python

"""Plotting utilities"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import itertools

fontsize = 16
minor_size = 14

# pt params
# pt_bins = (np.arange(0, 5., step=0.5).tolist() + np.arange(5, 11, step=1.0).tolist())
# pt_bins = np.arange(0.1, 1.5, step=0.1) # prob: last value isn't included
pt_bins = np.linspace(0.1, 1.5, num=15)   # will give 15 bins

pt_configs = {
    'bins': pt_bins,
    'histtype': 'step',
    'lw': 2,
    'log': False
}

# eta params
eta_bins = np.arange(-4, 4.4, step=0.4)

eta_configs = {
    'bins': eta_bins,
    'histtype': 'step',
    'lw': 2,
    'log': False
}


# get a plot
def get_plot(nrows=1, ncols=1, figsize=8, nominor=False):

    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(figsize*ncols, figsize*nrows),
                            constrained_layout=True)

    def format_axis(axis):
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())
        return axis

    if nrows * ncols == 1:
        ax = axs
        if not nominor:
            format_axis(ax)
    else:
        ax = [format_axis(x) if not nominor else x for x in axs.flatten()]

    return fig, ax


# add up x-axis
def add_up_xaxis(ax):
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(["" for x in ax.get_xticks()])
    ax2.xaxis.set_minor_locator(AutoMinorLocator())


# Get ratio
def get_ratio(x_vals, y_vals):
    res = [x/y if y != 0 else 0.0 for x, y in zip(x_vals, y_vals)]
    err = [x/y * math.sqrt((x+y)/(x*y)) if y != 0 and x != 0 else 0.0 for x, y in zip(x_vals, y_vals)]
    return res, err


# Get pairwise
def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


# Add mean and std
def add_mean_std(array, x, y, ax, color='k', dy=0.3, digits=2, fontsize=12, with_std=True):
    this_mean, this_std = np.mean(array), np.std(array)
    ax.text(x, y, "Mean: {0:.{1}f}".format(this_mean, digits), color=color, fontsize=fontsize)
    if with_std:
        ax.text(x, y-dy, "Standard Deviation: {0:.{1}f}".format(this_std, digits),
                color=color, fontsize=12)


# Make Plot
def make_cmp_plot(arrays, legends, configs,
                  xlabel, ylabel, ratio_label,
                  ratio_legends, outname, ymin):

    bins = 0.
    vals_list = []

    # make a plot
    fig, ax = get_plot()
    for array, legend in zip(arrays, legends):
        vals, bins, _ = ax.hist(array, **configs, label=legend)
        vals_list.append(vals)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    add_up_xaxis(ax)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(False)
    fig.savefig("{}.png".format(outname))

    # make a ratio plot
    fig, ax = get_plot()
    xvals = [0.5*(x[1]+x[0]) for x in pairwise(bins)]
    xerrs = [0.5*(x[1]-x[0]) for x in pairwise(bins)]

    for idx in range(1, len(arrays)):
        ratio, ratio_err = get_ratio(vals_list[-1], vals_list[idx-1])
        label = None if ratio_legends is None else ratio_legends[idx-1]
        ax.errorbar(xvals, ratio, yerr=ratio_err, fmt='o',
                    xerr=xerrs, lw=2, label=label)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ratio_label, fontsize=fontsize)
    ax.set_ylim([0., 1.])
    add_up_xaxis(ax)

    if ratio_legends is not None:
        ax.legend(loc='lower right', fontsize=12)

    ax.grid(False)
    fig.savefig("{}_ratio.png".format(outname))
