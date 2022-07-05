#!/usr/bin/env python

"""Plot track efficiency and purity"""

import os
import pandas as pd
from functools import partial
from utils_plot import make_cmp_plot, pt_configs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot tracking performance")
    add_arg = parser.add_argument
    add_arg("-f", "--fname", help="input file from eval_rec_trkx", required=True)
    add_arg("-o", "--outdir", help='output directory that saves outputs', required=True)
    add_arg('-p', '--prefix', default='out', help='prefix of output names')

    args = parser.parse_args()
    if not os.path.exists(args.fname):
        print("{} does not exists".format(args.fname))
        exit(1)

    outdir = args.outdir
    os.makedirs(os.path.abspath(outdir), exist_ok=True)
    out_prefix = args.prefix

    with pd.HDFStore(args.fname, 'r') as f:
        df = f['data']

    pt = df.pt.values
    rectable_idx = df.is_trackable
    matched_idx = df.is_matched
    
    # TODO: I dn't have the following vars in eval_particles.npz
    # eta = df.eta.values
    # vx = df.vx.values
    # vy = df.vy.values
    # d0 = np.sqrt(vx ** 2 + vy ** 2)
    # z0 = df.vz.values

    # plot the efficiency as a function of pT, eta
    make_cmp_plot_fn = partial(make_cmp_plot,
                               legends=["Generated", "Reconstructable", "Matched"],
                               ylabel="Events", ratio_label='Track Efficiency',
                               ratio_legends=["Physics Eff.", "Technical Eff."])
    
    print("pt_bins: ", pt_configs['bins'])
    print("n_bins: ", len(pt_configs['bins']))
    
    # fiducial cuts: pT > 1 GeV and |eta| < 4
    # all_cuts = [0., 0.1, 0.25]
    all_cuts = [0.]
    for cut_pt in all_cuts:
        cuts = (pt > cut_pt)
        gen_pt = pt[cuts]
        true_pt = pt[cuts & rectable_idx]
        reco_pt = pt[cuts & rectable_idx & matched_idx]

        # make pT plots
        make_cmp_plot_fn([gen_pt, true_pt, reco_pt],
                         configs=pt_configs,
                         xlabel=r"$p_t$ [GeV]",
                         outname=os.path.join(outdir, "{}_pt_cut{}".format(out_prefix, cut_pt)),
                         ymin=0.6)
