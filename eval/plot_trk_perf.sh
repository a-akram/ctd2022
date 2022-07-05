#!/bin/bash

# This script runs 'plot_trk_perf.py'

infile="run/trkx_reco_eval/all/particles.h5"
outputdir="run/trkx_reco_eval/all"
prefix="trk_perf"

# reco tracks from GNN
python plot_trk_perf.py -f $infile -o $outputdir -p $prefix
