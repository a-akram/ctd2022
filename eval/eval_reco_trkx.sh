#!/bin/bash

# This script runs 'trkx_reco_eval.py', the trkx_from_gnn.py must
# run before this script. Better way is to use 'reco_to_eval.sh'.

# Max Events
maxevts=5000

if test "$1" != ""; then
  maxevts=$1
fi


# Data Directories
gnn_pred="run/gnn_evaluation/test"
reco_tracks="run/trkx_from_gnn"
outputdir="run/trkx_reco_eval/eval"

# Evaluate Reco. Tracks
python eval_reco_trkx.py \
    --reco-tracks-path $reco_tracks \
    --raw-tracks-path $gnn_pred \
    --outname $outputdir \
    --max-evts $maxevts \
    --force \
    --min-hits-truth 7 \
    --min-hits-reco 4 \
    --min-pt 0. \
    --frac-reco-matched 0.5 \
    --frac-truth-matched 0.5
