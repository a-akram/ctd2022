#!/bin/bash

# Combined run of trkx_from_gnn.py (trkx_from_gnn.sh)
# and eval_reco_trkx.py (eval_reco_trkx.sh) scripts.


# params
maxevts=5000
epsilon=0.25
edge_score_cut=0.0

# input
if test "$1" != ""; then
  maxevts=$1
fi

if test "$2" != ""; then
  epsilon=$2
fi

if test "$3" != ""; then
  edge_score_cut=$3
fi


# Data Directories
inputdir="run/gnn_evaluation/test"
reco_trkx_dir="run/trkx_from_gnn"
mkdir -p $reco_trkx_dir

# Tracks from GNN
python trkx_from_gnn.py \
    --input-dir $inputdir \
    --output-dir $reco_trkx_dir \
    --max-evts $maxevts \
    --num-workers 8 \
    --score-name "score" \
    --edge-score-cut $edge_score_cut \
    --epsilon $epsilon \
    --min-samples 2


# Data Directories
eval_trkx_dir="run/trkx_reco_eval/eval"
mkdir -p $eval_trkx_dir

# Evaluate Reco. Tracks
python eval_reco_trkx.py \
    --reco-tracks-path $reco_trkx_dir \
    --raw-tracks-path $inputdir \
    --outname $eval_trkx_dir \
    --max-evts $maxevts \
    --force \
    --min-hits-truth 7 \
    --min-hits-reco 4 \
    --min-pt 0. \
    --frac-reco-matched 0.5 \
    --frac-truth-matched 0.5
