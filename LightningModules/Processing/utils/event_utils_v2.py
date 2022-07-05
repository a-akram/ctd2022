#!/usr/bin/env python
# coding: utf-8

"""
Utilities for Processing the Overall Event:

The module contains useful functions for handling the data at the event level. 
More fine-grained utilities are reserved for `detector_utils` and `cell_utils`.
"""

# TODO: Pull module IDs out into a csv file for readability

import os
import logging
import itertools

import numpy as np
import pandas as pd
import trackml.dataset

import torch
from torch_geometric.data import Data

from .graph_utils import get_input_edges, graph_intersection

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_layerwise_edges(hits):
    """Build Layerwise True Edges i.e. the True Graph. Here `hits` represent complete event."""
    
    # ADAK: Sort by increasing distance from production (IP)
    hits = hits.assign(
        R=np.sqrt(
            (hits.x - hits.vx) ** 2 + (hits.y - hits.vy) ** 2 + (hits.z - hits.vz) ** 2
        )
    )
    hits = hits.sort_values("R").reset_index(drop=True).reset_index(drop=False)
    hits.loc[hits["particle_id"] == 0, "particle_id"] = np.nan
    hit_list = (
        hits.groupby(["particle_id", "layer_id"], sort=False)["index"]  # ADAK: layer >> layer_id
        .agg(lambda x: list(x))
        .groupby(level=0)
        .agg(lambda x: list(x))
    )

    true_edges = []
    for row in hit_list.values:
        for i, j in zip(row[0:-1], row[1:]):
            true_edges.extend(list(itertools.product(i, j)))
            
    true_edges = np.array(true_edges).T
    return true_edges, hits
    
    
"""
def get_layerwise_edges(hits):
    
    # Calculate and assign parameter R = sqrt(x**2 + y**2 + z**2), 
    hits = hits.assign(
        R=np.sqrt(
            (hits.x - hits.vx) ** 2 + (hits.y - hits.vy) ** 2 + (hits.z - hits.vz) ** 2
        )
    )
    
    # Sort the hits according to R. First, reset_index and drop. Second, reset_index 
    # again to get the 'index' column which we can use later on.
    hits = hits.sort_values("R").reset_index(drop=True).reset_index(drop=False)
    
    # Find hits for particle_id == 0 and assign to -nan value 
    hits.loc[hits["particle_id"] == 0, "particle_id"] = np.nan
    
    # Get hit_list by particle_id, layer_id. It will return indice of hits
    # from each particle_id in all layers. The .agg(), are just to format the
    # the output for better handling, as we will see later.
    hit_list = (
        hits.groupby(["particle_id", "layer_id"], sort=False)["index"]
        .agg(lambda x: list(x))
        .groupby(level=0)
        .agg(lambda x: list(x))
    )
    
    # Build True Edges. First, get one row and cascade it (row[0:-1]: 0 to n-1 elements,
    # row[1:]: 1 to n elements). One can use itertools.product(b, b) to creat cartesian
    # product which is set of ordered pairs (a, b) provided a E row[0:-1] and b E row[1:].
    # The itertools.product(b, b) returns an iterable (list, set, etc), so use list.extend() 
    # to append the true_edges list. Note: list.append() add only one element to end of list.
    true_edges = []
    for row in hit_list.values:
        for i, j in zip(row[0:-1], row[1:]):
            true_edges.extend(list(itertools.product(i, j))) # extend(): extends existi
    
    # Convert to ndarray and transpose it.
    true_edges = np.array(true_edges).T
    
    # As hits are modified due R param so return it as well.
    return true_edges, hits
"""
    
    
def get_modulewise_edges(hits):
    """Get modulewise (layerless) true edge list. Here hits represent complete event."""
    signal = hits[
        ((~hits.particle_id.isna()) & (hits.particle_id != 0)) & (~hits.vx.isna())
    ]
    signal = signal.drop_duplicates(
        subset=["particle_id", "volume_id", "layer_id", "module_id"]
    )

    # Sort by increasing distance from production
    signal = signal.assign(
        R=np.sqrt(
            (signal.x - signal.vx) ** 2
            + (signal.y - signal.vy) ** 2
            + (signal.z - signal.vz) ** 2
        )
    )
    signal = signal.sort_values("R").reset_index(drop=False)

    # Handle re-indexing
    signal = signal.rename(columns={"index": "unsorted_index"}).reset_index(drop=False)
    signal.loc[signal["particle_id"] == 0, "particle_id"] = np.nan

    # Group by particle ID
    signal_list = signal.groupby(["particle_id"], sort=False)["index"].agg(
        lambda x: list(x)
    )

    true_edges = []
    for row in signal_list.values:
        for i, j in zip(row[:-1], row[1:]):
            true_edges.append([i, j])

    true_edges = np.array(true_edges).T

    true_edges = signal.unsorted_index.values[true_edges]

    return true_edges


def select_hits(event_file=None, noise=False, min_pt=None, skewed=False):
    """Hit selection method from Exa.TrkX. Build a full event, select hits based on certain criteria."""
    
    # load data using event_prefix (e.g. path/to/event0000000001)
    hits, tubes, particles, truth = trackml.dataset.load_event(event_file)
    
    # skip noise hits.
    if noise:
        # runs if noise=True
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz"]], on="particle_id", how="left"
        )
    else:
        # runs if noise=False
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz"]], on="particle_id", how="inner"
        )
    
    # assign pt (from tpx & tpy, why not px & py ???) and add to truth
    truth = truth.assign(pt=np.sqrt(truth.tpx**2 + truth.tpy**2))
    
    # apply min_pt on truth data frame
    if min_pt is not None:
        truth = truth[truth.pt > min_pt]
    
    # merge some columns of tubes to the hits, I need isochrone, skewed & sector_id
    hits = hits.merge(tubes[["hit_id", "isochrone", "skewed", "sector_id"]], on="hit_id")

    # skip skewed tubes
    if skewed is False:
        
        # filter non-skewed layers (skewed==0 for non-skewed layers & skewed==1 for skewed layers)
        hits = hits.query('skewed==0')
        
        # rename layer_ids from 0,1,2...,17 & assign a new colmn named "layer"
        vlids = hits.layer_id.unique()
        n_det_layers = hits.layer_id.unique().shape[0]
        vlid_groups = hits.groupby(['layer_id'])
        hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i) for i in range(n_det_layers)])
    
    else:
        # FIXME: This is conveniet to use layer as a column for both skewed=True or False.
        # The second way is to drop layer_id from hits, and rename layer to layer_id.
        hits = hits.rename(columns={"layer_id": "layer"})

    # Calculate derived hits variables
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    
    # Merge `hits` with `truth`, but first add `r` & `phi`
    hits = hits.assign(r=r, phi=phi).merge(truth, on="hit_id")
    
    # Add `event_id` column to this event.
    hits = hits.assign(event_id=int(event_file[-10:]))

    return hits
    
    
def build_event(event_file,
                feature_scale,
                modulewise=True,
                layerwise=True,
                noise=False,
                min_pt=None,
                # detector=None,
                inputedges=False,
                skewed=False):
    
    # Get true edge list using the ordering by R' = distance from production vertex of each particle
    hits = select_hits(event_file=event_file, noise=noise, min_pt=min_pt, skewed=skewed).assign(
        event_id=int(event_file[-10:])
    )
    
    # Make a unique module ID and attach to hits
    # TODO: Get module_id's for STT
    
    # if detector is not None:
    #    module_lookup = detector.reset_index()[["index", "volume_id", "layer_id", "module_id"]]
    #                                           .rename(columns={"index": "module_index"})
    #    hits = hits.merge(module_lookup, on=["volume_id", "layer_id", "module_id"], how="left")
    #    module_id = hits.module_index.to_numpy()
    # else:
    #    module_id = None
        
    # Get list of all layers
    layer_id = hits.layer.to_numpy()

    # Handle which truth graph(s) are being produced
    modulewise_true_edges, layerwise_true_edges = None, None
    
    # Get true edge list using the ordering of layers
    if layerwise:
        layerwise_true_edges, hits = get_layerwise_edges(hits)
        logging.info(
            "Layerwise truth graph built for {} with size {}".format(
                event_file, layerwise_true_edges.shape
            )
        )
    
    # Get true edge list without layer ordering
    if modulewise:
        modulewise_true_edges = get_modulewise_edges(hits)
        logging.info(
            "Modulewise truth graph built for {} with size {}".format(
                event_file, modulewise_true_edges.shape
            )
        )
    
    # Handle whether input graph(s) are being produced
    layerwise_input_edges = None
    
    # Get input edge list using order of layers.
    if inputedges:
        layerwise_input_edges = get_input_edges(hits)
        logging.info(
            "Layerwise input graph built for {} with size {}".format(
                event_file, layerwise_input_edges.shape
            )
        )

    # Get edge weight
    # TODO: No weights of tracks in STT data yet, skipping it.
    
    # edge_weights = (
    #    hits.weight.to_numpy()[modulewise_true_edges]
    #    if modulewise
    #    else hits.weight.to_numpy()[layerwise_true_edges]
    # )
    # edge_weight_average = (edge_weights[0] + edge_weights[1]) / 2
    # edge_weight_norm = edge_weight_average / edge_weight_average.mean()

    logging.info("Weights are not constructed, no weights for STT")

    return (
        hits[["r", "phi", "isochrone"]].to_numpy() / feature_scale,
        hits.particle_id.to_numpy(),
        layer_id,
        # module_id,
        modulewise_true_edges,
        layerwise_true_edges,
        layerwise_input_edges,
        hits["hit_id"].to_numpy(),
        hits.pt.to_numpy(),
        # edge_weight_norm,
    )    


def prepare_event(
    event_file,
    progressbar=None,
    output_dir=None,
    modulewise=True,
    layerwise=True,
    noise=False,
    min_pt=None,
    inputedges=True,
    skewed=False,
    overwrite=False,
    **kwargs
):

    """Prepare an event when called in FeatureStore Module"""
    try:
        evtid = int(event_file[-10:])
        filename = os.path.join(output_dir, str(evtid))

        if not os.path.exists(filename) or overwrite:
            logging.info("Preparing event {}".format(evtid))
            
            # feature scale for X=[r,phi,z]
            feature_scale = [100, np.pi, 100]
            
            # build event
            (
                X,
                pid,
                layer_id,
                # module_id,
                modulewise_true_edges,
                layerwise_true_edges,
                layerwise_input_edges,
                hid,
                pt,
                # weights,
            ) = build_event(
                event_file,
                feature_scale,
                modulewise=modulewise,
                layerwise=layerwise,
                noise=noise,
                min_pt=min_pt, 
                inputedges=inputedges,
                skewed=skewed
            )
            
            # build pytorch_geometric Data module
            data = Data(
                x=torch.from_numpy(X).float(),
                pid=torch.from_numpy(pid),
                # modules=torch.from_numpy(module_id),
                layers=torch.from_numpy(layer_id),
                event_file=event_file,
                hid=torch.from_numpy(hid),
                pt=torch.from_numpy(pt),
                # weights=torch.from_numpy(weights),
            )
            
            # add edges to pytorch_geometric Data module
            if modulewise_true_edges is not None:
                data.modulewise_true_edges = torch.from_numpy(modulewise_true_edges)
                
            if layerwise_true_edges is not None:
                data.layerwise_true_edges = torch.from_numpy(layerwise_true_edges)
            
            # NOTE: I am jumping from Processing to GNN stage, so I need ground truth (GT) of input
            # edges (edge_index). After Embedding, one gets GT as 'y', and after Filtering one gets 
            # the GT in the form of 'y_pid'. As I intend to skip both the Embedding & the Filtering
            # stages, the input graph and its GT is build in Processing stage. The GNN can run after
            # either Embedding or Filtering stages, so it looks for either 'y' or 'y_pid', existance of
            # one of these means the execution of these stages i.e. if 'y_pid' exists in data that means
            # both Embedding and Filtering stages has been executed. If only 'y' exists then only 
            # embedding stage has been executed. In principle, I should have only one of these in Data.
            
            # Now, for my case, I will build input graph duing Processing and also add its GT to the
            # data. If the 'edge_index' is build in Processing then ground truth (y or y_pid) should 
            # also be built here. The dimension of 'y(n)' and 'y_pid(m)' are given below, here m < n.
            
            # y(n): appears after Embedding stage along with e_radius(2,n), y.shape==e_radius.shape[1]
            # y_pid(m): appears after Filtering stage along with e_radius(2,m), y_pid.shape==e_radius.shape[1]

            if layerwise_input_edges is not None:
                input_edges = torch.from_numpy(layerwise_input_edges)
                new_input_graph, y = graph_intersection(input_edges, data.layerwise_true_edges)
                data.edge_index = new_input_graph
                # data.y = y     # if regime: [] will point to embedding
                data.y_pid = y   # if regime: [[pid]] points to filtering

            # TODO: add cell/tube information to Data, Check for STT
            
            # logging.info("Getting cell info")
            # if cell_information:
            #    data = get_cell_information(
            #        data, cell_features, detector_orig, detector_proc, endcaps, noise
            #    )

            with open(filename, "wb") as pickle_file:
                torch.save(data, pickle_file)

        else:
            logging.info("{} already exists".format(evtid))
    except Exception as inst:
        print("File:", event_file, "had exception", inst)
