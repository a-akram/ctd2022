#!/usr/bin/env python
# coding: utf-8

"""Utilities for Input Graph Construction"""

import logging
import scipy as sp
import numpy as np
import pandas as pd
import torch

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def select_edges(hits1, hits2, filtering=True):
    """Select edges using a particular phi range or sectors. Currently, I am selecting edges 
    only in the neighboring sectors i.e. hit1 is paired with hit2 in immediate sectors only."""
    
    # Start with all possible pairs of hits, sector_id is for sectorwise selection
    keys = ['event_id', 'r', 'phi', 'isochrone', 'sector_id']
    hit_pairs = hits1[keys].reset_index().merge(hits2[keys].reset_index(), on='event_id', suffixes=('_1', '_2'))
    
    if filtering:
        dSector = (hit_pairs['sector_id_1'] - hit_pairs['sector_id_2'])
        sector_mask = ((dSector.abs() < 2) | (dSector.abs() == 5))
        edges = hit_pairs[['index_1', 'index_2']][sector_mask]
    else:
        edges = hit_pairs[['index_1', 'index_2']]
        
    return edges


def construct_edges(hits, layer_pairs, filtering=True):
    """Construct edges between hit pairs in adjacent layers"""

    # Loop over layer pairs and construct edges
    layer_groups = hits.groupby('layer')
    edges = []
    for (layer1, layer2) in layer_pairs:
        
        # Find and join all hit pairs
        try:
            hits1 = layer_groups.get_group(layer1)
            hits2 = layer_groups.get_group(layer2)
        # If an event has no hits on a layer, we get a KeyError.
        # In that case we just skip to the next layer pair
        except KeyError as e:
            logging.info('skipping empty layer: %s' % e)
            continue
        
        # Construct the edges
        edges.append(select_edges(hits1, hits2, filtering))
    
    # Combine edges from all layer pairs
    edges = pd.concat(edges)
    return edges


def get_input_edges(hits, filtering=True):
    """Build edge_index list for GNN stage."""
    n_layers = hits.layer.unique().shape[0]
    layers = np.arange(n_layers)
    layer_pairs = np.stack([layers[:-1], layers[1:]], axis=1)
    edges = construct_edges(hits, layer_pairs, filtering)
    edge_index = edges.to_numpy().T
    return edge_index


def graph_intersection(pred_graph, truth_graph):
    """Get truth information about edge_index (function is from both Embedding/Filtering)"""
    
    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1
    
    if torch.is_tensor(pred_graph):
        l1 = pred_graph.cpu().numpy()
    else:
        l1 = pred_graph
        
    if torch.is_tensor(truth_graph):
        l2 = truth_graph.cpu().numpy()
    else:
        l2 = truth_graph
        
    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()

    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()
    
    del l1
    del l2
    
    e_intersection = (e_1.multiply(e_2) - ((e_1 - e_2) > 0)).tocoo()
    
    del e_1
    del e_2
    
    new_pred_graph = (
        torch.from_numpy(np.vstack([e_intersection.row, e_intersection.col]))
        .long()
        .to(device)
    )
    
    y = torch.from_numpy(e_intersection.data > 0).to(device)
    
    del e_intersection
    
    return new_pred_graph, y
