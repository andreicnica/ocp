"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from ocpmodels.common.utils import (
    compute_neighbors,
    get_pbc_distances,
    radius_graph_pbc,
    edge_set_difference
)


class BaseModel(nn.Module):
    def __init__(
        self, num_atoms=None, bond_feat_dim=None, num_targets=None
    ) -> None:
        super(BaseModel, self).__init__()
        self.num_atoms = num_atoms
        self.bond_feat_dim = bond_feat_dim
        self.num_targets = num_targets

    def forward(self, data):
        raise NotImplementedError

    def generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
        all_edges=None,
        enforce_max_neighbors_strictly=None,
        min_dist_th=0.00001,  # For radius_graph_pbc
    ):
        # We could have issues if we actually want to pass None and override the default values
        def _default_arg(arg, arg_name, default):
            return arg if arg is not None else getattr(self, arg_name, default)
        cutoff = _default_arg(cutoff, "cutoff", None)
        max_neighbors = _default_arg(max_neighbors, "max_neighbors", None)
        use_pbc = _default_arg(use_pbc, "use_pbc", None)
        otf_graph = _default_arg(otf_graph, "otf_graph", None)
        all_edges = _default_arg(all_edges, "all_edges", None)
        # Default to old behavior
        enforce_max_neighbors_strictly = _default_arg(
            enforce_max_neighbors_strictly, "enforce_max_neighbors_strictly", True
        )

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if use_pbc:
            if otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data,
                    cutoff,
                    max_neighbors,
                    enforce_max_neighbors_strictly,
                    min_dist_th=min_dist_th,
                )
                if all_edges:
                    # Combine edges from otf with edges from data and add cell_offsets [0,0,0] (same cell)

                    # Consider only edges within the unit cell
                    cell0_filter = (cell_offsets == 0).all(dim=-1)
                    cell0_edges = edge_index[:, cell0_filter]

                    # Check missing edges. We assume both cell0_edges & data.edge_index to have unique edges
                    # Return set(data.edge_index) - set(cell0_edges)
                    missing_edges, ei_mask = edge_set_difference(data.edge_index, cell0_edges, data.num_nodes)
                    # we need to also compute the neighbors for the missing edges
                    ei_slice = data._slice_dict["edge_index"]
                    missing_neighbors = torch.stack([
                        torch.sum(ei_mask[ii: jj]) for ii, jj in zip(ei_slice[:-1], ei_slice[1:])
                    ])

                    edge_index = torch.cat((edge_index, missing_edges), dim=-1)
                    zero_cells = torch.zeros(
                        missing_edges.shape[1], 3, device=cell_offsets.device, dtype=cell_offsets.dtype
                    )
                    cell_offsets = torch.cat((cell_offsets, zero_cells), dim=0)
                    neighbors += missing_neighbors

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
        else:
            if otf_graph:
                edge_index_otf = radius_graph(
                    data.pos,
                    r=cutoff,
                    batch=data.batch,
                    max_num_neighbors=max_neighbors,
                )

                if all_edges:
                    # Combine edges from otf with edges from data. Removing duplicates
                    edge_index = torch.cat((data.edge_index, edge_index_otf), dim=-1)
                    edge_index = torch.unique(edge_index, dim=-1)

            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(
                edge_index.shape[1], 3, device=data.pos.device
            )
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=data.pos.device
            )
            neighbors = compute_neighbors(data, edge_index, ordered_edges=otf_graph and not all_edges)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
