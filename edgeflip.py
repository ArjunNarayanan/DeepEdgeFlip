import torch
from torch_sparse import SparseTensor
import numpy as np
from torch_geometric.utils import to_undirected


def delete_edge_mask(row, col, node1, node2):
    """return a boolean mask which can be used to construct a new adjacency matrix that deletes edges
    between node1 and node2."""
    mask1 = torch.logical_or(row != node1, col != node2)
    mask2 = torch.logical_or(row != node2, col != node1)

    mask = torch.logical_and(mask1, mask2)
    return mask


def add_edge(adj, node1, node2):
    """return a new adjacency matrix that inserts a symmetric edge between node1 and node2"""
    num_nodes, _ = adj.sizes()
    edge_index = torch.tensor([[node1, node2],
                               [node2, node1]])
    new_edge = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))

    return adj + new_edge


def new_edge(row, col, node1, node2):
    """return the nodes which are connected by a new edge after edge flip"""
    neighbors1 = col[row == node1]
    neighbors2 = col[row == node2]
    edge = np.intersect1d(neighbors1, neighbors2)

    return edge


def degree(node, row, col):
    """return the degree of the given node"""
    neighbors = col[row == node]
    return len(neighbors)


def check_degree(node, row, col, maxdegree=9):
    """ return true if the degree of node is less than maxdegree"""
    d = degree(node, row, col)
    return d < maxdegree


def edge_flip(edge_index, node1, node2):
    """perform an edge flip by deleting an edge between node1 and node2 and adding an edge
    between the shared neighbors of node1 and node2"""

    row, col = edge_index[0], edge_index[1]

    edge = new_edge(row, col, node1, node2)
    if len(edge) == 2 and check_degree(edge[0], row, col) and check_degree(edge[1], row, col):
        mask = delete_edge_mask(row, col, node1, node2)
        edge_index = edge_index[:, mask]

        edges = np.hstack([edge, edge[::-1]])
        edges = torch.tensor(edges.reshape([2, 2]))
        edge_index = torch.cat([edge_index, edges], dim=1)

        return edge_index, True
    else:
        return edge_index, False


def flip_edge_by_index(edge_index, index):
    n1, n2 = edge_index[0, index].item(), edge_index[1, index].item()
    return edge_flip(edge_index, n1, n2)


num_nodes = 4
edge_index = torch.tensor([[0, 0, 0, 1, 2],
                           [1, 2, 3, 2, 3]], dtype=torch.long)
edge_index = to_undirected(edge_index)

new_edge_index = edge_flip(edge_index, 0, 2)
