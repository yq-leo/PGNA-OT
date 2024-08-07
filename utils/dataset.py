import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj


def load_data(dataset, p, use_attr, dtype=np.float32):
    """
    Load dataset.
    :param dataset: dataset name
    :param p: training ratio
    :param use_attr: whether to use input node attributes
    :param dtype: data type
    :return:
        edge_index1, edge_index2: edge list of graph G1, G2
        x1, x2: input node attributes of graph G1, G2
        anchor_links: training node alignments, i.e., anchor links
        test_pairs: test node alignments
    """

    data = np.load(f'{dataset}_{p:.1f}.npz')
    edge_index1, edge_index2 = data['edge_index1'].T.astype(np.int64), data['edge_index2'].T.astype(np.int64)
    anchor_links, test_pairs = data['pos_pairs'].astype(np.int64), data['test_pairs'].astype(np.int64)
    if use_attr:
        x1, x2 = data['x1'].astype(dtype), data['x2'].astype(dtype)
    else:
        x1, x2 = None, None

    return edge_index1, edge_index2, x1, x2, anchor_links, test_pairs


def build_nx_graph(edge_index, anchor_nodes, x=None):
    """
    Build a networkx graph from edge list and node attributes.
    :param edge_index: edge list of the graph
    :param anchor_nodes: anchor nodes
    :param x: node attributes of the graph
    :return: a networkx graph
    """

    G = nx.Graph()
    if x is not None:
        G.add_nodes_from(np.arange(x.shape[0]))
    G.add_edges_from(edge_index)
    G.x = x
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    G.anchor_nodes = anchor_nodes
    return G


def build_tg_graph(edge_index, x, rwr, dtype=torch.float32):
    """
    Build a PyG Data object from edge list and node attributes.
    :param edge_index: edge list of the graph
    :param x: node attributes of the graph
    :param rwr: RWR scores of the graph
    :param dtype: data type
    :return: a PyG Data object
    """

    edge_index_tensor = torch.from_numpy(edge_index.T).to(torch.int64)
    pos_x = torch.from_numpy(x).to(dtype)
    str_x = torch.from_numpy(np.sort(rwr, axis=1)).to(dtype)
    data = Data(edge_index=edge_index_tensor,
                num_nodes=pos_x.shape[0],
                pos_x=pos_x,
                str_x=str_x,
                adj=to_dense_adj(edge_index_tensor).squeeze(0))
    return data
