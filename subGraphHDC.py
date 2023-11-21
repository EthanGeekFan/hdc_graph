import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset

# HDC stuff
from hdc.bsc import BSC as HDC
from hdc.hv import HDCRepresentation
from hdc.itemmem import ItemMem
from hdc.encoder import Encoder
from hdc.levelencoder import LevelEncoder
from tqdm import tqdm
from typing import Type
from hdc.itemmem import HighResItemMem

hdc = HDC
N = 10240
encoder = LevelEncoder(hdc, N, -1, 1, 0.05)

dataset = DglNodePropPredDataset('ogbn-arxiv')

graph, node_labels = dataset[0]
print(graph)
print(node_labels)

node_features = graph.ndata['feat']
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()
print('Number of features:', num_features)
print('Number of classes:', num_classes)
# print range of node features
print('Node feature range: [{}, {}]'.format(node_features.min().item(), node_features.max().item()))

# split to train and test only
idx_split = dataset.get_idx_split()
train_nids = idx_split['train']
valid_nids = idx_split['valid']
test_nids = idx_split['test']

