import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset

dataset = DglNodePropPredDataset('ogbn-arxiv')
device = 'cpu'      # change to 'cuda' for GPU

graph, node_labels = dataset[0]
# Add reverse edges since ogbn-arxiv is unidirectional.
graph = dgl.add_reverse_edges(graph)
graph.ndata['label'] = node_labels[:, 0]
print(graph)
print(node_labels)

node_features = graph.ndata['feat']
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()
print('Number of classes:', num_classes)

idx_split = dataset.get_idx_split()
train_nids = idx_split['train']
valid_nids = idx_split['valid']
test_nids = idx_split['test']

sampler = dgl.dataloading.NeighborSampler([4, 4])
train_dataloader = dgl.dataloading.DataLoader(
    # The following arguments are specific to DGL's DataLoader.
    graph,              # The graph
    train_nids,         # The node IDs to iterate over in minibatches
    sampler,            # The neighbor sampler
    device=device,      # Put the sampled MFGs on CPU or GPU
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=1024,    # Batch size
    shuffle=True,       # Whether to shuffle the nodes for every epoch
    drop_last=False,    # Whether to drop the last incomplete batch
    num_workers=0       # Number of sampler processes
)

input_nodes, output_nodes, mfgs = example_minibatch = next(iter(train_dataloader))
print(example_minibatch)
print("To compute {} nodes' outputs, we need {} nodes' input features".format(len(output_nodes), len(input_nodes)))

mfg_0_src = mfgs[0].srcdata[dgl.NID]
mfg_0_dst = mfgs[0].dstdata[dgl.NID]
print(mfg_0_src)
print(mfg_0_dst)
print(torch.equal(mfg_0_src[:mfgs[0].num_dst_nodes()], mfg_0_dst))

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        # Lines that are changed are marked with an arrow: "<---"

        h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        h = self.conv1(mfgs[0], (x, h_dst))  # <---
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]  # <---
        h = self.conv2(mfgs[1], (h, h_dst))  # <---
        return h

model = Model(num_features, 128, num_classes).to(device)

opt = torch.optim.Adam(model.parameters())

valid_dataloader = dgl.dataloading.DataLoader(
    graph, valid_nids, sampler,
    batch_size=1024,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
)

import tqdm
import sklearn.metrics

best_accuracy = 0
best_model_path = 'model.pt'
for epoch in range(10):
    model.train()

    with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
            # feature copy from CPU to GPU takes place here
            inputs = mfgs[0].srcdata['feat']
            labels = mfgs[-1].dstdata['label']

            predictions = model(mfgs, inputs)

            loss = F.cross_entropy(predictions, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())

            tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)

    model.eval()

    predictions = []
    labels = []
    with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            inputs = mfgs[0].srcdata['feat']
            labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
            predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)

        # train for 1 epoch and save embeddings
        break

# get testing loader as well
test_dataloader = dgl.dataloading.DataLoader(
    graph, test_nids, sampler,
    batch_size=1024,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
)

"""
retrieving all the data and append into our HDC ready data, 
each entity is a tuple of (embedding, label), where embedding and label are numpy arrays
"""
train_emb = []
val_emb = []
test_emb = []
# save original features
train_features = []
val_features = []
test_features = []

# training
with tqdm.tqdm(train_dataloader) as tq:
    for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
        inputs = mfgs[0].srcdata['feat']
        labels = mfgs[-1].dstdata['label']
        predictions = model(mfgs, inputs)
        for i, l in enumerate(labels):
            train_emb.append((predictions[i].detach().numpy(), l.numpy()))
            train_features.append(inputs[i].detach().numpy())
print("Retreived", len(train_emb), "training embeddings")

# validation
with tqdm.tqdm(valid_dataloader) as tq:
    for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
        inputs = mfgs[0].srcdata['feat']
        labels = mfgs[-1].dstdata['label']
        predictions = model(mfgs, inputs)
        for i, l in enumerate(labels):
            val_emb.append((predictions[i].detach().numpy(), l.numpy()))
            val_features.append(inputs[i].detach().numpy())
print("Retreived",len(val_emb), "validation embeddings")

# testing
with tqdm.tqdm(test_dataloader) as tq:
    for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
        inputs = mfgs[0].srcdata['feat']
        labels = mfgs[-1].dstdata['label']
        predictions = model(mfgs, inputs)
        for i, l in enumerate(labels):
            test_emb.append((predictions[i].detach().numpy(), l.numpy()))
            test_features.append(inputs[i].detach().numpy())
print("Retreived", len(test_emb), " test embeddings")

# do something with the embeddings
# export embeddings to a file that HDC can read
# can be loaded with np.load()
normalized_train_emb = []
train_labels = []
normalized_val_emb = []
val_labels = []
normalized_test_emb = []
test_labels = []

for emb, label in train_emb:
    normalized_train_emb.append(emb / np.linalg.norm(emb))
    train_labels.append(label)
for emb, label in val_emb:
    normalized_val_emb.append(emb / np.linalg.norm(emb))
    val_labels.append(label)
for emb, label in test_emb:
    normalized_test_emb.append(emb / np.linalg.norm(emb))
    test_labels.append(label)

np.save("train_emb.npy", normalized_train_emb)
np.save("train_labels.npy", train_labels)
np.save("val_emb.npy", normalized_val_emb)
np.save("val_labels.npy", val_labels)
np.save("test_emb.npy", normalized_test_emb)
np.save("test_labels.npy", test_labels)


# for input_nodes, output_nodes, mfgs in train_dataloader:
#     train_features.append(mfgs[0].srcdata['feat'].numpy())
#     print(mfgs[0].srcdata['feat'].numpy().shape)
# for input_nodes, output_nodes, mfgs in valid_dataloader:
#     val_features.append(mfgs[0].srcdata['feat'].numpy())
# for input_nodes, output_nodes, mfgs in test_dataloader:
#     test_features.append(mfgs[0].srcdata['feat'].numpy())

# save original features
np.save("train_features.npy", train_features)
np.save("val_features.npy", val_features)
np.save("test_features.npy", test_features)

