#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import networkx as nx
import random
import os 

#%%
import torch 
import torch.nn as nn 
import torch.nn.functional as F

#%%
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GATConv

#%%
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

#%%
seed = 8808
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

#%%
class AttentionModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(AttentionModel, self).__init__()

        # self.conv1 = RDGATLayer(in_features, hidden_features, alpha, paper_dir, paper_ref, depth, mode, ref_aggr, dir_aggr)
        # self.conv2 = RDGATLayer(hidden_features, out_features, alpha, paper_dir, paper_ref, depth, mode, ref_aggr, dir_aggr)

        self.conv1 = GATConv(in_features, hidden_features, dropout = 0.5)
        self.conv2 = GATConv(hidden_features, out_features, dropout = 0.5)

    def forward(self, x, edge_index):
        # x, edge_index = graph.x, graph.edge_index # x: [2708, 1433], edge_index: [2, 10556]
        x = self.conv1(x, edge_index) # x: [2708, 100], edge_index: [2, 10556]
        x = F.relu(x)
        x = F.dropout(x, training = self.training)

        x = self.conv2(x, edge_index) # x: [2708, 7], edge_index: [2, 10556]
        x = F.log_softmax(x, dim = 1) # x: [2708, 7]

        return x

#%%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

#%%
aminer = torch.load(base_path + "0.datasets/processed/aminer_graph.pkl")
paper_dir = torch.load(base_path + "0.datasets/processed/paper_dir.pkl")
paper_ref = torch.load(base_path + "0.datasets/processed/paper_ref.pkl")

# %%
# model = AttentionModel(128, 32, 2, 0.2, paper_dir, paper_ref, depth, mode = "mean", ref_aggr = True, dir_aggr = True)
model = AttentionModel(128, 32, 2)

# %%
loss_fn = nn.NLLLoss()
optim = torch.optim.Adam(model.parameters(), lr = 0.02)
EPOCH = 30

# %%
# train
model.train()

for epoch in range(EPOCH):
    
    optim.zero_grad()
    
    output = model(aminer.x, aminer.edge_index)
    loss = loss_fn(output[aminer.train_mask], aminer.y[aminer.train_mask])
    print("{}, loss: {}".format(epoch + 1, loss.item()))
    
    loss.backward()
    optim.step()

# %%
# evaluate
model.eval()

def evaluate(model, graph):
    output = model(graph.x, graph.edge_index)
    _, pred = torch.max(output, dim = 1)

    cf_matrix = confusion_matrix(y_true = graph.y, y_pred = pred)
    # sns.heatmap(cf_matrix, cmap = "Blues")
    accurate = torch.sum(graph.y[graph.test_mask] == pred[graph.test_mask])
    print("accuracy: {}".format(accurate / int(graph.test_mask.sum())))
    print(cf_matrix)

evaluate(model, aminer)

#%%
# visualization
model.eval()

def visualize(h, color):

    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

h = model(aminer.x, aminer.edge_index)

visualize(h, aminer.y)

#%%
