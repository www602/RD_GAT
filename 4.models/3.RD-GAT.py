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

class RDGATLayer(nn.Module):

    def __init__(self, in_features, out_features, alpha, paper_dir, paper_ref, depth, mode = "linear", ref_aggr = True, dir_aggr = True):
        super(RDGATLayer, self).__init__()

        assert mode in ("linear", "sum", "mean", "max") 

        self.W_ref = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W_ref.data, nn.init.calculate_gain("leaky_relu", alpha))

        self.a_ref = nn.Parameter(torch.Tensor(out_features * 2, 1))
        nn.init.xavier_uniform_(self.W_ref.data, nn.init.calculate_gain("leaky_relu", alpha))


        self.W_dir = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W_dir.data, nn.init.calculate_gain("leaky_relu", alpha))

        self.a_dir = nn.Parameter(torch.Tensor(out_features * 2, 1))
        nn.init.xavier_uniform_(self.a_dir.data, nn.init.calculate_gain("leaky_relu", alpha))

        self.linear = nn.Linear(out_features * 2, out_features)

        
        self.leakyrelu = nn.LeakyReLU(alpha)

        self.paper_ref = paper_ref
        self.depth = depth

        self.paper_dir = paper_dir
        
        self.mode = mode
        self.ref_aggr = ref_aggr
        self.dir_aggr = dir_aggr


    def forward_ref(self, h):
        # h = F.dropout(h, training = self.training)
        Wh = torch.mm(h, self.W_ref)

        r = torch.zeros_like(Wh) # r: [2703, 100]

        # accumulate features of neighbors from different depth 
        for node, neighbors_d in self.paper_ref.items():
            alpha = []
            # print("current node: ", node)
            # calclulate reference features by accumulating the features of every neighbor layer.
            f_ref = torch.zeros(self.depth + 1, Wh.shape[1])

            for depth in range(self.depth + 1):
                neighbors = neighbors_d[depth]

                if len(neighbors) == 0:
                    f_ref[depth] = torch.zeros(Wh.shape[1])

                else:
                    # sample if the neighbors are too many
                    if len(neighbors) > 10:
                        neighbors = random.choices(neighbors, k = 10)

                    f_ref[depth] = torch.mean(torch.index_select(Wh, 0, torch.tensor(neighbors)), dim = 0)
            
            alpha = []
            for d in range(depth + 1):
                # e = F.leaky_relu(self.linear_2(torch.cat([x[node], f_ref[d]])), 0.2)
                e = self.leakyrelu(torch.mm(torch.cat([Wh[node], f_ref[d]]).view(1, -1), self.a_ref))
                alpha.append(e.item())

            alpha = F.softmax(torch.tensor(alpha), dim = -1)

            h = torch.zeros_like(Wh[0])
            
            for d in range(depth + 1):
                h += f_ref[d] * alpha[d]

            h = torch.sigmoid(h)
            
            r[node] = h

            if (node + 1) % 20000 == 0:
                print("Ref Step: {}".format(node + 1))
        
        return r


    def forward_dir(self, h):
        
        Wh = torch.mm(h, self.W_dir) # [2703, 100]

        r = torch.zeros_like(Wh)

        for node, neighbors in self.paper_dir.items():
            alpha = neighbors.copy()

            for i, neighbor in enumerate(neighbors):
                e = self.leakyrelu(torch.mm(torch.cat([Wh[node], Wh[neighbor]]).view(1, -1), self.a_dir))
                alpha[i] = e.item()

            alpha = F.softmax(torch.tensor(alpha), dim = -1)


            h = torch.zeros_like(Wh[0])
            for i, neighbor in enumerate(neighbors):
                h += Wh[neighbor] * alpha[i]
            h = torch.sigmoid(h)


            r[node] = h

            if (node + 1) % 20000 == 0:
                print("Dir Step: {}".format(node + 1))
        
        return r


    def forward(self, h):

        if self.ref_aggr and not self.dir_aggr:
            h_ref = self.forward_ref(h)
            return h_ref

        if not self.ref_aggr and self.dir_aggr:
            h_dir = self.forward_dir(h)
            return h_dir

        # print("h_ref shape: ", h_ref.shape)
        # print("h_dir shape: ", h_dir.shape)

        h_ref = self.forward_ref(h)
        h_dir = self.forward_dir(h)

        h_prime = None

        if self.mode == "linear":
            h_prime = self.linear(torch.cat([h_ref, h_dir], dim = 1))
        elif self.mode == "sum":
            h_prime = torch.add(h_ref, h_dir)
        elif self.mode == "mean":
            h_prime = torch.add(h_ref, h_dir) / 2
        elif self.mode == "max":
            h_prime = torch.where(h_ref > h_dir, h_ref, h_dir)

        return h_prime
        
#%%
class AttentionModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, alpha, paper_dir, paper_ref, depth, mode = "linear", ref_aggr = True, dir_aggr = True):
        super(AttentionModel, self).__init__()

        self.conv1 = RDGATLayer(in_features, hidden_features, alpha, paper_dir, paper_ref, depth, mode, ref_aggr, dir_aggr)
        self.conv2 = RDGATLayer(hidden_features, out_features, alpha, paper_dir, paper_ref, depth, mode, ref_aggr, dir_aggr)

    def forward(self, x):
        # x, edge_index = graph.x, graph.edge_index # x: [2708, 1433], edge_index: [2, 10556]
        x = self.conv1(x) # x: [2708, 100], edge_index: [2, 10556]
        x = F.relu(x)
        x = F.dropout(x, training = self.training)

        x = self.conv2(x) # x: [2708, 7], edge_index: [2, 10556]
        x = F.log_softmax(x, dim = 1) # x: [2708, 7]

        return x

#%%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

#%%
aminer = torch.load(base_path + "0.datasets/processed/aminer_graph.pkl")
paper_dir = torch.load(base_path + "0.datasets/processed/paper_dir.pkl")
paper_ref = torch.load(base_path + "0.datasets/processed/paper_ref.pkl")


#%%
depth = 3

# %%
model = AttentionModel(128, 32, 2, 0.2, paper_dir, paper_ref, depth, mode = "mean", ref_aggr = True, dir_aggr = True)

# %%
loss_fn = nn.NLLLoss()
optim = torch.optim.Adam(model.parameters(), lr = 0.02)
EPOCH = 30

# %%
# train
model.train()

for epoch in range(EPOCH):
    
    optim.zero_grad()
    
    output = model(aminer.x)
    loss = loss_fn(output[aminer.train_mask], aminer.y[aminer.train_mask])
    print("{}, loss: {}".format(epoch + 1, loss.item()))
    
    loss.backward()
    optim.step()

# %%
# evaluate
model.eval()

def evaluate(model, graph):
    output = model(graph.x)
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

h = model(model.x)

visualize(h, model.y)

#%%

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_score()