#%%
import numpy as np 
import pandas as pd 
import os

import random

import torch 
from torch_geometric.data import Data

#%%
seed = 8808
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

#%%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

# %%
# get train_mask and test_mask
def train_test_split(num_papers, train_ratio = 0.7):
    train_papers = int(num_papers * train_ratio)
    test_papers = num_papers - train_papers

    train_index = torch.tensor([True for x in range(train_papers)])
    test_index = torch.tensor([False for x in range(test_papers)])
    bool_index = torch.cat([train_index, test_index])

    train_mask = bool_index[torch.randperm(num_papers)]
    test_mask = torch.tensor([not x for x in train_mask])

    return train_mask, test_mask

#%%
train_mask, test_mask = train_test_split(666466)
train_mask, test_mask

# %%
# get edge_index
def get_edge_index(paper_mapping):

    df = pd.read_csv(base_path + "0.datasets/processed/paper_detail_v3.csv", usecols = ["#index", "paper_reference"])
    df["paper_reference"] = df["paper_reference"].apply(eval)

    edge_index = []

    for i, s in df.iterrows():
        edge_index.extend([[paper_mapping[s["#index"]], paper_mapping[reference]] for reference in s["paper_reference"]])

        if i % 5000 == 0:
            print(i)
    
    edge_index = torch.tensor(edge_index, dtype = torch.long).T

    return edge_index

#%%
paper_mapping = {}
with open(base_path + "0.datasets/mapping/paper_mapping.txt") as f:
    lines = f.readlines()
    for line in lines:
        items = line.split(",")
        paper_mapping[int(items[0])] = int(items[1])

len(paper_mapping)

#%%
edge_index = get_edge_index(paper_mapping)

edge_index.shape

#%%
# get x
# x.shape: [666466, 128]
def get_x(hidden_size = 128):
    paper_embedding = torch.load(base_path + "0.datasets/embedding/paper.mtp")
    assert paper_embedding.shape[1] == hidden_size
    return paper_embedding


emb = get_x()
# %%
# get y
def get_y():
    df = pd.read_csv(base_path + "0.datasets/processed/labels/paper_label.csv")
    
    df["#index"] = df["#index"].apply(lambda x : paper_mapping[x])

    df = df.sort_values(by = "#index", ascending = True)

    y = df["label_5"].to_list()

    y = torch.tensor(y, dtype = torch.long)

    return y

y = get_y()

y

#%%
aminer = Data(x = emb, edge_index = edge_index, y = y, train_mask = train_mask, test_mask = test_mask)

# %%
torch.save(aminer, base_path + "0.datasets/processed/aminer_graph.pkl")

# %%
