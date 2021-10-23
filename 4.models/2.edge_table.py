#%%
import numpy as np 
import networkx as nx
import random
import os

import torch 

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

#%%
aminer = torch.load(base_path + "0.datasets/processed/aminer_graph.pkl")

aminer

# %%
def find_ref(graph:nx.Graph, node, depth = 3):
    paper_ref = {}
    paper_ref[0] = [node]

    layers = dict(nx.bfs_successors(graph, source = node, depth_limit = depth))

    nodes = [node]

    for i in range(1, depth + 1):
        paper_ref[i] = []
        for x in nodes:
            paper_ref[i].extend(layers.get(x,[]))
        nodes = paper_ref[i]
    
    return paper_ref


def find_dir(edge_index, num_nodes):
    paper_dir = {}
    
    for i in range(num_nodes):
        paper_dir[i] = [i]

    for item in edge_index.T:
        paper_dir[item[0].item()].append(item[1].item())

    return paper_dir

#%%
edge_index = aminer.edge_index.T.tolist()

# %%
G = nx.Graph()
G.add_edges_from(edge_index)

# %%
num_nodes = G.number_of_nodes()

#%%

paper_dir = find_dir(aminer.edge_index, num_nodes)

print("neighbors generation has finished!")

#%%
torch.save(paper_dir, base_path + "0.datasets/processed/paper_dir.pkl")

# %%
depth = 3

def find_ref(paper_dir, depth, num_nodes):
    paper_ref = {}
    for paper in range(num_nodes):
        paper_ref[paper] = {}

        # for d in range(depth + 1):
        paper_ref[paper][0] = list(set([paper]))
        paper_ref[paper][1] = list(set(paper_dir[paper]) - set(paper_ref[paper][0]))

        ll = []
        for item in paper_ref[paper][1]:
            ll.extend(paper_dir[item])
        
        ll = set(ll)

        paper_ref[paper][2] = list(ll - set(paper_ref[paper][1]))

        ll = []
        for item in paper_ref[paper][2]:
            ll.extend(paper_dir[item])
        ll = set(ll)

        paper_ref[paper][3] = list(ll - set(paper_ref[paper][2]))

    return paper_ref

#%%
paper_ref = find_ref(paper_dir, 3, num_nodes)

print("reference generation has finished!")

#%%
torch.save(paper_ref, base_path + "0.datasets/processed/paper_ref.pkl")

# %%
