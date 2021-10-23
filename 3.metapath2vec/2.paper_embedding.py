#%%
import torch
from torch_geometric.data import HeteroData

#%%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

#%%
metapath = [
    ('author', 'writes', 'paper'),
    ('paper', 'published_in', 'venue'),
    ('venue', 'publishes', 'paper'),
    ('paper', 'written_by', 'author'),
]

#%%
model = torch.load(base_path + "0.datasets/embedding/aminer2vec.model", map_location = torch.device("cpu"))

# %%
types = set([x[0] for x in metapath]) | set([x[-1] for x in metapath])
types = sorted(list(types))

types

#%%
def get_embedding(node_type, batch = None):
    emb = model.embedding.weight[model.start[node_type]:model.end[node_type]]
    return emb if batch is None else emb[batch]

#%%
model.embedding.weight.data.shape

# %%
paper_embedding = get_embedding("paper")

# %%
torch.save(paper_embedding, base_path + "0.datasets/embedding/paper.mtp")

