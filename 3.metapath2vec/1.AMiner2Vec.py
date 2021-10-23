#%%
import numpy as np 
import pandas as pd 
import random

#%%
import torch 

#%%
from torch_geometric.nn import MetaPath2Vec
from torch_geometric.data import HeteroData

#%%
seed = 8808
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

#%%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

#%%
df = pd.read_csv(base_path + "0.datasets/processed/paper_detail_v3.csv")

df["paper_reference"] = df["paper_reference"].apply(eval)

#%%
def mapping(df):

    author_list = []
    for authors in df["#@"]:
        author_list.extend(authors.split(";"))
    author_list = sorted(author_list)
    author_mapping = {}

    for i, author in enumerate(author_list):
        author_mapping[author] = i


    venue_list = sorted(list(set(df["#c"].to_list())))
    venue_mapping = {}

    for i, venue in enumerate(venue_list):
        venue_mapping[venue] = i


    paper_list = df["#index"].to_list()
    paper_mapping = {}

    for i, paper in enumerate(paper_list):
        paper_mapping[paper] = i

    return author_mapping, venue_mapping, paper_mapping

#%%
author_mapping, venue_mapping, paper_mapping = mapping(df)

#%%
def relationship(df):
  
    author_writes_paper = []
    paper_written_by_author = []

    venue_pulishes_paper = []
    paper_published_in_venue = []

    for authors, venue, paper in zip(df["#@"], df["#c"], df["#index"]):
        # add venue-publishes-paper relation
        venue_pulishes_paper.append((venue_mapping[venue], paper_mapping[paper]))
        paper_published_in_venue.append((paper_mapping[paper], venue_mapping[venue]))

        # add author-writes-paper relation
        for author in authors.split(";"):
            author_writes_paper.append((author_mapping[author], paper_mapping[paper]))
            paper_written_by_author.append((paper_mapping[paper], author_mapping[author]))
    
    return author_writes_paper, paper_written_by_author, venue_pulishes_paper, paper_published_in_venue

#%%
author_writes_paper, paper_written_by_author, venue_pulishes_paper, paper_published_in_venue = relationship(df)

#%%
author_writes_paper = torch.tensor(author_writes_paper, dtype = torch.long).T
paper_written_by_author = torch.tensor(paper_written_by_author, dtype = torch.long).T

venue_pulishes_paper = torch.tensor(venue_pulishes_paper, dtype = torch.long).T
paper_published_in_venue = torch.tensor(paper_published_in_venue, dtype = torch.long).T

#%%
hgraph = {
    "author": {
        "num_nodes": len(author_mapping)
    },
    "venue": {
        "num_nodes": len(venue_mapping)
    },
    "paper": {
        "num_nodes": len(paper_mapping)
    },
    ("paper", "written_by", "author"): {
        "edge_index": paper_written_by_author
    },
    ("author", "writes", "paper"): {
        "edge_index": author_writes_paper
    },
    ("paper", "published_in", "venue"): {
        "edge_index": paper_published_in_venue
    },
    ("venue", "publishes", "paper"): {
        "edge_index": venue_pulishes_paper
    }
}

hgraph = HeteroData(hgraph)

#%%
hgraph

#%%
metapath = [
    ('author', 'writes', 'paper'),
    ('paper', 'published_in', 'venue'),
    ('venue', 'publishes', 'paper'),
    ('paper', 'written_by', 'author'),
]

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
model = MetaPath2Vec(hgraph.edge_index_dict, embedding_dim=128,
                     metapath=metapath, walk_length=50, context_size=7,
                     walks_per_node=5, num_negative_samples=5,
                     sparse=True).to(device)

#%%
loader = model.loader(batch_size=128, shuffle=True, num_workers=6)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

#%%
def train(epoch, log_steps=100, eval_steps=2000):
    model.train()

    total_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Loss: {total_loss / log_steps:.4f}'))
            total_loss = 0

        if (i + 1) % eval_steps == 0:
            # acc = test()
            # print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                  #  f'Acc: {acc:.4f}'))
            print("test step!")

#%%
def test(train_ratio=0.1):
    model.eval()

    z = model('author', batch = hgraph['author'].y_index)
    y = hgraph['author'].y

    perm = torch.randperm(z.size(0))
    train_perm = perm[:int(z.size(0) * train_ratio)]
    test_perm = perm[int(z.size(0) * train_ratio):]

    return model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],
                      max_iter=150)

#%%
for epoch in range(1, 6):
    train(epoch)
    # acc = test()
    # print(f'Epoch: {epoch + 1}, Accuracy: {acc:.4f}')
    print(f'Epoch: {epoch + 1}, Finished!')
    
#%%
# torch.save(model, base_path + "0.datasets/embedding/aminer2vec.model")

