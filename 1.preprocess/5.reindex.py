#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData

#%%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

#%%
df = pd.read_csv(base_path + "0.datasets/processed/paper_detail_v3.csv")

df["paper_reference"] = df["paper_reference"].apply(eval)

#%%
df.sample(5)


#%%
def mapping(df):

    author_list = []
    for authors in df["#@"]:
        author_list.extend(authors.split(";"))
    author_list = sorted(author_list)
    author_mapping = {}

    with open(base_path + "0.datasets/mapping/author_mapping.txt", mode = "w") as f:
        for i, author in enumerate(author_list):
            f.write(author + "," + str(i) + "\n")
            author_mapping[author] = i


    venue_list = sorted(list(set(df["#c"].to_list())))
    venue_mapping = {}

    with open(base_path + "0.datasets/mapping/venue_mapping.txt", mode = "w") as f:
        for i, venue in enumerate(venue_list):
            f.write(venue + "," + str(i) + "\n")
            venue_mapping[venue] = i


    paper_list = df["#index"].to_list()
    paper_mapping = {}

    with open(base_path + "0.datasets/mapping/paper_mapping.txt", mode = "w") as f:
        for i, paper in enumerate(paper_list):
            f.write(str(paper) + "," + str(i) + "\n")
            paper_mapping[paper] = i


    return author_mapping, venue_mapping, paper_mapping

#%%
author_mapping, venue_mapping, paper_mapping = mapping(df)

# %%
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
len(venue_pulishes_paper)

# %%
author_writes_paper = torch.tensor(author_writes_paper, dtype = torch.long).T
paper_written_by_author = torch.tensor(paper_written_by_author, dtype = torch.long).T

venue_pulishes_paper = torch.tensor(venue_pulishes_paper, dtype = torch.long).T
paper_published_in_venue = torch.tensor(paper_published_in_venue, dtype = torch.long).T

# %%
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

# %%
hgraph
