#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from collections import Counter

#%%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

#%%
df = pd.read_csv(base_path + "0.datasets/processed/paper_detail_v3.csv")

df["paper_reference"] = df["paper_reference"].apply(eval)

#%%
df.sample(5)

#%%
def author_information(df:pd.DataFrame):
    """
    detect the authors' information in the network

    """
    authors_counter = Counter()
    authros_list = df["#@"]
    for i, items in enumerate(authros_list):
        authors = items.split(";")
        for author in authors:
            if authors_counter[author] == None:
                authors_counter[author] = 1
            else:
                authors_counter[author] += 1
        
        if (i + 1) % 100000 == 0:
            print("step {} finished".format(i + 1))

    return authors_counter

authors_counter = author_information(df)
# %%
len(authors_counter)

# %%
authors_counter.most_common(10)

#%%
def venue_information(df:pd.DataFrame):
    """
    detect the venues' information in the network

    """
    venue_counter = Counter()
    venue_list = df["#c"]
    for i, venue in enumerate(venue_list):
        if venue_counter[venue] == None:
            venue_counter[venue] = 1
        else:
            venue_counter[venue] += 1
    
        if (i + 1) % 100000 == 0:
            print("step {} finished".format(i + 1))

    return venue_counter


venue_counter = venue_information(df)

# %%
len(venue_counter)

# %%
venue_counter.most_common(10)

# %%
hgraph = {
    "author": {
        "num_nodes": 0
    },
    "venue": {
        "num_nodes": 0
    },
    "paper": {
        "num_nodes": 0
    },
    ("paper", "written_by", "author"): {
        "edge_index": []
    },
    ("author", "writes", "paper"): {
        "edge_index": []
    },
    ("paper", "published_in", "venue"): {
        "edge_index": []
    },
    ("venue", "publishes", "paper"): {
        "edge_index": []
    }
}

hgraph

# %%
len(authors_counter), len(venue_counter)

# %%
hgraph["author"]["num_nodes"] = len(authors_counter)
hgraph["venue"]["num_nodes"] = len(venue_counter)
hgraph["paper"]["num_nodes"] = df.shape[0]

hgraph

# %%
