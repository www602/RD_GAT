#%%
from collections import Counter
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# %%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

df = pd.read_csv(base_path + "0.datasets/processed/paper_detail_v1.csv")

#%%
df["paper_reference"] = df["paper_reference"].apply(eval)

#%%
# calculate not <unk> in abstract
print("the number of all paper is ", df.shape[0])
print("the number of <unk> of abstract is ", df[df["#!"] == "<unk>"].shape[0])
print("the number of not <unk> of abstract is ", df.shape[0] - df[df["#!"] == "<unk>"].shape[0])

# %%
# calculate paper reference number
reference_not_null = df[df["paper_reference"].apply(lambda x: True if len(x) > 0 else False)].shape[0]

print("the number of null of reference is ", df.shape[0] - reference_not_null)
print("the number of non-null of reference is ", reference_not_null)

# %%
def cited_num(df):
    citation_list = []
    for referece_list in df["paper_reference"]:
        citation_list.extend([int(id) for id in referece_list])
    cc = Counter(citation_list)
    df["citation_num"] = df["#index"].apply(lambda x : cc.get(x, 0))
    
    return df
    
df = cited_num(df)
df.sample(10)

df.to_csv(base_path + "0.datasets/processed/paper_detail_v2.csv", index = None)



