#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from collections import Counter 

#%%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

#%%
df = pd.read_csv(base_path + "0.datasets/processed/paper_detail_v2.csv")

df["paper_reference"] = df["paper_reference"].apply(eval)

#%%
# plot reference number figure
def plot_refrence_num(df):
    df["reference_number"] = df["paper_reference"].apply(lambda x: len(x))
    x = df["reference_number"].to_list()
    from collections import Counter
    cc = Counter(x)
    x = list(cc.keys())
    y = list(cc.values())
    y = np.log10(np.array(y) + 0.01).tolist()
    plt.scatter(x, y)
    plt.xlabel("reference number")
    plt.ylabel("quantity (log10)")
    plt.show()

plot_refrence_num(df)

# %%
# plot citation number figure
def plot_citation_num(df):
    citation_list = []
    for referece_list in df["paper_reference"]:
        citation_list.extend([int(id) for id in referece_list])
    cc = Counter(citation_list)
    x = list(cc.keys())
    # print(x[:10])
    y = list(cc.values())
    y = np.log10(np.array(y) + 0.01).tolist()
    plt.hist(y, 20)
    plt.xlabel("citation number")
    plt.ylabel("quantity (log10)")
    plt.show()

plot_citation_num(df)
