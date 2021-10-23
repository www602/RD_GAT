#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from collections import Counter
from functional import seq

# %%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"
df = pd.read_csv(base_path + "0.datasets/processed/paper_citation_reduce.csv")

#%%
df.sample(10)

# %%
def convert2seq(df:pd.DataFrame):
    print(df.shape)
    
    col_names = ["#index", "pub_year"]
    
    col_names.extend(["cited_" + str(i) for i in range(0, 11)])

    df_seq = pd.DataFrame(np.zeros([df.shape[0], 13]), columns = col_names)
    df_seq["#index"] = df["#index"]
    df_seq["pub_year"] = df["pub_year"]

    result = {
        "cited_0": [],
        "cited_1": [],
        "cited_2": [],
        "cited_3": [],
        "cited_4": [],
        "cited_5": [],
        "cited_6": [],
        "cited_7": [],
        "cited_8": [],
        "cited_9": [],
        "cited_10": []
        }

    for left, right in zip(df.iterrows(), df_seq.iterrows()):
        left_index, left_row = left
        right_index, right_row = right

        for i in range(0, 11):
            current_year = int(left_row["pub_year"]) + i
            if current_year > 2014:
                result["cited_" + str(i)].append(0)
            else:
                result["cited_" + str(i)].append(left_row[str(current_year)])
    
    for i in range(0, 11):
        df_seq["cited_" + str(i)] = result["cited_" + str(i)]

    return df_seq 

df_inner = convert2seq(df.copy())

#%%
df_inner.sample(10)

# %%
df_inner.to_csv(base_path + "0.datasets/processed/paper_citation_seq.csv", index = None)

# %%
