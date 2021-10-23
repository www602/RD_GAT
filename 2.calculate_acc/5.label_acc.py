#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#%%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

#%%
df_ACN = pd.read_csv(base_path + "0.datasets/processed/author_ACN.csv")

df_paper = pd.read_csv(base_path + "0.datasets/processed/author_paper_cited.csv")

#%%
df_ACN.sample(10)

#%%
df_paper.sample(10)

# %%
def label_paper(df_ACN: pd.DataFrame, df_paper: pd.DataFrame):
    print("df_ACN shape: ", df_ACN.shape)
    print("df_paper shape: ", df_paper.shape)

    df_all = pd.merge(left = df_ACN, right = df_paper, how = "right", left_on = ["author", "year"], right_on = ["author", "pub_year"])
    
    print("df_all shape: ", df_all.shape)


    df_all["label_3"] = df_all["proportion"] * df_paper["cited_3"] - df_all["ACN"]
    df_all["label_5"] = df_all["proportion"] * df_all["cited_5"] - df_all["ACN"]
    df_all["label_10"] = df_all["proportion"] * df_all["cited_10"] - df_all["ACN"]

    df_all["label_3"] = df_all["label_3"].apply(lambda x : 1 if x > 0 else 0)
    df_all["label_5"] = df_all["label_5"].apply(lambda x : 1 if x > 0 else 0)
    df_all["label_10"] = df_all["label_10"].apply(lambda x : 1 if x > 0 else 0)


    df_result = pd.DataFrame(columns = ["#index", "label_3", "label_5", "label_10"])

    df_result["#index"] = df_all["#index"].copy()
    df_result["label_3"] = df_all["label_3"].copy()
    df_result["label_5"] = df_all["label_5"].copy()
    df_result["label_10"] = df_all["label_10"].copy()

    return df_result


df_inner = label_paper(df_ACN.copy(), df_paper.copy())

# %%
df_inner.sample(10)

# %%
df_inner.describe()

# %%
df_inner.to_csv(base_path + "0.datasets/processed/paper_label.csv", index = None)

# %%
def split_dataset(df_ACN: pd.DataFrame, df_year: pd.DataFrame):
    df_all = pd.merge(df_ACN, df_year, how = "inner", on = "#index")

    df_large = df_all[df_all["pub_year"] <= 2011]
    df_middle = df_all[df_all["pub_year"] <= 2009]
    df_small = df_all[df_all["pub_year"] <= 2004]

    print("large dataset: ", df_large.shape)
    print("middle dataset: ", df_middle.shape)
    print("small dataset: ", df_small.shape)

    return df_large.copy(), df_middle.copy(), df_small.copy()


df_large, df_middle, df_small = split_dataset(df_inner.copy(), df_paper[["#index", "pub_year"]].copy())

# %%
df_large.sample(10)

# %%
df_large.describe()

# %%
df_large.to_csv(base_path + "0.datasets/processed/paper_label_large.csv", index = None)

df_middle.to_csv(base_path + "0.datasets/processed/paper_label_middle.csv", index = None)

df_large.to_csv(base_path + "0.datasets/processed/paper_label_small.csv", index = None)

#%%
