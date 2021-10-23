#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#%%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

#%%
df = pd.read_csv(base_path + "0.datasets/processed/paper_detail_v3.csv", usecols = ["#index", "#@"])

# %%
def primary_author_info(df:pd.DataFrame):
    df["author"] = df["#@"].apply(lambda x : x.split(";")[0])

    def get_position(authors:str):
        l = len(authors.split(";"))
        if l == 1:
            return "A1"
        elif l == 2:
            return "B1"
        elif l >= 3:
            return "C1"

    df["position"] = df["#@"].apply(get_position)
    return df

df_inner_1 = primary_author_info(df.copy())

# %%
def assign_proportion(df:pd.DataFrame):
    def proportion(position:str):
        position_weight = {
            "A1": 1.0,
            
            "B1": 0.7,
            "B2": 0.3,
            
            "C1": 0.6,
            "C2": 0.2,
            "C3": 0.2
        }
        
        return position_weight[position]


    df["proportion"] = df["position"].apply(proportion)

    return df

df_inner_2 = assign_proportion(df_inner_1.copy())

#%%
df_inner_2.to_csv(base_path + "0.datasets/processed/author_detail.csv", index = None)

#%%
