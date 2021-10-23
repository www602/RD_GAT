# %%
import pandas as pd 
import os

#%%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

def merge_paper_partitions():

    result = None

    for file_name in os.listdir(base_path + "0.datasets/processed/blocks"):
        if result is None:
            result = pd.read_csv(base_path + file_name)
        else:
            df = pd.read_csv(base_path + file_name)
            result = pd.concat([result, df])
            del df
        print(result.shape)
    
    result.to_csv(base_path + "0.datasets/processed/paper_detail_v1.csv", index = None)

#%%

