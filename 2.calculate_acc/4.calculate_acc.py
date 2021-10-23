#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#%%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

#%%
df_author = pd.read_csv(base_path + "0.datasets/processed/author_detail.csv", usecols = ["#index", "author", "position", "proportion"])
df_paper_cited = pd.read_csv(base_path + "0.datasets/processed/paper_citation_seq.csv")

#%%
"""
合并两个dataframe 表

"""
def merge_author_cited(df_author:pd.DataFrame, df_paper_cited:pd.DataFrame):

    df_all = pd.merge(df_author, df_paper_cited, how = "inner", on = "#index")
    
    return df_all

df_all = merge_author_cited(df_author.copy(), df_paper_cited.copy())

#%%
df_all.to_csv(base_path + "0.datasets/processed/author_paper_cited.csv", index = None)

# %%
def calculate_ACN(df: pd.DataFrame):

    l_author = []
    l_year = []
    l_ACN = []

    count = 0
    for author, group_author_df in df.groupby(by = "author"):
        for year, group_year_df in group_author_df.groupby(by = "pub_year"):
            
            ACN = (group_year_df["proportion"] * group_year_df["cited_1"]).mean()

            l_author.append(author)
            l_year.append(year)
            l_ACN.append(ACN)

            count += 1
            if count % 10000 == 0:
                print("No.{} has Completed!".format(count))
                print()
                
    df_result = pd.DataFrame(data = {
        "author": l_author,
        "year": l_year,
        "ACN": l_ACN 
    }, columns = ["author", "year", "ACN"])

    return df_result


df_ACN = calculate_ACN(df_all)
#%%
df_ACN.sample(10)

# %%
df_ACN.to_csv(base_path + "0.datasets/processed/author_ACN.csv", index = None)

#%%
