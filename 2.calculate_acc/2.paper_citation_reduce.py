#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from collections import Counter

#%%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"
df = pd.read_csv(base_path + "0.datasets/processed/paper_detail_v3.csv", usecols = ["#index", "#t", "#*", "paper_reference"])

df["paper_reference"] = df["paper_reference"].apply(eval)

# %%
def paper_citation_every_year(df:pd.DataFrame):
    df_citatioin_num = pd.DataFrame(np.zeros([df.shape[0], 19]), columns = ["#index", "pub_year", "1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014"])

    df_citatioin_num["#index"] = df["#index"].apply(int)
    df_citatioin_num["pub_year"] = df["#t"].apply(int)

    df_citatioin_num.index = df["#index"].apply(int)

    for year in range(1998, 2015):
        ddf = df[df["#t"] == year]
        all_cited = []

        for citations in ddf["paper_reference"]:
            all_cited.extend(citations)
        
        cc = Counter(all_cited)

        # batch operation, complexity equals to O(1)
        df_citatioin_num.loc[list(cc.keys()), [str(int(year))]] = list(cc.values()) 

    return df_citatioin_num

df_inner_1 = paper_citation_every_year(df.copy())

#%%
df_inner_1.sample(10)

#%%
def paper_citation_reduce(df:pd.DataFrame):

    print(df.shape[0])
    print(df.columns)

    for i in range(1999, 2015):
        df[str(i)] = df[str(i - 1)] + df[str(i)]

    return df

df_inner_2 = paper_citation_reduce(df_inner_1.copy())

# %%
df_inner_2.sample(10)

# %%
df_inner_2.to_csv(base_path + "0.datasets/processed/paper_citation_reduce.csv", index = None)

# %%
