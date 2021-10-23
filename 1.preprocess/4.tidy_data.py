#%%
from collections import Counter
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# %%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

df = pd.read_csv(base_path + "0.datasets/processed/paper_detail_v3.csv")

df["paper_reference"] = df["paper_reference"].apply(eval)

# %%
print("total number of paper: ", df.shape[0])

# %%
# tidy data
def tidy_df(df: pd.DataFrame):

    df = df.dropna()
    print("paper delete null: ", df.shape[0])

    df = df[df["#!"] != "<unk>"]
    print("paper with abstract: ", df.shape[0])

    df = df[df["#t"] >= 1998]
    print("paper after 1998: ", df.shape[0])
    
    df = df[df["reference_number"] > 0]
    print("paper with reference: ", df.shape[0])

    # preserve the 0 in-degree paper
    # df = df[df["citation_num"] > 0]
    # print("paper with citation: ", df.shape[0])

    df = df[df["#@"].apply(lambda x : True if type(x) == type("str") else False)]
    print("paper with author: ", df.shape[0])
    
    return df

#%%
df_inner = tidy_df(df.copy())

# %%
def construct_graph(df: pd.DataFrame):
    print("Nodes in graph: ", df.shape[0])
    
    df["paper_reference"] = df["paper_reference"].apply(lambda x : [int(id) for id in x])

    # print(df.sample(10)["paper_reference"])

    # reconstruct the reference relationship
    all_paper_index = list(df["#index"])
    all_paper_index = set(all_paper_index)


    def get_range(refer_list: list):
        result = []
        for paper in refer_list:
            if paper in all_paper_index:
                result.append(paper)
        return result

    # df["paper_reference"] = df["paper_reference"].apply(lambda x : [id for id in x if id in all_paper_index])

    df["paper_reference"] = df["paper_reference"].apply(get_range)

    # df = df[df["paper_reference"].apply(lambda x : True if len(x) > 0 else False)]

    # print("delete null reference: ", df.shape[0])

    print("Nodes in graph: ", df.shape[0])
    
    return df


df_inner_2 = construct_graph(df_inner.copy())

#%%
# delete nodes with no citation and reference, that's to say. the isolation node
def delete_isolation(df:pd.DataFrame):
    print(df.shape)

    set_all = set(df["#index"])

    set_0_exit = set(list(df[df["paper_reference"].apply(lambda x : True if len(x) == 0 else False)]["#index"]))

    set_non_0_exit = set_all - set_0_exit

    result = []
    for item in df["paper_reference"]:
        result.extend(item)

    set_non_0_enter = set(result)
    set_0_enter = set_all - set_non_0_enter

    # assert set_0_exit < set_all & set_non_0_exit < set_all & set_0_enter < set_all & set_non_0_enter < set_all, "非子集"

    assert (set_0_exit < set_all) & (set_non_0_exit < set_all) & (set_0_enter < set_all) & (set_non_0_enter < set_all), "非子集"

    print("Total nodes: ", len(set_all))
    print("0 in-degree: ", len(set_0_exit))
    print("0 out-degree: ", len(set_0_enter))
    print("Not 0 in-degree: ", len(set_non_0_enter))
    print("Not 0 out-degree: ", len(set_non_0_exit))

    print("0 in-degree and 0 out-degree: ", len(set_0_exit & set_0_enter))
    print("0 out-degree and not 0 in-degree: ", len(set_0_exit & set_non_0_enter))
    print("not 0 out-degree and 0 in-degree: ", len(set_non_0_exit & set_0_enter))
    print("not 0 out-degree and not 0 out-degree: ", len(set_non_0_exit & set_non_0_enter))


    print("expurgate the dataframe.")

    set_0_0 = set_0_exit & set_0_enter

    df_return = df[df["#index"].apply(lambda x : True if x not in set_0_0 else False)]

    return df_return

df_processed = delete_isolation(df_inner_2.copy())

print("after deleting isolation paper: ", df_processed.shape[0])

# %%
df_processed["reference_number"] = df_processed["paper_reference"].apply(lambda x : len(x))

#%%
def cited_num(df):
    citation_list = []
    for referece_list in df["paper_reference"]:
        citation_list.extend([int(id) for id in referece_list])
    cc = Counter(citation_list)
    df["citation_num"] = df["#index"].apply(lambda x : cc.get(x, 0))
    
    return df
    
df_processed = cited_num(df_processed)

#%%
df_processed.sample(10)

# %%
# save the dataframe
# df_processed.to_csv(base_path + "0.datasets/processed/paper_detail_v3.csv", index = None)

# %%
print("reference edge: ", np.sum(df_processed["reference_number"]))
print("citation edge: ", np.sum(df_processed["citation_num"]))

# %%
