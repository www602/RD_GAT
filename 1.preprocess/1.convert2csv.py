#%%
"""
convert the raw data into dataframes separately (50000 records per .csv file), the schema is (“#index”, “#*”, “#@”, “#o”, “#t”, “#c”, “paper_reference”, “#!”).

#index: Id for each paper
#*: title
#@: authors (separated by semicolons)
#o: affilations (separated by semicolons)
#t: year
#c: publication venue
Paper_reference: [id_r1,id_ r2, id_r3, …]
#!: abstract

"""

#%%
import pandas as pd
import os 
from collections import Counter

# %%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/submission/"

file = open(base_path + "datasets/raw/AMiner-Paper.txt")
ss = file.read()
ss = ss.strip()
file.close()

ss_list = ss.split("\n\n")

#%%
batch_size = 50000

for step in range(len(ss_list) // batch_size):

    paper_list = []
    for paper in ss_list[step * batch_size : (step + 1) * batch_size]:
        onePaper = {
            "#index": "<unk>",
            "#*": "<unk>",
            "#@": "<unk>",
            "#o": "<unk>",
            "#t": "<unk>",
            "#c": "<unk>",
            "paper_reference": [],
            "#!": "<unk>"
        }

        """
        handle one paper per step

        """
        properties = paper.split("\n")
        for row in properties:
            """
            handle one row per step
            """
            item = row.split(" ")
            name = item[0]
            value = " ".join(item[1:])


            if name != "#%":
                onePaper[name] = value
            else:
                onePaper["paper_reference"].append(value)

        paper_list.append(onePaper)

    df = pd.DataFrame(paper_list)

    df.to_csv(base_path + "/processed/blocks/" + str(step * batch_size) + "_" + str((step + 1) * batch_size) + ".csv", index = None)

    del(paper_list)
    del(df)

    print("block {} finished!".format(step))
