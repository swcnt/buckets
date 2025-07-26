import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# Add the directory containing generated csvs here:
dirname = "./plots/day2"
directory = os.fsencode(dirname)
out_filename = "day2.png"

lambdas = []
mrts = []
labels = []

def get_data(df):
        
    lam = df['Lambda'][1:]
    mrt = df[' Mean Response Time'][1:]
    assert len(lam) == len(mrt)
    adj_lam,adj_mrt = [],[]
    
    # go through and stop once an error hits
    
    for i in range(1,len(mrt)+1): # wow the dataframe starts at 1, so cool
        if "Error" in lam[i]:
            break
        else:
            adj_lam.append(lam[i])
            adj_mrt.append(mrt[i])
    return adj_lam,adj_mrt

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    ext = os.getcwd()
    full = ext + '/' + dirname +  '/' + filename
    if filename.endswith(".csv"):
        df = pd.read_csv(full,sep=";")
        labels.append(df['Lambda'][0])
        L,M = get_data(df)
        lambdas.append(L)
        mrts.append(M)
    else:
        continue

plt.figure(figsize=(8, 6))
ax = plt.gca()

for i in range(len(lambdas)):
    sns.lineplot(
        x=lambdas[i],
        y=mrts[i],
        label=labels[i], 
        ax=ax,               
        linewidth=2.5,
        marker='o'          
    )

ax.set(
    title='Policy Analysis',
    xlabel='Lambda (Arrival Rate)',
    ylabel='Mean Response Time'
)

# custom legend (did not know you could do this)
plt.legend(
    title='Legend',
    loc='lower center',        
    bbox_to_anchor=(0.5, 1.05), 
    ncol=1,                      
    frameon=True                
)

plt.grid(True)

plt.savefig('./plots/outputs/' + out_filename, bbox_inches='tight')


