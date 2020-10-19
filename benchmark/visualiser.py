import pickle5 as pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os

def return_times(filename, typename):
    with open(filename, 'rb') as file:
        result = pickle.load(file)
    times = []
    for n, dict in result['noisy_bipartite']['low'].items():
        times.append([n, dict['walltime']])
    df = pd.DataFrame(np.array(times), columns=['n', 'times'])
    df['type'] = typename
    return df

import os
for file in os.listdir("results"):
    if file.endswith(".pickle"):
        print(os.path.join("results", file))

sns.set_style("darkgrid")

# nonfaulty
d1 = return_times('small_results/faulty1_bte.pickle', 'bte_7_faults')
d2 = return_times('small_results/faulty1_qte.pickle', 'qte_7_faults')
d3 = return_times('small_results/faulty1_minorminer.pickle', 'mm_7_faults')
d4 = return_times('results/faulty1_bte.pickle', 'bte_7_faults')
d5 = return_times('results/faulty1_qte.pickle', 'qte_7_faults')
d6 = return_times('results/faulty1_minorminer.pickle', 'mm_7_faults')

# # fault1
# d1 = return_times('small_results/faulty1_bte.pickle', 'bte_7_faults')
# d2 = return_times('small_results/faulty1_qte.pickle', 'qte_7_faults')
# d3 = return_times('small_results/faulty1_minorminer.pickle', 'mm_7_faults')
# d4 = return_times('results/faulty1_bte.pickle', 'bte_7_faults')
# d5 = return_times('results/faulty1_qte.pickle', 'qte_7_faults')
# d6 = return_times('results/faulty1_minorminer.pickle', 'mm_7_faults')

# # fault2
# d1 = return_times('small_results/faulty2_bte.pickle', 'bte_17_faults')
# d2 = return_times('small_results/faulty2_qte.pickle', 'qte_17_faults')
# d3 = return_times('small_results/faulty2_minorminer.pickle', 'mm_17_faults')
# d4 = return_times('results/faulty2_bte.pickle', 'bte_17_faults')
# d5 = return_times('results/faulty2_qte.pickle', 'qte_17_faults')
# d6 = return_times('results/faulty2_minorminer.pickle', 'mm_17_faults')


# df = df1.set_index('n').join(df2.set_index('n'), on='n')

df = pd.concat([d1, d2, d3, d4, d5, d6])
# df = df[df.times <= 200]
df = df.rename(columns={"n": "number of guest graph nodes", "times": "time to embed"})

plt.title("Template Embedding on Chimera 2000Q with 17 Faults")
sns.lineplot(data=df, x="number of guest graph nodes", y="time to embed", hue='type')
plt.savefig("plots/" + plt.title + ".png")
