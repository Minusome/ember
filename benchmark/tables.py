# import pickle5 as pickle
#
# with open(f'results/faulty1_qte.pickle', 'rb') as handle:
#     nonfaulty_bte = pickle.load(handle)
#
# for graph_type, dict1 in nonfaulty_bte.items():
#     print("graph type: " + graph_type)
#     for edge_density,dict2 in dict1.items():
#         print("edge density: " + edge_density)
#         max_n = max(dict2, key=int)
#         print(max_n, dict2[max_n])
#         # print(dict2)
#     print()

import pickle5 as pickle
import pandas as pd
import numpy as np

def return_max_n(filename):
    with open(filename, 'rb') as file:
        result = pickle.load(file)
    max_n = []
    for graph_type, dict1 in result.items():
        temp = []
        for edge_density, dict2 in dict1.items():
            n = max(dict2, key=int)
            temp.append(n)
        max_n.extend(temp)
    return max_n


c1 = return_max_n('results/nonfaulty_bte.pickle')
c2 = return_max_n('results/nonfaulty_qte.pickle')
c3 = return_max_n('results/faulty1_bte.pickle')
c4 = return_max_n('results/faulty1_qte.pickle')
c5 = return_max_n('results/faulty2_bte.pickle')
c6 = return_max_n('results/faulty2_qte.pickle')
c7 = return_max_n('results/nonfaulty_minorminer.pickle')
c8 = return_max_n('results/faulty1_minorminer.pickle')
c9 = return_max_n('results/faulty2_minorminer.pickle')
c10 = return_max_n('results/nonfaulty_coa.pickle')
# non_faulty_template = np.array([c1, c2]).transpose()
# print(non_faulty_template)
# non_faulty_template_df = pd.DataFrame(non_faulty_template, columns=['bipartite', 'quadripartite'])
# print(non_faulty_template_df.to_latex(index=False))

# faulty_template = np.array([c3, c4, c5, c6]).transpose()
# # for row in faulty_template:
# #     print(row[3])
#
minorminer = np.array([c10]).transpose()
for row in minorminer:
    print(row[0])

