import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# group ROIs according to annotations, compare cell distributions between different groups
xls_fn = "/infodev1/non-phi-data/junjiang/MxIF/OVCA_TMA22/OVTMA_MarkerQCAnnotations_scores_FromJun_upgraded_04_13_2022.xlsx"
df = pd.read_excel(open(xls_fn, 'rb'), skiprows=[0])
df = df.set_index('Position')

sns.clustermap(df, row_cluster=True, col_cluster=False)
plt.show()

# sns.clustermap(df, standard_scale=1, row_cluster=True, col_cluster=False)
# plt.show()
#
# sns.clustermap(df, z_score=1, row_cluster=True, col_cluster=False)
# plt.show()

from sklearn.cluster import KMeans
q_arr = np.array(df.iloc[:, 1:]).astype(float)
q_arr_norm = q_arr/4.0
kmeans = KMeans(n_clusters=5, random_state=0).fit(q_arr_norm)
pos_labels = kmeans.labels_
roi_id_list = range(1, 349)
roi_groups = []
for i in set(pos_labels):
    idx = np.nonzero(pos_labels == i)[0]
    roi_groups.append([roi_id_list[r_id] for r_id in idx])






# group marker qualities, investigate if markers at the same round could be more likely to associate with each other
sns.clustermap(df, row_cluster=False, col_cluster=True)
plt.show()

K = 9
kmeans = KMeans(n_clusters=K, random_state=0).fit(np.transpose(q_arr_norm))
pos_labels = kmeans.labels_
marker_list = df.keys()
marker_groups = []
for i in set(pos_labels):
    idx = np.nonzero(pos_labels == i)[0]
    marker_groups.append([marker_list[r_id] for r_id in idx])

print("Debug")

















