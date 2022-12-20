import os, glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

data_dir = "/infodev1/non-phi-data/junjiang/MxIF/OVCA_TMA22/cell_mesurements"
# data_dir = "\\\\mfad\\researchmn\\HCPR\HCPR-GYNECOLOGICALTUMORMICROENVIRONMENT\\Multiplex_Img\\OVCA_TMA22\\cell_mesurements"
fn_list = ["OVTMA_0_Quant.tsv", "OVTMA_100_Quant.tsv", "OVTMA_200_Quant.tsv", "OVTMA_300_Quant.tsv"]


output_dir = "/infodev1/non-phi-data/junjiang/MxIF/OVCA_TMA22/pca_cell_features"

roi_id_range = range(1, 349)





img_id = "OVCA_TMA22_region_" + "{:03d}".format(19) + ".ome.tiff"
print("Processing %s" % img_id)
fn = os.path.join(data_dir, fn_list[0])
df = pd.read_csv(fn, sep="\t")
case_df = df[df["Image"].str.contains(img_id)]
cell_locations = np.array(case_df.iloc[:, 5:7]).astype(float)
cell_features_orig = np.array(case_df.iloc[:, 7:]).astype(float)
cell_features = cell_features_orig / cell_features_orig.max(axis=0)
cell_features = np.nan_to_num(cell_features)

dm_red = PCA(n_components=2)
pca_cell_f = dm_red.fit(cell_features)

nan_locations_list = []

for roi_id in roi_id_range:
    if roi_id < 100:
        fn = os.path.join(data_dir, fn_list[0])
    elif 100 <= roi_id < 200:
        fn = os.path.join(data_dir, fn_list[1])
    elif 200 <= roi_id < 300:
        fn = os.path.join(data_dir, fn_list[2])
    else:
        fn = os.path.join(data_dir, fn_list[3])

    img_id = "OVCA_TMA22_region_"+"{:03d}".format(roi_id)+".ome.tiff"
    print("Processing %s" % img_id)

    df = pd.read_csv(fn, sep="\t")
    # df = pd.read_csv(fn, sep="\t", nrows=4)

    case_df = df[df["Image"].str.contains(img_id)]

    cell_locations = np.array(case_df.iloc[:, 5:7]).astype(float)
    cell_features_orig = np.array(case_df.iloc[:, 7:]).astype(float)


    # nan_locations_1 = np.isnan(cell_features_orig)*1
    # nan_dis_1 = np.sum(nan_locations_1, axis=0)
    # cell_features_sel = cell_features_orig[~np.isnan(cell_features_orig).any(axis=1), :] # exclude rows with NaN
    # a = cell_features_sel.min(axis=0)
    # b = cell_features_sel.ptp(axis=0)
    # cell_features = (cell_features_sel - cell_features_sel.min(axis=0)) / cell_features_sel.ptp(axis=0)
    # cell_features = cell_features_sel - a
    # nan_locations = np.isnan(cell_features) * 1
    # nan_dis = np.sum(nan_locations, axis=0)
    # # cell_features = cell_features[~np.isnan(cell_features).any(axis=0), :]
    # cell_features = cell_features[:, ~np.isnan(cell_features).any(axis=0)]

    # vector_norm = np.linalg.norm(cell_features_orig)
    # cell_features = cell_features_orig / vector_norm

    cell_features = cell_features_orig / cell_features_orig.max(axis=0)
    cell_features = np.nan_to_num(cell_features)
    # pca_cell_f = dm_red.fit_transform(cell_features)

    pca_cell_f = dm_red.transform(cell_features)
    # if roi_id == roi_id_range[0]:
    #     pca_cell_f = dm_red.fit_transform(cell_features)
    # else:
    #     pca_cell_f = dm_red.transform(cell_features)

    plt.scatter(pca_cell_f[:, 0], pca_cell_f[:, 1])
    plt.title("Cell features PCA: %s" % "OVCA_TMA22_region_"+"{:03d}".format(roi_id))
    save_to = os.path.join(output_dir, "OVCA_TMA22_region_"+"{:03d}".format(roi_id) + "_cells_pca.png")
    plt.savefig(save_to)
    plt.close()
    print("debug")


#
# labels_txt = ["others", "cd_8"]
# pathology_labels = [0, 1]
# color_map = ['r', 'b']
#
#
#
# # cell distribution, including samples with quality issues
# X_all = []
# # dm_rd = TSNE(n_components=2, learning_rate=200.0, verbose=1, init='random')
# dm_rd = PCA(n_components=2)
# pca_embedded_all = dm_rd.fit(X_all)
#
#
#
# high_q_ind = [1, 2]
#
# # cell distribution, without samples have quality issues
# X = [i for i in X_all[high_q_ind, :]]
# pca_embedded = dm_rd.fit_transform(X)
#
#
#
# plt.figure(1)
# for idx, pl in enumerate(pathology_labels):
#     label_idx = [i == pl for i in labels_txt]
#     plt.scatter(pca_embedded_all[label_idx, 0], pca_embedded_all[label_idx, 1], c=color_map[idx], label=pl)
# plt.title("Cell scatters (With low quality ROIs)")
# plt.legend()
# plt.show()
#
# plt.figure(2)
# for idx, pl in enumerate(pathology_labels):
#     label_idx = [i == pl for i in labels_txt]
#     plt.scatter(pca_embedded[label_idx, 0], pca_embedded[label_idx, 1], c=color_map[idx], label=pl)
#
# plt.title("Cell scatters (Without low quality ROIs)")
# plt.legend()
# plt.show()
#






