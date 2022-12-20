import os, glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import umap
from utils import read_dapi
from PIL import Image, ImageDraw



data_dir = "/infodev1/non-phi-data/junjiang/MxIF/OVCA_TMA22/cell_mesurements"
# data_dir = "\\\\mfad\\researchmn\\HCPR\HCPR-GYNECOLOGICALTUMORMICROENVIRONMENT\\Multiplex_Img\\OVCA_TMA22\\cell_mesurements"
fn_list = ["OVTMA_0_Quant.tsv", "OVTMA_100_Quant.tsv", "OVTMA_200_Quant.tsv", "OVTMA_300_Quant.tsv"]

img_dir = "/infodev1/non-phi-data/junjiang/MxIF/OVCA_TMA22/OME_TIFF_Images"
output_dir = "/infodev1/non-phi-data/junjiang/MxIF/OVCA_TMA22/umap_cluster_cell_features"
img_output_dir = "/infodev1/non-phi-data/junjiang/MxIF/OVCA_TMA22/cell_loc_imgs"
eval_img_output_dir = "/infodev1/non-phi-data/junjiang/MxIF/OVCA_TMA22/cell_distributions"

roi_id_range = range(1, 349)

point_size = 5  # size of labels that show the locations of cells on DAPI
show_dapi_in = "gray"  # or "red"
# show_dapi_in = "red"  #

img_id = "OVCA_TMA22_region_" + "{:03d}".format(19) + ".ome.tiff" # suppose ROI#19 is one of the case with best imaging quality
print("Processing %s" % img_id)
fn = os.path.join(data_dir, fn_list[0])
df = pd.read_csv(fn, sep="\t")
case_df = df[df["Image"].str.contains(img_id)]
cell_locations = np.array(case_df.iloc[:, 5:7]).astype(float)
cell_features_orig = np.array(case_df.iloc[:, 7:]).astype(float)
cell_features = cell_features_orig / cell_features_orig.max(axis=0)
cell_features = np.nan_to_num(cell_features)

# dm_red = PCA(n_components=2)
dm_red = umap.UMAP()
pca_cell_f = dm_red.fit_transform(cell_features)

# class 1
c1_cells = cell_features[pca_cell_f[:, 0] >= 0.0]
dm_red_c1 = umap.UMAP()
dm_red_c1.fit(c1_cells)
# class 2
c2_cells = cell_features[pca_cell_f[:, 0] < 0.0]
dm_red_c2 = umap.UMAP()
dm_red_c2.fit(c2_cells)

# ROIs with tissue damage
low_quality_roi_id_list_tissue_damage = [1, 3, 11, 40, 45, 49, 73, 74, 75, 76, 77, 78, 81,
                           100, 112, 119, 121, 122, 123, 140, 141, 143, 145, 146, 162,
                           188, 198, 237, 243, 244, 246, 248, 253, 259, 262, 282, 288,
                           289, 294, 302, 303, 305, 308, 309, 317, 318, 327, 328, 331,
                           336, 340]
# ROIs with artifacts
low_quality_roi_id_list_artifacts = [3, 6, 7, 21, 25, 28, 36, 37, 40, 46, 51, 62,
                                     65, 73, 74, 75, 76, 77, 78, 86, 97, 105, 114,
                                     134, 145, 147, 159, 164, 165, 166, 174, 188, 202,
                                     203, 212, 216, 219, 220, 226,227, 241, 249, 250,
                                     257, 269, 270, 273, 282, 294, 301, 303, 325, 326]

# low_quality_roi_id_list = low_quality_roi_id_list_tissue_damage + low_quality_roi_id_list_artifacts

low_quality_roi_id_list = low_quality_roi_id_list_tissue_damage + low_quality_roi_id_list_artifacts

nan_locations_list = []

low_quality_cells_A = np.empty((0, 2), dtype=float)
low_quality_cells_C1 = np.empty((0, 2), dtype=float)
low_quality_cells_C2 = np.empty((0, 2), dtype=float)
high_quality_cells_A = np.empty((0, 2), dtype=float)
high_quality_cells_C1 = np.empty((0, 2), dtype=float)
high_quality_cells_C2 = np.empty((0, 2), dtype=float)

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
    tif_fn = os.path.join(img_dir, img_id)
    if os.path.exists(tif_fn):
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

        save_to = os.path.join(output_dir, "OVCA_TMA22_region_"+"{:03d}".format(roi_id) + "_cells_umap.png")
        if not os.path.exists(save_to):
            plt.scatter(pca_cell_f[:, 0], pca_cell_f[:, 1])
            plt.title("Cell features UMap: %s" % "OVCA_TMA22_region_"+"{:03d}".format(roi_id))
            plt.savefig(save_to)
            plt.close()

        # class 1
        c1_cells = cell_features[pca_cell_f[:, 0] >= 0.0]
        pca_cell_f1 = dm_red_c1.transform(c1_cells)
        save_to = os.path.join(output_dir, "OVCA_TMA22_region_" + "{:03d}".format(roi_id) + "_c1_cells_umap.png")
        if not os.path.exists(save_to):
            plt.scatter(pca_cell_f1[:, 0], pca_cell_f1[:, 1])
            plt.title("Cell features UMap: %s" % "OVCA_TMA22_region_" + "{:03d}".format(roi_id))
            plt.savefig(save_to)
            plt.close()

        c2_cells = cell_features[pca_cell_f[:, 0] < 0.0]
        pca_cell_f2 = dm_red_c2.transform(c2_cells)
        save_to = os.path.join(output_dir, "OVCA_TMA22_region_" + "{:03d}".format(roi_id) + "_c2_cells_umap.png")
        if not os.path.exists(save_to):
            plt.scatter(pca_cell_f2[:, 0], pca_cell_f2[:, 1])
            plt.title("Cell features UMap: %s" % "OVCA_TMA22_region_" + "{:03d}".format(roi_id))
            plt.savefig(save_to)
            plt.close()

        # if roi_id in low_quality_roi_id_list:
        #     low_quality_cells_A = np.append(low_quality_cells_A, pca_cell_f, axis=0)
        #     low_quality_cells_C1 = np.append(low_quality_cells_C1, pca_cell_f1, axis=0)
        #     low_quality_cells_C2 = np.append(low_quality_cells_C2, pca_cell_f2, axis=0)
        # else:
        #     high_quality_cells_A = np.append(high_quality_cells_A, pca_cell_f, axis=0)
        #     high_quality_cells_C1 = np.append(high_quality_cells_C1, pca_cell_f1, axis=0)
        #     high_quality_cells_C2 = np.append(high_quality_cells_C2, pca_cell_f2, axis=0)

        if roi_id not in low_quality_roi_id_list:
            high_quality_cells_A = np.append(high_quality_cells_A, pca_cell_f, axis=0)
            high_quality_cells_C1 = np.append(high_quality_cells_C1, pca_cell_f1, axis=0)
            high_quality_cells_C2 = np.append(high_quality_cells_C2, pca_cell_f2, axis=0)

        if roi_id in low_quality_roi_id_list_artifacts:
        # if roi_id in low_quality_roi_id_list_tissue_damage:
            low_quality_cells_A = np.append(low_quality_cells_A, pca_cell_f, axis=0)
            low_quality_cells_C1 = np.append(low_quality_cells_C1, pca_cell_f1, axis=0)
            low_quality_cells_C2 = np.append(low_quality_cells_C2, pca_cell_f2, axis=0)


        # show c1 cells and c2 cells on DAPI channel with different colors
        save_to = os.path.join(img_output_dir, "OVCA_TMA22_region_" + "{:03d}".format(roi_id) + "_cells_loc.png")
        if not os.path.exists(save_to):
            dapi_img_arr = read_dapi(tif_fn, scale_to=(0, 255)).astype(np.uint8)
            if show_dapi_in == "red":
                z = np.zeros(dapi_img_arr.shape).astype(np.uint8)
                dapi_img_arr = np.stack([dapi_img_arr, z, z], axis=2)
            else:
                dapi_img_arr = np.repeat(dapi_img_arr[:, :, np.newaxis], 3, axis=2)
            dapi_img = Image.fromarray(dapi_img_arr)
            dapi_img_draw = ImageDraw.Draw(dapi_img)
            c1_xy_list = cell_locations[pca_cell_f[:, 0] >= 0.0]
            for c1_xy in c1_xy_list:
                xy = [(c1_xy[0] - point_size, c1_xy[1] - point_size), (c1_xy[0] + point_size, c1_xy[1] + point_size)]
                dapi_img_draw.ellipse(xy, fill='green') # tumor
            c2_xy_list = cell_locations[pca_cell_f[:, 0] < 0.0]
            for c2_xy in c2_xy_list:   # stroma
                xy = [(c2_xy[0] - point_size, c2_xy[1] - point_size), (c2_xy[0] + point_size, c2_xy[1] + point_size)]
                dapi_img_draw.ellipse(xy, fill='blue')

            dapi_img.save(save_to)
            # plt.imshow(dapi_img)
            # plt.axis(False)
            # plt.show()

        # if roi_id > 5:
        #     break
        # print("debug")

#################################################################
print("Total cell count in HQ ROIs %d" % high_quality_cells_A.shape[0])
print("Tumor cell count in HQ ROIs %d" % high_quality_cells_C1.shape[0])
print("Stroma cell count in HQ ROIs %d" % high_quality_cells_C2.shape[0])
print("Total cell count in LQ ROIs %d" % low_quality_cells_A.shape[0])
print("Tumor cell count in LQ ROIs %d" % low_quality_cells_C1.shape[0])
print("Stroma cell count in LQ ROIs %d" % low_quality_cells_C2.shape[0])

#################################################################
save_to = os.path.join(output_dir, "OVCA_TMA22_region_low_quality_cells_umap.png")
if not os.path.exists(save_to):
    plt.scatter(low_quality_cells_A[:, 0], low_quality_cells_A[:, 1])
    plt.title("Cell features UMap")
    plt.savefig(save_to)
    plt.close()

# class 1
save_to = os.path.join(output_dir, "OVCA_TMA22_region_low_quality_cells_C1_umap.png")
if not os.path.exists(save_to):
    plt.scatter(low_quality_cells_C1[:, 0], low_quality_cells_C1[:, 1])
    plt.title("Cell features UMap")
    plt.savefig(save_to)
    plt.close()

# class 2
save_to = os.path.join(output_dir, "OVCA_TMA22_region_low_quality_cells_C2_umap.png")
if not os.path.exists(save_to):
    plt.scatter(low_quality_cells_C2[:, 0], low_quality_cells_C2[:, 1])
    plt.title("Cell features UMap")
    plt.savefig(save_to)
    plt.close()

####################################################
save_to = os.path.join(output_dir, "OVCA_TMA22_region_high_quality_cells_umap.png")
if not os.path.exists(save_to):
    low_quality_cell_cnt = low_quality_cells_A.shape[0]
    random_indices = np.random.choice(high_quality_cells_A.shape[0], size=low_quality_cell_cnt, replace=False)
    high_quality_cells_A = high_quality_cells_A[random_indices, :]
    plt.scatter(high_quality_cells_A[:, 0], high_quality_cells_A[:, 1])
    plt.title("Cell features UMap")
    plt.savefig(save_to)
    plt.close()

# class 1
save_to = os.path.join(output_dir, "OVCA_TMA22_region_high_quality_cells_C1_umap.png")
if not os.path.exists(save_to):
    low_quality_cell_cnt = low_quality_cells_C1.shape[0]
    random_indices = np.random.choice(high_quality_cells_C1.shape[0], size=low_quality_cell_cnt, replace=False)
    high_quality_cells_C1 = high_quality_cells_C1[random_indices, :]
    plt.scatter(high_quality_cells_C1[:, 0], high_quality_cells_C1[:, 1])
    plt.title("Cell features UMap")
    plt.savefig(save_to)
    plt.close()

save_to = os.path.join(output_dir, "OVCA_TMA22_region_high_quality_cells_C2_umap.png")
if not os.path.exists(save_to):
    low_quality_cell_cnt = low_quality_cells_C2.shape[0]
    random_indices = np.random.choice(high_quality_cells_C2.shape[0], size=low_quality_cell_cnt, replace=False)
    high_quality_cells_C2 = high_quality_cells_C2[random_indices, :]
    plt.scatter(high_quality_cells_C2[:, 0], high_quality_cells_C2[:, 1])
    plt.title("Cell features UMap")
    plt.savefig(save_to)
    plt.close()

print("debug")

from scipy.stats import gaussian_kde

##############################################
xy = np.vstack([high_quality_cells_C1[:, 0], high_quality_cells_C1[:, 1]])
kde_high_c1 = gaussian_kde(xy)(xy)
plt.scatter(high_quality_cells_C1[:, 0], high_quality_cells_C1[:, 1], c=kde_high_c1, marker=".", cmap='jet')
plt.grid()
plt.title("Tumor Cell Distributions: HQ ROIs")
plt.xlim([0, 10])
plt.ylim([5, 14.5])
plt.colorbar()
plt.axis('equal')
# plt.show()
save_to = os.path.join(eval_img_output_dir, "TumorCellDistribution_HQ.png")
plt.savefig(save_to)
plt.close()

#
xy = np.vstack([low_quality_cells_C1[:, 0], low_quality_cells_C1[:, 1]])
kde_low_c1 = gaussian_kde(xy)(xy)
plt.scatter(low_quality_cells_C1[:, 0], low_quality_cells_C1[:, 1], c=kde_low_c1, marker=".", cmap='jet')
plt.grid()
plt.title("Tumor Cell Distributions: LQ ROIs")
plt.xlim([0, 10])
plt.ylim([6, 14.5])
plt.colorbar()
plt.axis('equal')
# plt.show()
save_to = os.path.join(eval_img_output_dir, "TumorCellDistribution_LQ.png")
plt.savefig(save_to)
plt.close()

##############################################
xy = np.vstack([high_quality_cells_C2[:, 0], high_quality_cells_C2[:, 1]])
kde_high_c2 = gaussian_kde(xy)(xy)
plt.scatter(high_quality_cells_C2[:, 0], high_quality_cells_C2[:, 1], c=kde_high_c2, marker=".", cmap='jet')
plt.grid()
plt.title("Stroma Cell Distributions: HQ ROIs")
plt.xlim([0, 10])
plt.ylim([2, 12])
plt.colorbar()
plt.axis('equal')
# plt.show()
save_to = os.path.join(eval_img_output_dir, "StromaCellDistribution_HQ.png")
plt.savefig(save_to)
plt.close()


#
xy = np.vstack([low_quality_cells_C2[:, 0], low_quality_cells_C2[:, 1]])
kde_low_c2 = gaussian_kde(xy)(xy)
plt.scatter(low_quality_cells_C2[:, 0], low_quality_cells_C2[:, 1], c=kde_low_c2, marker=".", cmap='jet')
plt.grid()
plt.title("Stroma Cell Distributions: LQ ROIs")
plt.xlim([0, 10])
plt.ylim([2, 12])
plt.colorbar()
plt.axis('equal')
# plt.show()
save_to = os.path.join(eval_img_output_dir, "StromaCellDistribution_LQ.png")
plt.savefig(save_to)
plt.close()



print("Debug")


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






