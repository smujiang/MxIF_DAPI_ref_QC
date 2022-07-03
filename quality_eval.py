import numpy as np
import os
import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
import matplotlib.pyplot as plt
from utils import get_SSIM_array, get_dapis_for_a_ROI, get_dapis_std, get_staining_orders, get_overall_annotations

# FOVs with halo-like artifacts
halo_art = ["region_28", "region_36", "region_40", "region_46", "region_134", "region_135",
            "region_145", "region_147", "region_150", "region_159", "region_164", "region_165",
            "region_166", "region_202", "region_203", "region_216", "region_219", "region_220",
            "region_226", "region_227", "region_241", "region_249", "region_250", "region_257",
            "region_270", "region_273", "region_285", "region_301", "region_325"]
# FOVs with tissue missing
FOV_miss = ["region_001", "region_003", "region_011", "region_045", "region_049", "region_073",
            "region_074", "region_075", "region_076", "region_077", "region_078", "region_081",
            "region_100", "region_112", "region_119", "region_121", "region_122", "region_123",
            "region_140", "region_143", "region_145", "region_146", "region_162", "region_188",
            "region_198", "region_237", "region_243", "region_244", "region_246", "region_248",
            "region_253", "region_259", "region_262", "region_282", "region_288", "region_289",
            "region_294", "region_302", "region_303", "region_305", "region_308", "region_309",
            "region_317", "region_318", "region_327", "region_328", "region_331", "region_336",
            "region_340"]

def load_annotation(FOV_N):
    if FOV_miss:
        FOV_miss_binary = np.zeros((len(FOV_N), 1), int)
        for i in FOV_miss:
            ele = i.split("_")[1]
            roi_id = int(ele)-1
            if roi_id < len(FOV_N):
                FOV_miss_binary[roi_id] = 1
    else:
        FOV_miss_binary = None

    if halo_art:
        halo_art_binary = np.zeros((len(FOV_N), 1), int)
        # halo_art_binary = FOV_miss_binary
        for i in halo_art:
            ele = i.split("_")[1]
            roi_id = int(ele) - 1
            if roi_id < len(FOV_N):
                halo_art_binary[roi_id] = 1
    else:
        halo_art_binary = None

    return FOV_miss_binary, halo_art_binary

def eval_tissue_damage(pickle_dir, ROI_range, FOV_miss_binary, vis_dir):
    fit_errs = []
    for r in ROI_range:  # regions range
        fn = os.path.join(pickle_dir, "dapi_ssim_array", "ssim_array_region" + str(r) + ".pickle")
        fp = open(fn, 'rb')
        ssim_array = pickle.load(fp)
        m = ssim_array.shape[0]
        idx = (np.arange(1, m + 1) + (m + 1) * np.arange(m - 1)[:, None]).reshape(m, -1)
        ssim_array = ssim_array.ravel()[idx]

        vec_DAPI_SSIM_avg = np.mean(ssim_array, axis=1)
        fit_err = np.sum(1 - vec_DAPI_SSIM_avg)
        fit_errs.append(fit_err)
    roc_threshold_range = np.arange(start=min(fit_errs), stop=max(fit_errs), step=0.1)
    fpr_list = []
    tpr_list = []
    precision_list = []
    recall_list = []
    fscore_list = []
    for i in roc_threshold_range:
        selected_r_idx = (np.array(fit_errs) > i) * 1
        precision, recall, fscore, support = precision_recall_fscore_support(selected_r_idx, FOV_miss_binary, average='macro')
        fpr, tpr, thr = roc_curve(selected_r_idx, FOV_miss_binary)
        fpr_list.append(fpr[1])
        tpr_list.append(tpr[1])
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(fscore)
    fpr_list.append(1)
    tpr_list.append(1)

    # locate the index of the largest F1 score
    iz = np.argmax(np.array(fscore_list))
    best_value_str = 'Best option\n precision=%.3f\n recall=%.3f\n F1=%.3f' % (precision_list[iz], recall_list[iz], fscore_list[iz])
    str_wrt = 'Best Threshold=%.3f' % (roc_threshold_range[iz])
    print(str_wrt)
    best_threshold_ssim_fn = os.path.join(pickle_dir, "best_ssim_threshold.pickel")
    with open(best_threshold_ssim_fn, "wb") as fp:
        pickle.dump(roc_threshold_range[iz], fp)

    plt.figure()
    plt.plot(roc_threshold_range, precision_list, color="r", lw=2, label="Precision")
    plt.plot(roc_threshold_range, recall_list, color="g", lw=2, label="Recall")
    plt.plot(roc_threshold_range, fscore_list, color="b", lw=2, label="F1")
    plt.text(3, 0.45, best_value_str)
    plt.xlabel("SSIM accumulated error")
    plt.ylabel("Value")
    plt.title("Precision, Recall and F1 Score: tissue off")
    plt.legend(loc="lower right")
    save_to = os.path.join(vis_dir, "tissue_off_PRF.png")
    plt.savefig(save_to)

    # gmeans = np.sqrt(np.array(tpr_list)**2 + (1-np.array(fpr_list))**2)
    # ix = np.argmax(gmeans) # locate the index of the largest g-mean
    # plt.figure()
    # plt.plot(fpr_list, tpr_list, color="darkorange", lw=2, label="ROC curve")
    # plt.scatter(fpr_list[ix], tpr_list[ix], color="blue", lw=4)
    # plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # plt.text(0.2, 0.85, str_wrt)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic: tissue missing")
    # plt.legend(loc="lower right")
    # save_to = os.path.join(vis_dir, "tissue_missing_ROC.png")
    # plt.savefig(save_to)

def det_tissue_damage(best_threshold, FOVs, qc_out_dir, Aligned_img_dir, N_iteration):
    results = []
    vec_DAPI_SSIM_avg_list = []
    for FOV in FOVs:
        fn = os.path.join(qc_out_dir, "ssim_array_region" + str(FOV) + ".pickle")
        if os.path.exists(fn):
            fp = open(fn, 'rb')
            ssim_array = pickle.load(fp)
        else:
            ssim_array = get_SSIM_array(FOV, Aligned_img_dir, N_iteration, qc_out_dir)

        m = ssim_array.shape[0]
        idx = (np.arange(1, m + 1) + (m + 1) * np.arange(m - 1)[:, None]).reshape(m, -1)
        ssim_array = ssim_array.ravel()[idx]
        vec_DAPI_SSIM_avg = np.mean(ssim_array, axis=1)
        vec_DAPI_SSIM_avg_list.append(vec_DAPI_SSIM_avg)
        fit_err = np.sum(1 - vec_DAPI_SSIM_avg)
        if fit_err < best_threshold:  # if there is no tissue off
            results.append(False)
        else:
            print("Region %d tissue damage detected" % FOV)
            results.append(True)
    return results, vec_DAPI_SSIM_avg_list

def get_low_quality_rounds(fov_id, dapi_ssim_result, vec_DAPI_SSIM_avg_list, dapi_std_result, threshold=0.7):
    # print(dapi_ssim_result)
    # print(dapi_std_result)
    # print(fov_id)
    if dapi_ssim_result[fov_id-1] or dapi_std_result[fov_id-1]:
        rr = np.where(vec_DAPI_SSIM_avg_list[fov_id-1] > threshold)[0]
        ssim_score = np.array(vec_DAPI_SSIM_avg_list[fov_id-1])[rr]
        return rr, ssim_score
    else:
        return None, None

def get_marker_txt_name_by_round(panel, round_list):
    marker_txt = []
    if round_list is not None:
        for r in round_list:
            round_key = "S{:03d}".format(r)
            if round_key in panel.keys():
                markers = panel[round_key]
                marker_txt.append(markers)
            else:
                print("not a marker round")
    return marker_txt

def eval_halo_artifacts(QC_out_dir, ROIs, halo_art_binary, vis_dir):
    halo_anno = []
    std_img_avg_list = []
    for r in ROIs:
        fn = os.path.join(QC_out_dir, "dapi_std_img", str(r) + ".pickle")
        f = open(fn, 'rb')
        std_img = pickle.load(f)
        std_img_avg_list.append(np.average(std_img))
        halo_anno.append(halo_art_binary[r - 1])

    roc_threshold_range = np.arange(min(std_img_avg_list), max(std_img_avg_list), 0.1)
    fpr_list = []
    tpr_list = []
    precision_list = []
    recall_list = []
    fscore_list = []
    for i in roc_threshold_range:
        selected_r_idx = (np.array(std_img_avg_list) > i) * 1
        precision, recall, fscore, support = precision_recall_fscore_support(selected_r_idx, np.array(halo_anno),
                                                                             average='macro')
        fpr, tpr, _ = roc_curve(selected_r_idx, np.array(halo_anno))
        fpr_list.append(fpr[1])
        tpr_list.append(tpr[1])
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(fscore)

    fpr_list.append(1)
    tpr_list.append(1)
    ########################################################################################################
    iz = np.argmax(np.array(fscore_list))
    best_value_str = 'Best option: \n precision=%.3f \n recall=%.3f, \n F1=%.3f' % (precision_list[iz], recall_list[iz], fscore_list[iz])
    best_threshold = roc_threshold_range[iz]
    print('Best Threshold=%.3f' % best_threshold)
    best_threshold_fn = os.path.join(QC_out_dir, "best_dapi_std_threshold.pickel")
    with open(best_threshold_fn, "wb") as fp:
        pickle.dump(roc_threshold_range[iz], fp)

    plt.figure()
    plt.plot(roc_threshold_range, precision_list, color="r", lw=2, label="Precision")
    plt.plot(roc_threshold_range, recall_list, color="g", lw=2, label="Recall")
    plt.plot(roc_threshold_range, fscore_list, color="b", lw=2, label="F1")
    plt.text(5, 0.45, best_value_str)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel("DAPI std image average")
    plt.ylabel("Value")
    plt.title("Precision, Recall and F1 Score: halo_artifacts")
    plt.legend(loc="lower right")
    save_to = os.path.join(vis_dir, "halo_artifacts_PRF.png")
    plt.savefig(save_to)

    # gmeans = np.sqrt(np.array(tpr_list)**2 + (1-np.array(fpr_list))**2)
    # ix = np.argmax(gmeans)
    # plt.figure()
    # plt.plot(fpr_list, tpr_list, color="darkorange", lw=2, label="ROC curve")
    # plt.scatter(fpr_list[ix], tpr_list[ix], color="blue", lw=4)
    # plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # plt.text(0.2, 0.85, str_wrt)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic: halo_artifacts")
    # plt.legend(loc="lower right")
    # save_to = os.path.join(vis_dir, "halo_artifacts_ROC.png")
    # plt.savefig(save_to)
    return best_threshold

def det_halo_artifacts(best_threshold_dapi_std, FOVs, QC_out_dir, Aligned_img_dir, N_range):
    FOV_results = []
    Round_results = []
    for FOV in FOVs:
        fn = os.path.join(QC_out_dir, "dapi_std_img", str(FOV) + ".pickle")
        if os.path.exists(fn):
            fp = open(fn, 'rb')
            std_img = pickle.load(fp)
        else:
            dapi_imgs = get_dapis_for_a_ROI(FOV, Aligned_img_dir, N_range, gray_scale=(0, 255))
            std_img = get_dapis_std(dapi_imgs, QC_out_dir, FOV)  # save dapi std

        # Round_results.append(std_img > best_threshold_dapi_std)
        std_img_avg = np.average(std_img)
        if std_img_avg < best_threshold_dapi_std:  # if there is no artifact
            FOV_results.append(False)
        else:
            print("Region %d halo-like artifacts detected" % FOV)
            FOV_results.append(True)
    return FOV_results, Round_results


def create_QC_stat_vis(FOV_results, Round_results, anno_df, panel_design, output_dir):
    meaningful_rounds, markers_in_rounds = get_staining_orders(panel_design)
    QC_pass_scores = []
    QC_fail_scores = []
    for idx_f_r, f_r in enumerate(FOV_results):
        if f_r:
            for idx_rr, rr in enumerate(Round_results[idx_f_r]):
                if idx_rr+2 in meaningful_rounds:
                    anno_txt_list, anno_scores = get_overall_annotations(anno_df, panel_design, f_r, idx_rr + 2)
                    if rr:
                        QC_pass_scores.append(anno_scores)
                    else:
                        QC_fail_scores.append(anno_scores)

    plt.figure(1, dpi=300)
    plt.hist(QC_pass_scores, bins=4, histtype="step")
    plt.hist(QC_fail_scores, bins=4, histtype="step")
    plt.grid()
    plt.legend(["QC_pass", "QC_fail"])
    save_to = os.path.join(output_dir, "QC_stat_hist.png")
    plt.savefig(save_to)

