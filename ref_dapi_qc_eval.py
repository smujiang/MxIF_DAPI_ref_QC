import multiprocessing
import os
import argparse
import pandas as pd
import pickle

from utils import get_panel_design, get_SSIM_array, plot_SSIM_array, \
    get_dapis_for_a_ROI, get_dapis_std, plot_dapi_thumbnails, plot_dapi_std, vis_marker_images, \
    plot_marker_quality_scatter, get_meaningful_round
from quality_eval import load_annotation, eval_tissue_damage, det_tissue_damage, eval_halo_artifacts, \
    det_halo_artifacts, create_QC_stat_vis

# parser = argparse.ArgumentParser()
# parser.add_argument("-aid", "--aid",
#                     required=True,
#                     dest='Aligned_img_dir',
#                     help="Aligned image directory")
#
# parser.add_argument("-c", "--cid",
#                     required=True,
#                     dest="case_id",
#                     help="Case ID, Type: string")
#
# parser.add_argument("-r", "--rois",
#                     required=True,
#                     dest="ROIs",
#                     help="ROI id list, Type: list/range")
#
# parser.add_argument("-N", "--N_iterations",
#                     required=True,
#                     dest="rounds",
#                     help="Total number of imaging iterations")
#
# parser.add_argument("-otd", "--otd",
#                     default=os.getcwd(),
#                     dest='otd',
#                     help="OME tiff file saving directory")
#
# parser.add_argument("-af", "--af",
#                     default="",
#                     dest='af',
#                     help="Annotation file name with full path")
#
# parser.add_argument("-od", "--output",
#                     default=os.getcwd(),
#                     dest='output',
#                     help="Metrics output directory")
#
# parser.add_argument("-vd", "--vis",
#                     default=os.getcwd(),
#                     dest='vis',
#                     help="Visualization output directory")
#
# args = parser.parse_args()
# Aligned_img_dir = args.aid
# case_id = args.case_id
# ROIs = args.ROIs
# N_it = args.rounds
# OME_TIFF_dir = args.otd
# annotation_fn = args.af
# QC_out_dir = os.path.join(args.output, case_id)  # output directory: save pickle files for arrays, variables.
# vis_dir = os.path.join(args.vis, case_id)  # output directory: save pictures/plots


# example:
Aligned_img_dir = "/research/bsi/archive/PI/Goode_Ellen_m004290/tertiary/s302493.MxIF_Ovarian_Cancer/integrated/OVCA_TMA22_Pilot/OVCA_TMA22/RegisteredImages"  # aligned image directory
case_id = "OVCA_TMA22"  # case ID
ROIs = range(1, 348)  # range/list of ROIs
# ROIs = range(1, 3)  # range/list of ROIs
N_range = range(2, 32)  # range of imaging iterations (skip the first round, as it's an overview of slide)
OME_TIFF_dir = "/research/bsi/projects/staff_analysis/m192500/MxIF_CellSeg/OVCA_TMA22/OME_TIFF_Images"  # directory to save OME.TIFF files
annotation_fn = "./OVTMA_MarkerQCAnnotations_FromJun_upgraded_04_08_2022.txt"  # annotation csv file name
output = "/research/bsi/projects/staff_analysis/m192500/MxIF_CellSeg/OME_TIFF/QC_out"
vis = "/research/bsi/projects/staff_analysis/m192500/MxIF_CellSeg/OME_TIFF/QC_vis"
QC_out_dir = os.path.join(output, case_id)  # output directory: save pickle files for arrays, variables.
vis_dir = os.path.join(vis, case_id)  # output directory: save pictures/plots


def calculate_all(roi):
    ssim_array = get_SSIM_array(roi, Aligned_img_dir, N_range, QC_out_dir)
    plot_SSIM_array(ssim_array, roi, N_range, vis_dir)  # create figures()

    dapi_imgs = get_dapis_for_a_ROI(roi, Aligned_img_dir, N_range, gray_scale=(0, 255))
    std_img = get_dapis_std(dapi_imgs, QC_out_dir, roi)  # save dapi std
    plot_dapi_thumbnails(dapi_imgs, roi, vis_dir)  # TODO: have issue, don't show?
    plot_dapi_std(std_img, roi, vis_dir)  # TODO: double check


if not os.path.exists(OME_TIFF_dir):
    print("TODO: create OME.TIFF file")
elif not os.path.exists(annotation_fn):
    print("Annotation not exists, some results may not generated.")
else:
    # get panel design
    int_img_dir = os.path.split(Aligned_img_dir)[0]
    panel_des = get_panel_design(int_img_dir, QC_out_dir)

    # Calculate QC metrics: SSIM and std (using multiprocessing)
    # a_pool = multiprocessing.Pool(4)
    # a_pool.map(calculate_all, ROIs)

    # Calculate QC metrics: SSIM and std
    # for roi in ROIs:
    #     print("processing ROI %d" % roi)
    #     ssim_array = get_SSIM_array(roi, Aligned_img_dir, N_range, QC_out_dir)
    #     plot_SSIM_array(ssim_array, roi, N_range, vis_dir)  # create figures()
    #
    #     dapi_imgs = get_dapis_for_a_ROI(roi, Aligned_img_dir, N_range, gray_scale=(0, 255))
    #     std_img = get_dapis_std(dapi_imgs, QC_out_dir, roi)  # save dapi std
    #     plot_dapi_thumbnails(dapi_imgs, roi, vis_dir)
    #     plot_dapi_std(std_img, roi, vis_dir)

    # load overall annotations and evaluate performance accordingly, or detect quality issues
    if os.path.exists(annotation_fn):
        # plot scatter plot based on annotation
        anno_df = pd.read_csv(annotation_fn, sep='\t')
        # plot_marker_quality_scatter(panel_des, ROIs, anno_df, QC_out_dir, vis_dir)

        # Evaluate or detect: 1) tissue damage; 2) halo artifacts
        FOV_miss_binary, halo_art_binary = load_annotation(ROIs)
        if FOV_miss_binary is not None:
            # evaluate and save the best threshold to file name
            eval_tissue_damage(QC_out_dir, ROIs, FOV_miss_binary, vis_dir)

        if halo_art_binary is not None:
            # evaluate and save the best threshold to file name
            best_threshold_dapi_std = eval_halo_artifacts(QC_out_dir, ROIs, halo_art_binary, vis_dir)
            FOV_results, Round_results = det_halo_artifacts(best_threshold_dapi_std, ROIs, QC_out_dir, Aligned_img_dir,
                                                            N_range)
            create_QC_stat_vis(FOV_results, Round_results, anno_df, panel_des, vis_dir)

        # visualize markers in the same round (only when the OMETIFF file is available)
        n_round, _ = get_meaningful_round(panel_des, N_range)  # skip the bleaching round
        for fov in ROIs:
            for r in n_round:
                out_dir = os.path.join(vis_dir, "Markers_per_round")
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                vis_marker_images(case_id, fov, r, panel_des, anno_df, OME_TIFF_dir, out_dir)

    else:
        anno_df = None
        best_threshold_ssim_fn = os.path.join(QC_out_dir, "best_ssim_threshold.pickel")  # the best threshold
        if not os.path.exists(best_threshold_ssim_fn):
            print("Don't have tissue damage annotation, use saved threshold to detect.")
            best_threshold_ssim = 4.253  # use hard code values
        else:
            fp = open(best_threshold_ssim_fn, "wb")
            best_threshold_ssim = pickle.load(fp)
        results, vec_DAPI_SSIM_avg_list = det_tissue_damage(best_threshold_ssim, ROIs, QC_out_dir, Aligned_img_dir, N_range)

        best_threshold_dapi_std_fn = os.path.join(QC_out_dir, "best_dapi_std_threshold.pickel")
        if not os.path.exists(best_threshold_dapi_std_fn):
            print("Don't have halo artifacts annotation, use saved threshold to detect.")
            best_threshold_dapi_std = 7.77  # use hard code values
        else:
            fp = open(best_threshold_dapi_std_fn, "wb")
            best_threshold_dapi_std = pickle.load(fp)
        FOV_results, Round_results = det_halo_artifacts(best_threshold_dapi_std, ROIs, QC_out_dir, Aligned_img_dir,
                                                        N_range)



