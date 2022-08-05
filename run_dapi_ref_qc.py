import multiprocessing
import os

import argparse
import numpy as np
import json
from quality_eval import det_halo_artifacts, det_tissue_damage, get_low_quality_rounds, get_marker_txt_name_by_round
from utils import get_panel_design, get_SSIM_array, get_SSIM_array_from_dapi, plot_SSIM_array, \
    get_dapis_for_a_ROI, get_dapis_std, plot_dapi_thumbnails, plot_dapi_std, \
    pretty_marker_name_list, get_FOV_count, get_iteration_count

html_str_css = """
<style>
table.minimalistBlack {
border: 3px solid #000000;
text-align: center;
border-collapse: collapse;
}
table.minimalistBlack td, table.minimalistBlack th {
border: 1px solid #000000;
padding: 5px 8px;
}
table.minimalistBlack tbody td {
font-size: 15px;
}
table.minimalistBlack thead {
background: #CFCFCF;
background: -moz-linear-gradient(top, #dbdbdb 0%, #d3d3d3 66%, #CFCFCF 100%);
background: -webkit-linear-gradient(top, #dbdbdb 0%, #d3d3d3 66%, #CFCFCF 100%);
background: linear-gradient(to bottom, #dbdbdb 0%, #d3d3d3 66%, #CFCFCF 100%);
border-bottom: 3px solid #000000;
}
table.minimalistBlack thead th {
font-size: 19px;
font-weight: bold;
color: #000000;
text-align: left;
}
table.minimalistBlack tfoot td {
font-size: 14px;
}
</style>
"""

def generateHTMLTable(tissue_off_results, vec_DAPI_SSIM_avg_list, FOV_artifacts_results, panel_des):
    htmlList = ["</br><div><h3>DAPI referenced evaluation</h3>", "<table class=\"minimalistBlack\">",
                "<tr><th>Region ID</th><th>Tissue Damage</th><th>Artifacts</th><th>Notes</th><th>DAPI thumbnails</th><th>DAPI SSIM array</th><th>DAPI std image</th></tr>"]
    for idx, r in enumerate(tissue_results):
        low_q_round, scores = get_low_quality_rounds(idx+1, tissue_off_results, vec_DAPI_SSIM_avg_list, FOV_artifacts_results)
        recom_txt = ""
        DAPI_thumbnails = "<img style='max-width: 250px;' src='" + os.path.join("./DAPI_thumbnails", "ROI_" + str(idx+1) + "_all_dapi.png") + "'>"
        dapi_ssim_array = "<img style='max-width: 250px;' src='" + os.path.join("./dapi_ssim_array", "ssim_img_region" + str(idx+1) + ".png") + "'>"
        std_img = "<img style='max-width: 250px;' src='" + os.path.join("./dapi_std_img", "std_img_" + str(idx+1) + ".png") + "'>"
        if r:
            tissue_damage_indicator = "<p style='color:red'>Detected</p>"
            recom_txt += "<p> Suspect tissue damage.</p>"
        else:
            tissue_damage_indicator = "undetected"
        if FOV_artifacts_results[idx]:
            artifacts_indicator = "<p style='color:red'>Detected</p>"
            recom_txt += "<p> Suspect artifact.</p>"
        else:
            artifacts_indicator = "undetected"
        if r or FOV_artifacts_results[idx]:
            marker_txt = get_marker_txt_name_by_round(panel_des, low_q_round)
            recom_txt += "<p> Low quality markers: " + pretty_marker_name_list(marker_txt) + "</p>"

        wrt_row = "<tr><th>%d</th><th>%s</th><th>%s</th><th>%s</th><th>%s</th><th>%s</th><th>%s</th></tr>" % (idx+1, tissue_damage_indicator, artifacts_indicator, recom_txt, DAPI_thumbnails, dapi_ssim_array, std_img)
        # wrt_row = "<tr><th>%d</th><th>%s</th><th>%s</th><th>%s</th></tr>" % (idx, tissue_damage_indicator, artifacts_indicator, recom_txt)
        htmlList.append(wrt_row)
    htmlList.append("</table></div>")
    return "\n".join(htmlList)

def create_json_obj(tissue_off_results, vec_DAPI_SSIM_avg_list, FOV_artifacts_results, panel_des):
    result_dic = []
    for idx, r in enumerate(tissue_off_results):
        low_q_round, scores = get_low_quality_rounds(idx + 1, tissue_off_results, vec_DAPI_SSIM_avg_list,
                                                     FOV_artifacts_results)
        if r:
            tissue_damage_indicator = "True"
        else:
            tissue_damage_indicator = "False"
        if FOV_artifacts_results[idx]:
            artifacts_indicator = "True"
        else:
            artifacts_indicator = "False"
        if r or FOV_artifacts_results[idx]:
            marker_txt = get_marker_txt_name_by_round(panel_des, low_q_round)
        else:
            marker_txt = ""

        fov_result = {
            "FOV_ID": str(idx+1),
            "tissue_damage_indicator": tissue_damage_indicator,
            "artifacts_indicator": artifacts_indicator,
            "low_quality_markers": str(marker_txt)
        }
        result_dic.append(fov_result)
    json_object = json.dumps(result_dic, indent=4)
    return json_object

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--aligned_img_dir",
    #                     required=True,
    #                     dest='Aligned_img_dir',
    #                     help="Aligned image directory")
    #
    # parser.add_argument("-c", "--case_id",
    #                     required=True,
    #                     dest="case_id",
    #                     help="Case ID, Type: string")
    #
    # parser.add_argument("-o", "--output_dir",
    #                     default=os.getcwd(),
    #                     dest='output_dir',
    #                     help="Metrics output directory")
    #
    # args = parser.parse_args()
    # img_base_dir = args.Aligned_img_dir
    # case_ID = args.case_id
    # out_base_Dir = args.output_dir
    #

    # img_base_dir = "/research/bsi/archive/PI/Goode_Ellen_m004290/tertiary/s302493.MxIF_Ovarian_Cancer/integrated/OVCA_TMA22_Pilot"
    # case_ID = "OVCA_TMA22"
    # img_base_dir = "/research/bsi/archive/PI/Markovic_Svetomir_snm02/tertiary/s210155.CellSegmentation/integrated/SLN_Maus_June2019"
    # case_ID = "SLN3"

    img_base_dir = "/research/bsi/archive/PI/Markovic_Svetomir_snm02/tertiary/s210155.CellSegmentation/integrated/MelanomaLN_BMSAim1_Batch1"
    case_ID = "Mel30_BMS"


    out_base_Dir = "/research/bsi/projects/staff_analysis/m192500/MxIF_CellSeg/OME_TIFF/QC_out"

    Aligned_img_dir = os.path.join(img_base_dir, case_ID, "RegisteredImages")
    qc_out_dir = os.path.join(out_base_Dir, case_ID, "DAPI_QC")
    if not os.path.exists(qc_out_dir):
        os.makedirs(qc_out_dir)

    # N_FOVs = 348  # TODO: uncomment to debug
    # N_iter = 30  # TODO: uncomment to debug
    N_FOVs = get_FOV_count(Aligned_img_dir)  # TODO: uncomment to release
    N_iter = get_iteration_count(Aligned_img_dir)  # TODO: uncomment to release

    range_FOVs = range(1, N_FOVs)
    range_iter = range(2, N_iter+2)
    print(list(range_iter))
    print("Number of FOVs:%d" % N_FOVs)
    print("Number of imaging iterations:%d" % (N_iter+2))

    def pre_report(roi):
        print("\t processing ROI %d" % roi)
        dapi_imgs = get_dapis_for_a_ROI(roi, Aligned_img_dir, range_iter)  # get all DAPI images
        plot_dapi_thumbnails(dapi_imgs, roi, qc_out_dir)  # create DAPI thumbnails

        ssim_array = get_SSIM_array(roi, Aligned_img_dir, range_iter, qc_out_dir)
        # ssim_array = get_SSIM_array_from_dapi(dapi_imgs, roi, range_iter, qc_out_dir)  # save DAPI SSIM to pickle file
        plot_SSIM_array(ssim_array, roi, range_iter, qc_out_dir)  # create SSIM array heatmaps

        std_img = get_dapis_std(dapi_imgs, qc_out_dir, roi)  # save DAPI std to pickle file
        plot_dapi_std(std_img, roi, qc_out_dir)

    ########################################################
    # calculate metrics and create images
    ########################################################
    print("Creating metrics and images")
    # Calculate QC metrics: SSIM and std (using multiprocessing)
    # a_pool = multiprocessing.Pool(32)
    # a_pool.map(pre_report, range_FOVs)

    # Calculate QC metrics: SSIM and std (using single thread)
    for roi in range_FOVs:
        print("\t processing ROI %d" % roi)

        dapi_imgs = get_dapis_for_a_ROI(roi, Aligned_img_dir, range_iter)  # get all DAPI images
        plot_dapi_thumbnails(dapi_imgs, roi, qc_out_dir)  # create DAPI thumbnails

        ssim_array = get_SSIM_array(roi, Aligned_img_dir, range_iter, qc_out_dir)
        # ssim_array = get_SSIM_array_from_dapi(dapi_imgs, roi, range_iter, qc_out_dir)  # save DAPI SSIM to pickle file
        plot_SSIM_array(ssim_array, roi, range_iter, qc_out_dir)  # create SSIM array heatmaps

        std_img = get_dapis_std(dapi_imgs, qc_out_dir, roi)  # save DAPI std to pickle file
        plot_dapi_std(std_img, roi, qc_out_dir)

    ########################################################
    # get detection results
    ########################################################
    tissue_dam_threshold = 4.253  # default values
    artifacts_threshold = 7.77  # default values
    tissue_results, vec_DAPI_SSIM_avg_list = det_tissue_damage(tissue_dam_threshold, range_FOVs, qc_out_dir, Aligned_img_dir, range_iter)
    FOV_results, _ = det_halo_artifacts(artifacts_threshold, range_FOVs, qc_out_dir, Aligned_img_dir, range_iter)

    import matplotlib.pyplot as plt
    arr_DAPI_SSIM = np.array(vec_DAPI_SSIM_avg_list)
    all_ssim_avg = []
    for r in range(28):
        round_ssim = arr_DAPI_SSIM[:, r]
        # hist_round_ssim = np.histogram(round_ssim)
        all_ssim_avg.append(round_ssim)
    plt.hist(np.array(all_ssim_avg).flatten())
    plt.xlim(-0.2, 1.2)
    print(np.std(np.array(all_ssim_avg).flatten()))
    print(np.mean(np.array(all_ssim_avg).flatten()))
    print(np.median(np.array(all_ssim_avg).flatten()))
    print(np.min(np.array(all_ssim_avg).flatten()))
    print(np.max(np.array(all_ssim_avg).flatten()))
    plt.ylim(0, 3000)
    plt.savefig("all_ssim.png")
    plt.close()


    ########################################################
    # create html reports
    ########################################################
    panel_des = get_panel_design(os.path.join(img_base_dir, case_ID), qc_out_dir)

    print("Creating html reports")
    Html_Out_fn = os.path.join(qc_out_dir, case_ID + "_dapi_ref_qc_report.html")

    Html_Out_file = open(Html_Out_fn, "w")
    Html_Out_file.write(html_str_css)

    h1 = generateHTMLTable(tissue_results, vec_DAPI_SSIM_avg_list, FOV_results, panel_des)
    Html_Out_file.write(h1)
    Html_Out_file.close()

    ########################################################
    # create json file
    ########################################################
    print("Creating json file")
    json_object = create_json_obj(tissue_results, vec_DAPI_SSIM_avg_list, FOV_results, panel_des)
    json_fn = os.path.join(qc_out_dir, case_ID + "_dapi_ref_qc_result.json")
    with open(json_fn, "w") as outfile:
        outfile.write(json_object)



