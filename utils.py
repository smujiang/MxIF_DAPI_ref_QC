import os, sys, io
import numpy
import numpy as np
from glob import glob
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
import matplotlib as mpl
import tifffile as tf
from PIL import Image, ImageDraw, ImageEnhance
import xml.etree.ElementTree
from vis_utils import autoSetDisplayRange

def get_FOV_count(Aligned_img_case_dir):
    return len(glob(os.path.join(Aligned_img_case_dir, 'S002', "*mono_dapi_*.tif")))

def get_iteration_count(Aligned_img_case_dir):
    return len(glob(os.path.join(Aligned_img_case_dir, "*"))) - 1

def get_panel_design(Aligned_img_dir, panel_save_dir):
    save_to = os.path.join(panel_save_dir, "panel_design.pickle")
    if os.path.exists(save_to):
        f = open(save_to, 'rb')
        panel = pickle.load(f)
    else:
        panel = {}
        panelDirs = glob(os.path.join(Aligned_img_dir, "*_dapi_dapi_*"))
        if len(panelDirs) < 1:
            print("No Scans panel!")
            sys.exit()
        for dd in panelDirs:
            nom = os.path.basename(dd)
            if "HLA" in nom:
                nom = nom.replace('HLA_', 'HLA-')
            if "BMS" in nom:
                nom = nom.replace('_BMS', '-BMS')
            if "Gores" in nom:
                nom = nom.replace('_Gores', '-Gores')
            if "Caspase" in nom:
                nom = nom.replace('Caspase_', 'Caspase-')
            if "isotype" in nom:
                nom = nom.replace('isotype_', 'isotype-')
            dbase = nom.split("_")
            # pprint(dbase)

            if dbase[4] == "fitc":
                panel[dbase[0]] = [dbase[3], dbase[5], dbase[7]]
            elif dbase[4] == "cy3":
                if len(dbase) > 5:
                    panel[dbase[0]] = ['-', dbase[3], dbase[5]]
                else:
                    panel[dbase[0]] = ['-', dbase[3], '-']
            elif dbase[4] == "cy5":
                panel[dbase[0]] = ['-', '-', dbase[3]]
            else:
                print(dbase)
        save_to = os.path.join(panel_save_dir, "panel_design.pickle")
        with open(save_to, "wb") as f:
            pickle.dump(panel, f)
        f.close()
    return panel


def read_dapi(fn, scale_to=(0, 65535)):
    if os.path.exists(fn):
        openOME = tf.TiffFile(fn)
        img = openOME.pages[0].asarray().astype(np.float)
        c = (scale_to[1] * (img - np.amin(img)) / np.ptp(img)).astype(int)
        return c
    else:
        raise Exception("file not exist")

def pretty_marker_name_list(marker_names_in_list):
    wrt_str = ""
    for l in marker_names_in_list:
        for m in l:
            if not m=="-":
                wrt_str += (m + ", ")
    return wrt_str

def get_dapis_for_a_ROI(roi_idx, img_dir, img_rounds, gray_scale=(0, 255), resize_to=(512, 512)):
    # print("ROI: %d" % roi_idx)
    r_s = "{:03d}".format(roi_idx)
    img_list = []
    for idx_i, r_n_i in enumerate(img_rounds):
        s_i = "S{:03d}".format(r_n_i)
        fn_i = os.path.join(img_dir, s_i, s_i + "_mono_dapi_reg_pyr16_region_" + r_s + ".tif")  # DAPI file name
        if gray_scale is not None:
            img_i = read_dapi(fn_i, scale_to=gray_scale).astype(np.uint8)
        else:
            img_i = read_dapi(fn_i)
        if resize_to is not None:
            img_obj = Image.fromarray(img_i, mode="L")
            img_obj.resize(resize_to)
            img_i = np.array(img_obj).astype(np.uint8)
            # img_obj.save("t1.jpg")
            # plt.imshow(img_i)
            # plt.savefig("t.jpg")
            # n = 2
            # b = img_i.shape[0] // n
            # img_i = img_i.reshape(-1, n, b, n).sum((-1, -3)) / n
            # img_i = np.resize(img_i, resize_to)
        img_list.append(img_i)
    return np.array(img_list)


def get_dapis_std(dapi_imgs, save_dir, ROI_id):

    save_to = os.path.join(save_dir, "dapi_std_img", str(ROI_id) + ".pickle")
    if not os.path.exists(os.path.join(save_dir, "dapi_std_img")):
        os.makedirs(os.path.join(save_dir, "dapi_std_img"))
    if os.path.exists(save_to):
        f = open(save_to, 'rb')
        std_img = pickle.load(f)
    else:
        std_img = np.std(dapi_imgs, axis=0)
        with open(save_to, "wb") as f:
            pickle.dump(std_img, f)
        f.close()
        print("\t\tDAPI std saved to %s" % save_to)
    return std_img


def get_SSIM_array(roi_idx, aligned_img_dir, N_range, out_dir):
    array_save_to = os.path.join(out_dir, "dapi_ssim_array", "ssim_array_region" + str(roi_idx) + ".pickle")
    if os.path.exists(array_save_to):
        f = open(array_save_to, 'rb')
        ssim_array = pickle.load(f)
    else:
        if not os.path.exists(os.path.join(out_dir, "dapi_ssim_array")):
            os.makedirs(os.path.join(out_dir, "dapi_ssim_array"))
        r_s = "{:03d}".format(roi_idx)
        # print("ROI: %d" % roi_idx)
        ssim_array = np.zeros([len(N_range), len(N_range)])
        for idx_i, r_n_i in enumerate(N_range):
            s_i = "S{:03d}".format(r_n_i)
            fn_i = os.path.join(aligned_img_dir, s_i,
                                s_i + "_mono_dapi_reg_pyr16_region_" + r_s + ".tif")  # DAPI file name
            # print("\t" + fn_i)
            for idx_j, r_n_j in enumerate(N_range):
                if idx_j > idx_i:
                    s_j = "S{:03d}".format(r_n_j)
                    fn_j = os.path.join(aligned_img_dir, s_j,
                                        s_j + "_mono_dapi_reg_pyr16_region_" + r_s + ".tif")  # DAPI file name

                    img_i = read_dapi(fn_i, scale_to=(0, 65535))
                    img_j = read_dapi(fn_j, scale_to=(0, 65535))
                    # print(fn_j)
                    # print(fn_i)
                    # print(img_i.shape)
                    # print(img_j.shape)
                    ssim_val = ssim(img_i, img_j, data_range=65535)
                    ssim_array[idx_i, idx_j] = ssim_val
                elif idx_j == idx_i:
                    ssim_array[idx_i, idx_j] = 1
        for idx_i, r_n_i in enumerate(N_range):
            for idx_j, r_n_j in enumerate(N_range):
                if idx_j < idx_i:
                    ssim_array[idx_i, idx_j] = ssim_array[idx_j, idx_i]
        with open(array_save_to, "wb") as f:
            pickle.dump(ssim_array, f)
        print("\t\tSSIM array saved to %s" % array_save_to)
    return ssim_array

def get_SSIM_array_from_dapi(dapi_imgs, roi_idx, N_range, out_dir):
    array_save_to = os.path.join(out_dir, "dapi_ssim_array", "ssim_array_region" + str(roi_idx) + ".pickle")
    if os.path.exists(array_save_to):
        f = open(array_save_to, 'rb')
        ssim_array = pickle.load(f)
    else:
        if not os.path.exists(os.path.join(out_dir, "dapi_ssim_array")):
            os.makedirs(os.path.join(out_dir, "dapi_ssim_array"))
        r_s = "{:03d}".format(roi_idx)
        # print("ROI: %d" % roi_idx)
        ssim_array = np.zeros([len(N_range), len(N_range)])
        for idx_i, r_n_i in enumerate(N_range):
            for idx_j, r_n_j in enumerate(N_range):
                if idx_j > idx_i:
                    img_i = dapi_imgs[idx_i]
                    img_j = dapi_imgs[idx_j]
                    # print(fn_j)
                    # print(fn_i)
                    # print(img_i.shape)
                    # print(img_j.shape)
                    ssim_val = ssim(img_i, img_j, data_range=65535)
                    ssim_array[idx_i, idx_j] = ssim_val
                elif idx_j == idx_i:
                    ssim_array[idx_i, idx_j] = 1
        for idx_i, r_n_i in enumerate(N_range):
            for idx_j, r_n_j in enumerate(N_range):
                if idx_j < idx_i:
                    ssim_array[idx_i, idx_j] = ssim_array[idx_j, idx_i]
        with open(array_save_to, "wb") as f:
            pickle.dump(ssim_array, f)
        print("\t\tSSIM array saved to %s" % array_save_to)
    return ssim_array


def plot_SSIM_array(ssim_array, roi_idx, N_range, save_to_dir):
    save_to = os.path.join(save_to_dir, "dapi_ssim_array", "ssim_img_region" + str(roi_idx) + ".png")
    if not os.path.exists(os.path.join(save_to_dir, "dapi_ssim_array")):
        os.makedirs(os.path.join(save_to_dir, "dapi_ssim_array"))
    if not os.path.exists(save_to):
        ssim_array = np.tril(ssim_array)  # only plot lower triangle of an array
        sns.heatmap(ssim_array, linewidth=0.5, xticklabels=N_range, yticklabels=N_range)
        plt.title("SSIM between iterations: Region_%d" % roi_idx)
        plt.tight_layout()
        plt.savefig(save_to)
        plt.close()
        print("\t\tDAPIs SSIM image save to %s" % save_to)


def plot_dapi_thumbnails(dapi_imgs, roi_id, out_dir):
    save_to = os.path.join(out_dir, "DAPI_thumbnails", "ROI_%d_all_dapi.png" % roi_id)
    if not os.path.exists(os.path.join(out_dir, "DAPI_thumbnails")):
        os.makedirs(os.path.join(out_dir, "DAPI_thumbnails"))
    if not os.path.exists(save_to):
        img_w_h = (4, 8)
        ele_img_sz = 200
        entire_img = np.zeros([img_w_h[0] * ele_img_sz, img_w_h[1] * ele_img_sz]).astype(np.uint8)
        for idx_i, dapi in enumerate(dapi_imgs):
            c = (255.0 * (dapi - np.amin(dapi)) / np.ptp(dapi)).astype(np.uint8)
            img = Image.fromarray(c, mode="L")
            # img.save("t1.jpg")
            # plt.imshow(c)
            # plt.savefig("t.jpg")
            img = img.resize((ele_img_sz, ele_img_sz))
            draw_loc = (5, 5)
            draw = ImageDraw.Draw(img)
            draw.text(draw_loc, str(idx_i), fill=255)
            x = idx_i // img_w_h[1]
            y = idx_i % img_w_h[1]
            img_arr = np.array(img)
            entire_img[x * ele_img_sz: (x + 1) * ele_img_sz, y * ele_img_sz: (y + 1) * ele_img_sz] = img_arr

        cm_hot = mpl.cm.get_cmap('hot')
        im = cm_hot(entire_img)
        entire_img = np.uint8(im * 255)

        img = Image.fromarray(entire_img)
        draw = ImageDraw.Draw(img)
        draw_loc = (entire_img.shape[1] - 50, entire_img.shape[1] - 50)
        draw.text(draw_loc, "ROI" + str(roi_id), fill=(255, 255, 255))
        img.save(save_to)
        print("\t\tDAPIs thumbnail image save to %s" % save_to)


def plot_dapi_std(dapi_std, roid_id, out_dir):
    save_to = os.path.join(out_dir, "dapi_std_img", "std_img_" + str(roid_id) + ".png")
    if not os.path.exists(os.path.join(out_dir, "dapi_std_img")):
        os.makedirs(os.path.join(out_dir, "dapi_std_img"))
    if not os.path.exists(save_to):
        plt.imshow(dapi_std)
        plt.axis('off')
        plt.savefig(save_to)
        plt.close()
        print("\t\tDAPI STD image save to %s" % save_to)


def get_overall_annotations(anno_df, panel_design, FOV, Round):
    round_key = "S{:03d}".format(Round)
    grades = ["Poor", "Fair", "Good", "Excellent"]
    anno_txt_list = []
    anno_scores = []
    if round_key in panel_design.keys():
        markers = panel_design[round_key]
        print(markers)
        for m in markers:
            if m != "-":
                anno_txt = anno_df.loc[FOV - 1, m.upper()]
                anno_txt_list.append(anno_txt)
                score_val = grades.index(anno_txt.strip()) - 1
                anno_scores.append(score_val)
    else:
        print("Round not in the OME Tiff file.")
    return anno_txt_list, anno_scores


# return rounds with images (bleaching round are excluded)
def get_meaningful_round(panel, round_range):
    meaningful_rounds = []
    markers_in_rounds = []
    for ss in round_range:
        SS = "S{:03d}".format(ss)
        if SS in panel.keys():
            meaningful_rounds.append(ss)
            cnt = 0
            markers = []
            for stain in panel[SS]:
                if stain != "-":
                    cnt += 1
                    markers.append(stain.upper())
            markers_in_rounds.append(markers)
    return meaningful_rounds, markers_in_rounds

# img_sz = 512
# factor = 1.8 #1.5, increase contrast, 1, gives original image; 0.5 #decrease constrast
def vis_marker_images(case_id, FOV, Round, panel, anno_df, ome_tiff_dir, out_dir, factor=1.8):
    all_imgs = get_images_vis(case_id, FOV, Round, panel, anno_df, ome_tiff_dir)
    # all_imgs = get_images(case_id, FOV, Round, panel, anno_df, ome_tiff_dir)
    FOV_dir = "ROI_%d" % FOV
    if not os.path.exists(os.path.join(out_dir, FOV_dir)):
        os.makedirs(os.path.join(out_dir, FOV_dir))
    save_to = os.path.join(out_dir, FOV_dir, "Round_%d.png" % Round)
    all_img_in_round = Image.fromarray(all_imgs, 'RGBA')
    # plt.imshow(all_img_in_round)
    # plt.savefig("test.png")
    enhancer = ImageEnhance.Contrast(all_img_in_round)
    all_img_in_round = enhancer.enhance(factor)
    draw = ImageDraw.Draw(all_img_in_round)
    draw_loc = (round(all_imgs.shape[1] / 2), all_imgs.shape[0] - 15)
    draw.text(draw_loc, "FOV%d, Round%d" % (FOV, Round), fill=(0, 0, 0))
    all_img_in_round.save(save_to)


def get_channel_names(tiff_file):
    omexml_string = tiff_file.pages[0].description
    root = xml.etree.ElementTree.parse(io.StringIO(omexml_string))
    namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    channels = root.findall('ome:Image[1]/ome:Pixels/ome:Channel', namespaces)
    channel_names = [c.attrib['Name'] for c in channels]
    return channel_names


def get_channel_index(panel, Round, channel_names):
    r_s = "S{:03d}".format(Round)
    if r_s in panel.keys():
        markers = panel[r_s]
        indx = []
        for m in markers:
            if m != "-":
                try:
                    e = channel_names.index(m.upper())
                except ValueError:
                    e = -1
                indx.append(e)
        return indx
    else:
        raise Exception("Round not in the OME Tiff file.")


def get_images_vis(case_id, fov, Round, panel, anno_df, img_dir, img_sz=512):
    global annotations
    c_maps = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples']
    if anno_df is not None:
        annotations, _ = get_overall_annotations(anno_df, panel, fov, Round)

    fn = os.path.join(img_dir, case_id + "_region_{:03d}.ome.tiff".format(fov))
    openOME = tf.TiffFile(fn)
    channel_names = get_channel_names(openOME)
    # for n in channel_names:
    #     print(n)
    indx = get_channel_index(panel, Round, channel_names)
    all_img = np.zeros((img_sz, img_sz * len(indx), 4)).astype(np.uint8)
    for i_idx, i in enumerate(indx):
        if i != -1:
            img_arr = openOME.pages[i].asarray().astype(np.float)
            v_min, v_max = autoSetDisplayRange(img_arr)
            img_arr[img_arr < v_min] = 0
            img_arr[img_arr > v_max] = v_max
            img_arr = (255.0 * (img_arr - np.amin(img_arr)) / np.ptp(img_arr)).astype(np.uint8)
        else:
            img_arr = np.zeros((img_sz, img_sz), dtype=np.uint8)
        img = Image.fromarray(img_arr, mode="L")
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(5.8)
        # https://ipython-books.github.io/111-manipulating-the-exposure-of-an-image/
        # img = skie.equalize_adapthist(img)
        img = img.resize((img_sz, img_sz))

        draw = ImageDraw.Draw(img)
        draw_loc = (5, 5)
        draw.text(draw_loc, channel_names[i], fill=255)

        draw_loc = (img_sz - 35, 5)
        if anno_df is not None:
            draw.text(draw_loc, annotations[i_idx], fill=255)

        cm_hot = mpl.cm.get_cmap(c_maps[i_idx])
        im = cm_hot(np.array(img))
        img_color_arr = np.uint8(im * 255)
        # print(channel_names[i])
        all_img[0: img_sz, i_idx * img_sz: (i_idx + 1) * img_sz, :] = img_color_arr
    return all_img

def get_images(case_id, fov, Round, panel, anno_df, img_dir, img_sz=512):
    global annotations
    c_maps = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples']
    if anno_df is not None:
        annotations, _ = get_overall_annotations(anno_df, panel, fov, Round)

    fn = os.path.join(img_dir, case_id + "_region_{:03d}.ome.tiff".format(fov))
    openOME = tf.TiffFile(fn)
    channel_names = get_channel_names(openOME)
    # for n in channel_names:
    #     print(n)
    indx = get_channel_index(panel, Round, channel_names)
    all_img = np.zeros((img_sz, img_sz * len(indx), 4)).astype(np.uint8)
    for i_idx, i in enumerate(indx):
        if i != -1:
            img_arr = openOME.pages[i].asarray().astype(np.float)
            img_arr = (255.0 * (img_arr - np.amin(img_arr)) / np.ptp(img_arr)).astype(np.uint8)
        else:
            img_arr = np.zeros((img_sz, img_sz), dtype=np.uint8)
        img = Image.fromarray(img_arr, mode="L")
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(5.8)
        # https://ipython-books.github.io/111-manipulating-the-exposure-of-an-image/
        # img = skie.equalize_adapthist(img)
        img = img.resize((img_sz, img_sz))

        draw = ImageDraw.Draw(img)
        draw_loc = (5, 5)
        draw.text(draw_loc, channel_names[i], fill=255)

        draw_loc = (img_sz - 35, 5)
        if anno_df is not None:
            draw.text(draw_loc, annotations[i_idx], fill=255)

        cm_hot = mpl.cm.get_cmap(c_maps[i_idx])
        im = cm_hot(np.array(img))
        img_color_arr = np.uint8(im * 255)
        # print(channel_names[i])
        all_img[0: img_sz, i_idx * img_sz: (i_idx + 1) * img_sz, :] = img_color_arr
    return all_img


def get_staining_orders(panel):
    # get staining orders and round #
    column_orders = []
    imaging_round = []
    round_per_stain = []
    meaningful_rounds = []
    markers_in_rounds = []
    for ss in range(32):
        SS = "S{:03d}".format(ss)
        if SS in panel.keys():
            meaningful_rounds.append(ss)
            imaging_round.append(SS)
            cnt = 0
            markers = []
            for stain in panel[SS]:
                if stain != "-":
                    cnt += 1
                    markers.append(stain.upper())
                    column_orders.append(stain.upper())
            round_per_stain.append(cnt)
            markers_in_rounds.append(markers)
    return meaningful_rounds, markers_in_rounds


def convert_score_to_value(list_txt_score):
    grades = ["Poor", "Fair", "Good", "Excellent"]  # 0, 1, 2, 3
    score_val = []
    for sc in list_txt_score:
        if sc.strip() == "A":
            score_val.append(1)
        else:
            score_val.append(grades.index(sc.strip()))
    return score_val


def get_colors(color_val_list):
    # colors = ['#400000', '#800000', '#BF0000', '#FF0000']
    colors = ['r', 'm', 'b', 'g']
    # colors = [[255, 0, 0], [255, 0, 255], [0, 0, 255], [0, 255, 0]]
    color_list = []
    color_val_list = color_val_list.squeeze()
    for c in color_val_list:
        color_list.append(colors[c])
    return color_list


def plot_marker_quality_scatter(panel, FOV_range, anno_df, ssim_pick_dir, out_dir):
    # for each round, read samples (regions), if there is potential issue (measured by DAPI SSIM)
    save_to = os.path.join(out_dir, "marker_image_quality_annotated.png")
    if not os.path.exists(save_to):
        meaningful_rounds, markers_in_rounds = get_staining_orders(panel)
        plt.figure(figsize=(10, 10), dpi=300)
        for idx, round_i in enumerate(meaningful_rounds):
            markers_in_the_round = markers_in_rounds[idx]
            print("Markers in round %d: %s" % (round_i, str(markers_in_the_round)))

            ssim_avg_list = []
            ssim_std_list = []
            anno_score_list = []
            for r in FOV_range:
                fn = os.path.join(ssim_pick_dir, "dapi_ssim_array", "ssim_array_region" + str(r) + ".pickle")
                fp = open(fn, 'rb')
                ssim_array = pickle.load(fp)

                vec_DAPI_SSIM_avg = np.mean(ssim_array, axis=1)
                vec_DAPI_SSIM_std = np.std(ssim_array, axis=1)
                ssim_value_avg_in_the_round = vec_DAPI_SSIM_avg[round_i - 3]
                ssim_value_std_in_the_round = vec_DAPI_SSIM_std[round_i - 3]

                anno_scores = anno_df.loc[r - 1, markers_in_the_round]
                score_val = convert_score_to_value(list(anno_scores))

                anno_score_list.append(score_val)

                ssim_avg_list.append(ssim_value_avg_in_the_round)
                ssim_std_list.append(ssim_value_std_in_the_round)

            anno_score_arr = np.array(anno_score_list)
            x = np.log(np.array(ssim_avg_list) / (1 - np.array(ssim_avg_list)))
            y = np.log(np.array(ssim_std_list) / (1 - np.array(ssim_std_list)))

            if len(markers_in_the_round) == 1:
                plt.scatter(x, y, marker='o', c=get_colors(anno_score_arr), s=15)

            else:
                for ax_idx in range(len(markers_in_the_round)):
                    plt.scatter(x, y, marker='o', color=get_colors(anno_score_arr[:, ax_idx]), s=15)

        title_str = ("Marker Image Quality")
        plt.title(title_str)
        plt.ylabel('SSIM_std (rev-sigmoid)')
        plt.xlabel('SSIM_avg (rev-sigmoid)')

        from matplotlib.lines import Line2D
        colors = ['r', 'm', 'b', 'g']
        scores = ["Poor", "Fair", "Good", "Excellent"]
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], label=scores[0]),
                           Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], label=scores[1]),
                           Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], label=scores[2]),
                           Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[3], label=scores[3])]
        plt.legend(handles=legend_elements)

        plt.savefig(save_to)
        plt.close()
