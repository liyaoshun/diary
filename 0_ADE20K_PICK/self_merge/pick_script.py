import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import time
import random
import cv2 as cv

DATA_ROOT = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2"
pick_path = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/self_merge/pick_scene.txt"
mapd_path = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/self_merge/ade20k/sceneCategories.txt"
DATA_LIST = []

PICK_SCENE = []

merge_list = []

# 这个列表数组是通过人工赛选出来的，主要是根据场景类别信息进行赛选
idx_pick_20 = ["0,2,3,6,9,10,14,17,21,22,26,27,28,35,37,38,42,44,47,48,50,52,53,61,62,66,68,69,77,80,81,82,83,84,85,86,87,88,90,91,92,93,94,95,96,99,103,104,105,106,107,108,109,110,112,123,114,115,117,116,120,121,123,126,129,132,133,134,135,137,138,140,141,143,145,146,148,149,150",
"1,19,23,59,64,101,102,124,131,147",
"11,16,20,25,31,32,34,36,41,45,57,63,65,70,71,72,74,76,89,98,111,113,119,122,125,130",
"5,18,67,73,136",
"24,31",
"4,12,30,55",
"8,118",
"15",
"13",
"29",
"33,39",
"43",
"46,56,78,79,100",
"51",
"54,60,97",
"40,58",
"75,142,144",
"127",
"128",
"139"]

# 这个列表数组是通过人工赛选出来的，主要是根据场景类别信息进行赛选。然后结合第一个版本的分割效果重新调整了的分割类别。
idx_pick_15 = ["0,2,3,6,9,10,14,17,21,22,26,27,28,33,35,37,38,39,42,43,44,47,48,50,52,53,61,62,66,68,69,77,80,81,82,83,84,85,86,87,88,90,91,92,93,94,95,96,99,103,104,105,106,107,108,109,110,112,123,114,115,117,116,120,121,123,126,128,129,132,133,134,135,137,138,139,140,141,143,145,146,148,149,150",
"1,19,23,59,64,101,102,124,131,147",
"11,16,20,25,31,32,34,36,41,45,46,56,57,63,65,70,71,72,74,76,78,79,89,98,100,111,113,119,122,125,130",
"5,18,67,73,136",
"24,31",
"4,12,30,55",
"8,118",
"15",
"13",
"29",
"51",
"54,60,97",
"40,58",
"75,142,144",
"127"]


def parsing_category_information():
    category_file_path = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/self_merge/ade20k/objectInfo150.txt"
    category_info_list = [line.strip() for line in open(category_file_path)]
    index = -1
    fl = open("mapped_category_idx.txt", "w")
    for i in range(1, len(category_info_list)):
        Idx, Ratio, Train, Val, Name = category_info_list[i].split('\t')
        Name = Name.split(",")[0].strip()
        fl.write(Name+"\n")
        ss = 0
    fl.close()
    xxx = 0

## 映射ade20看数据到我们自己挑选的类别
def remap_idx_to_category():
    map_array = [i for i in range(151)]
    for idx in range(len(idx_pick_15)):
        category_idx = [int(x) for x in idx_pick_15[idx].split(",")]
        for cdx in category_idx:
            map_array[cdx] = idx
    return map_array

## 
def merge_category_information_to():
    save_root = os.path.join(DATA_ROOT, "ADE20K/merge15")
    save_images_root = os.path.join(save_root, "images")
    save_annotations_root = os.path.join(save_root, "annotations")

    image_path_root = os.path.join(DATA_ROOT, "ADE20K/ADEChallengeData2016/images")
    lable_path_root = os.path.join(DATA_ROOT, "ADE20K/ADEChallengeData2016/annotations")

    pick_names, pick_scenes = collect_pick_scene()
    index = -1
    flg_train = "training"
    flg_val = "validation"
    map_array = remap_idx_to_category()
    for image_name in pick_names:
        index+=1
        # flag = image_name.split("_")[1]
        if "train" in image_name:
            image_path_root_ = os.path.join(image_path_root, flg_train)
            lable_path_root_ = os.path.join(lable_path_root, flg_train)
        else:
            image_path_root_ = os.path.join(image_path_root, flg_val)
            lable_path_root_ = os.path.join(lable_path_root, flg_val)

        image_path = os.path.join(image_path_root_, image_name + ".jpg")
        lable_path = os.path.join(lable_path_root_, image_name + ".png")

        if os.path.isfile(image_path):
            if os.path.isfile(lable_path):
                lable = cv.imread(lable_path, 0)
                if 49 in lable or 137 in lable:
                    continue;
                image = cv.imread(image_path)
                newLable = np.copy(lable)
                indx = 0
                for idx in map_array:
                    newLable[lable==indx] = idx
                    indx += 1
                sv_path_lable = os.path.join(save_annotations_root, image_name + ".png")
                sv_path_image = os.path.join(save_images_root, image_name + ".png")
                cv.imwrite(sv_path_lable, newLable)      
                cv.imwrite(sv_path_image, image)         
            else:
                continue
        else:
            continue


def collect_pick_scene():
    scene_name_list = [line.strip() for line in open(pick_path)]
    scene_mapd_list = [line.strip() for line in open(mapd_path)]
    PICK_IMAGES = []
    PICK_SCENES = []
    for scene_mapd in scene_mapd_list:
        image_name, scene_name =scene_mapd.split(' ')
        if scene_name in scene_name_list:
            PICK_IMAGES.append(image_name)
            PICK_SCENES.append(scene_name)
    return PICK_IMAGES, PICK_SCENES


def show_pick_image():
    pick_names, pick_scenes = collect_pick_scene()
    index = -1
    for image_name in pick_names:
        index+=1
        image_path_root = os.path.join(DATA_ROOT, "ADE20K/ADEChallengeData2016/images/training")
        lable_path_root = os.path.join(DATA_ROOT, "ADE20K/ADEChallengeData2016/annotations/training")
        image_path = os.path.join(image_path_root, image_name + ".jpg")
        lable_path = os.path.join(lable_path_root, image_name + ".png")
        if os.path.isfile(image_path):
            if os.path.isfile(lable_path):
                image = cv.imread(image_path)
                lable = cv.imread(lable_path, 0)
                plt.figure(0)
                plt.subplot(121)
                plt.imshow(image)
                plt.title(pick_scenes[index])
                plt.subplot(122)
                plt.imshow(lable)
                plt.show()
            else:
                continue
        else:
            continue


def show_item():
    pick_names, pick_scenes = collect_pick_scene()
    index = -1
    for image_name in pick_names:
        index+=1
        image_path_root = os.path.join(DATA_ROOT, "ADE20K/ADEChallengeData2016/images/training")
        lable_path_root = os.path.join(DATA_ROOT, "ADE20K/ADEChallengeData2016/annotations/training")
        image_path = os.path.join(image_path_root, image_name + ".jpg")
        lable_path = os.path.join(lable_path_root, image_name + ".png")
        if os.path.isfile(image_path):
            if os.path.isfile(lable_path):
                lable = cv.imread(lable_path, 0)
                if 137 in lable:
                    image = cv.imread(image_path)
                    plt.figure(0)
                    plt.subplot(121)
                    plt.imshow(image)
                    plt.subplot(122)
                    plt.imshow(lable)
                    plt.title(pick_scenes[index])
                    plt.show()
            else:
                continue
        else:
            continue


def show_remap_image():
    dir_path = os.path.join(DATA_ROOT, "ADE20K/merge22/annotations")
    image_path_root = os.path.join(DATA_ROOT, "ADE20K/ADEChallengeData2016/images")
    flg_train = "training"
    flg_val = "validation"
    lable_lists = os.listdir(dir_path)
    for lable_name in lable_lists:
        if "train" in lable_name:
            image_path = os.path.join(image_path_root, flg_train, lable_name.replace(".png", ".jpg"))
        else:
            image_path = os.path.join(image_path_root, flg_val, lable_name.replace(".png", ".jpg"))
        lable_path = os.path.join(dir_path, lable_name)
        if os.path.isfile(image_path):
            image = cv.imread(image_path)
            lable = cv.imread(lable_path, 0)
            plt.figure(0)
            plt.subplot(121)
            plt.imshow(image)
            plt.subplot(122)
            plt.imshow(lable)
            plt.show()
        else:
            continue


def pixcel_statistic(class_num):
    lable_path_root = os.path.join(DATA_ROOT, "ADE20K/map_ade_cdbj_15_v2/annotations")
    lable_lists = os.listdir(lable_path_root)
    sta_array = [0 for i in range(class_num)]
    for lable_name in lable_lists:
        lable_path = os.path.join(lable_path_root, lable_name)
        if os.path.isfile(lable_path):
            lable = cv.imread(lable_path, 0)
            for j in range(len(sta_array)):
                sta_array[j] += np.sum( lable == j )
    print(sta_array)
    return sta_array

remap_mapv3_array =   [0, 5, 6, 2, 8, 7, 2, 2, 9,  0,  2, 10, 11, 0, 1, 14, 3]

# 将chenjun之前转好的数据挑选一部分重新转换标签为我们新的数据集中。主要是挑选动物类别的数据
def trans_mapv3_to_category():
    root = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/Self_Datas/sun"
    cd_lb_root = os.path.join(root, "annotations")
    cd_img_root = os.path.join(root, "images")
    # sv_cd_root = os.path.join(root, "annotations_remap")

    list_names = os.listdir(cd_lb_root)
    for name in list_names:
        # if "bj" in name or "cd" in name:
        lb_path = os.path.join(cd_lb_root, name)
        img_path = os.path.join(cd_img_root, name.replace(".png", ".jpg"))
        image = cv.imread(img_path)
        lable = cv.imread(lb_path, 0)
        sv_lb_root = os.path.join(root, "annotations_remaps")
        sv_img_root = os.path.join(root, "images_remaps")
        newLable = np.copy(lable)
        indx = 0
        for value in remap_mapv3_array:
            newLable[lable==indx] = value
            indx += 1
        sv_lb_path = os.path.join(sv_lb_root, name)
        sv_img_path = os.path.join(sv_img_root, name)
        cv.imwrite(sv_lb_path, newLable)
        cv.imwrite(sv_img_path, image)
        # plt.figure(0)
        # plt.subplot(131)
        # plt.imshow(lable)
        # plt.subplot(132)
        # plt.imshow(newLable)
        # plt.subplot(133)
        # plt.imshow(image)
        # plt.show()


def class_weights_cal_23():
    class_freq = [322851786, 651218545, 177378835, 289054334, 15358901, 53219282, 236708043, 
    121188525, 41277063, 35677308, 22119893, 7097458, 5416218, 6577992, 14845760, 8757682, 7169084, 
    19715315, 7271558, 341742, 254650, 1440863, 528812]
    weights = 1/np.log1p(class_freq)
    weights = 23 * weights / np.sum(weighs)
    print(weights)
    return weights

def class_weights_cal_20():
    class_freq = pixcel_statistic(20)
    weights = 1/np.log1p(class_freq)
    weights = 20 * weights / np.sum(weights)
    print(weights)
    return weights

def class_weights_cal(class_num = 15):
    class_freq = pixcel_statistic(class_num)
    weights = 1/np.log1p(class_freq)
    weights = np.array(class_num * weights / np.sum(weights))
    print(weights)
    return weights

# def pick_png():
#     image_root = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/Self_Datas"
#     list_names = os.listdir(os.path.join(image_root, "images"))
#     lable_root = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/segmentation-datas/mapv3_clear/annotations"
#     sv_lable_root = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/Self_Datas/annotations"
#     for name in list_names:
#         lable_path = os.path.join(lable_root, name.replace(".jpg", ".png"))
#         sv_path = os.path.join(sv_lable_root, name.replace(".jpg", ".png"))
#         lable = cv.imread(lable_path, 0)
#         cv.imwrite(sv_path, lable)
#         ss = 0


def check_mask_20():
    root_lable = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/ADE20K/map_ade_cdbj/annotations"
    lable_list = os.listdir(root_lable)
    for lable_name in lable_list:
        lable_path = os.path.join(root_lable, lable_name)
        lable = cv.imread(lable_path, 0)
        mx = int(np.max(lable))
        if mx >= 20:
            sss = 10
            print(mx)


def check_mask(num_cls = 15):
    root_lable = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/ADE20K/map_ade_cdbj_15_v2/annotations"
    lable_list = os.listdir(root_lable)
    for lable_name in lable_list:
        lable_path = os.path.join(root_lable, lable_name)
        lable = cv.imread(lable_path, 0)
        mx = int(np.max(lable))
        if mx >= num_cls:
            sss = 10
            print(mx)


def miou():
    array = [0.5744, 0.7128, 0.6192, 0.4908, 0.5602, 0.7215, 0.7975, 0.3947, 0.7071,
        0.5090, 0.5986, 0.5294, 0.5756, 0.4115]
    
    len_arr = len(array)
    m_value = np.mean(array)
    print(m_value)

def read_cocopanoptic_info():
    import json
    path = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/COCOPanoptic/panoptic_coco_categories.json"
    class_map = []
    with open(path, 'r') as f:
        fl = open("cocopanoptic.txt", "w")
        data = json.load(f)
        class_num = len(data)
        out_data = []
        for i in range(class_num):
            item = data[i]
            idx = item['id']
            name = item['name']
            fl.write(str(idx)+"  "+name+"\n")
            out_data.append({"idx":idx, "name":name})
            # out_data.append(idx)
        fl.close()
        return out_data


def copy_voc_dog_to_ourfiles():
    from shutil import copyfile
    image_root = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/Self_Datas/voc/images_remaps"
    lable_root = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/Self_Datas/voc/annotations_remaps"
    image_list = os.listdir(image_root)
    image_sv_root = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/ADE20K/map_ade_cdbj_15/images"
    lable_sv_root = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/ADE20K/map_ade_cdbj_15/annotations"
    for name in image_list:
        image_path = os.path.join(image_root, name)
        lable_path = os.path.join(lable_root, name)

        image_sv_path = os.path.join(image_sv_root, name)
        lable_sv_path = os.path.join(lable_sv_root, name)

        copyfile(image_path, image_sv_path)
        copyfile(lable_path, lable_sv_path)


def pick_image_from_COCOPanoptic_2017():
    root = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/COCOPanoptic"
    annotations_root_lb = os.path.join(root, "panoptic_annotations_trainval2017/annotations")

    train_annotations_root_lb = os.path.join(annotations_root_lb, "panoptic_train2017")
    val_annotations_root_lb = os.path.join(annotations_root_lb, "panoptic_val2017")

    train_image_root = os.path.join(root, "train2017")
    val_image_root = os.path.join(root, "val2017")

    val_list = os.listdir(val_image_root)
    class_array = [0 for i in range(133)]
    class_map =  read_cocopanoptic_info()

    for val_name in val_list:
        val_image_path = os.path.join(val_image_root, val_name)
        val_lable_path = os.path.join(val_annotations_root_lb, val_name.replace(".jpg", ".png"))

        lable = cv.imread(val_lable_path, 0)
        image = cv.imread(val_image_path)
        image=cv.cvtColor(image, cv.COLOR_BGR2RGB)

        uniq = np.unique(lable)
        plt.figure(0)
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(lable)
        plt.show()


def merge_coco():
    en_path = "./cocopanoptic.txt"
    ch_path = "./cocopanoptic_ch.txt"
    out_path = "./merge_cocopanoptic.txt"
    fl_en = open(en_path)
    fl_ch = open(ch_path)
    fl_merg = open(out_path, "w")
    en_lines = fl_en.readlines()
    ch_lines = fl_ch.readlines()
    len_en = len(en_lines)
    for i in range(len_en):
        en, ch = en_lines[i], ch_lines[i]
    # for en, ch in enumerate(en_lines, ch_lines):
        fl_merg.write(en.strip('\n')+"  "+ch.strip('\n')+"\n")
    fl_merg.close()
    fl_ch.close()
    fl_en.close()

def nyu_sun_statistic():
    root = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/Self_Datas"
    nyu_root = os.path.join(root, "nyu")
    sun_root = os.path.join(root, "sun")
    
    nyu_root = os.path.join(root, "nyu")
    nyu_image_root = os.path.join(nyu_root, "images")
    nyu_lable_root = os.path.join(nyu_root, "annotations")

    sun_image_root = os.path.join(sun_root, "images")
    sun_lable_root = os.path.join(sun_root, "annotations")

    nyu_list = os.listdir(nyu_lable_root)
    sun_list = os.listdir(sun_lable_root)

    nyu_array = [0 for i in range(17)]
    sun_array = [0 for i in range(17)]

    for nyu_name in nyu_list:
        nyu_labe_path = os.path.join(nyu_lable_root, nyu_name)
        nyu_lable = cv.imread(nyu_labe_path, 0)
        uni_array = np.unique(nyu_lable)
        for uni in uni_array:
            nyu_array[uni] += 1

    for sun_name in sun_list:
        sun_labe_path = os.path.join(sun_lable_root, sun_name)
        sun_lable = cv.imread(sun_labe_path, 0)
        uni_array = np.unique(sun_lable)
        for uni in uni_array:
            sun_array[uni] += 1

    xx = 0


def show_remaped_nyu_sun():
    root = "/media/robot/f6ee4448-930c-42d0-aa5d-74f09e9dbf93/mseg_dataset/MsegV2/ADE20K/map_ade_cdbj_15_v2"
    lable_list = os.listdir(os.path.join(root, "annotations"))
    for name in lable_list:
        lable_path = os.path.join(root,"annotations", name)
        image_path = os.path.join(root,"images", name)
        lable = cv.imread(lable_path, 0)
        image = cv.imread(image_path)

        plt.figure(0)
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(lable)
        plt.show()
        

if __name__ == "__main__":
    # step 1
    merge_category_information_to()
    # step 2
    trans_mapv3_to_category()
    # step 3
    class_weights_cal(class_num = 15)
    # step 4
    check_mask(num_cls = 15)
