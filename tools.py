from collections import defaultdict
from matplotlib import pyplot as plt
import os
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import json
import random
import shutil

label_file_path = 'list_patition_label(1).txt'
weight_hight_file_path = 'width_height.txt'
original_data_path = './original'
new_labeled_img_path = './random_300x7_img'
new_label_file_path = 'new_label_infos.txt'


# label_file_path：包含图片类别标签的文本文件，每一行内容为图片文件名和对应的类别
# weight_hight_file_path：包含所有图片的宽高信息，每一行内容为宽和高
# original_data_path：原数据集路径，包含所有图片
# new_labeled_img_path：人工标注文件夹路径

def plot_class_distribution():
    """
    统计数据集中每个类别的样本数量分布
    :return:
    """
    with open(label_file_path) as ls:
        lines = ls.readlines()
        sumer = defaultdict(int)
        for line in lines:
            temp = line.strip().split()[1]
            sumer[temp] += 1
        print(sumer)
        ordered_key = [key for key in sumer.keys()]
        ordered_key.sort()
        plt.bar(ordered_key, [sumer[key] for key in ordered_key])
        plt.show()


def width_height_extraction():
    """
    统计出所有数据集图片的宽和高，存储在文本文件中，方便后续分析使用
    :return:
    """
    with open(weight_hight_file_path, 'w') as wh:
        for p in os.listdir(original_data_path):
            temp_path = os.path.join(original_data_path, p)
            img = Image.open(temp_path)
            wh.write(str(img.size[0]) + ',' + str(img.size[1]) + '\n')


def plot_ratios():
    """
    统计数据集中每张图片的宽高信息及比例，并画图显示
    :return:
    """
    width_height = []
    with open(weight_hight_file_path) as wh:
        lines = wh.readlines()
        for line in lines:
            img_size = line.strip().split(',')
            width_height.append([float(img_size[0]), float(img_size[1])])
        ratios = [wh[1] / wh[0] for wh in width_height]
        areas = [wh[1] * wh[0] for wh in width_height]
        widths = [wh[0] for wh in width_height]
        heights = [wh[1] for wh in width_height]
        print('最大高：', max(heights),
              '最小高：', min(heights),
              '平均高：', sum(heights) / len(heights))
        print('最大宽：', max(widths),
              '最小宽：', min(widths),
              '平均宽：', sum(widths) / len(widths))
        print('最大高宽比：', max(ratios),
              '最小高宽比：', min(ratios),
              '平均高宽比：', sum(ratios) / len(ratios))
        print('最大面积：', max(areas),
              '最小面积：', min(areas),
              '平均面积：', sum(areas) / len(areas))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.plot([i for i in range(len(heights))], sorted(heights))
        plt.xlabel('第x个样本')
        plt.ylabel('高')
        plt.show()
        plt.plot([i for i in range(len(widths))], sorted(widths))
        plt.xlabel('第x个样本')
        plt.ylabel('宽')
        plt.show()
        plt.plot([i for i in range(len(ratios))], sorted(ratios))
        plt.xlabel('第x个样本')
        plt.ylabel('高宽比')
        plt.show()
        plt.plot([i for i in range(len(areas))], sorted(areas))
        plt.xlabel('第x个样本')
        plt.ylabel('面积')
        plt.show()
        plt.scatter([wh[0] for wh in width_height], [wh[1] for wh in width_height])
        plt.xlabel('宽')
        plt.ylabel('高')
        plt.show()


def w_h_clustering():
    """
    对宽高进行聚类
    :return:
    """
    db = DBSCAN(eps=10, min_samples=10)
    km = KMeans()
    width_height = []
    with open(weight_hight_file_path) as wh:
        lines = wh.readlines()
        for line in lines:
            img_size = line.strip().split(',')
            width_height.append([float(img_size[0]), float(img_size[1])])
    db.fit(width_height)
    plt.scatter([wh[0] for wh in width_height], [wh[1] for wh in width_height], c=db.labels_)
    plt.show()
    km.fit(width_height)
    plt.scatter([wh[0] for wh in width_height], [wh[1] for wh in width_height], c=km.labels_)
    plt.show()

    print(max(db.labels_), min(db.labels_))
    print(max(km.labels_), min(km.labels_))


def x_y_w_h_clustering():
    """
    对标注框中心坐标进行聚类
    :return:
    """
    db = DBSCAN(eps=10, min_samples=10)
    km = KMeans()
    x_y = []
    w_h = []
    new_label_infos = get_all_new_label_infos()
    for k, v in new_label_infos.items():
        boundings = v['boundings']
        for bounding in boundings:
            x_y.append([bounding[0] + (bounding[1] - bounding[0]) / 2, bounding[2] + (bounding[3] - bounding[2]) / 2])
            w_h.append([bounding[1] - bounding[0], bounding[3] - bounding[2]])
    db.fit(x_y)
    plt.scatter([wh[0] for wh in x_y], [wh[1] for wh in x_y], c=db.labels_)
    plt.show()
    km.fit(x_y)
    plt.scatter([wh[0] for wh in x_y], [wh[1] for wh in x_y], c=km.labels_)
    plt.show()

    print(max(db.labels_), min(db.labels_))
    print(max(km.labels_), min(km.labels_))

    db.fit(w_h)
    plt.scatter([wh[0] for wh in w_h], [wh[1] for wh in w_h], c=db.labels_)
    plt.show()
    km.fit(w_h)
    plt.scatter([wh[0] for wh in w_h], [wh[1] for wh in w_h], c=km.labels_)
    plt.show()

    print(max(db.labels_), min(db.labels_))
    print(max(km.labels_), min(km.labels_))


def random_300x7_img_from_dataset():
    """
        数据集中每个类别随机抽300张出来，总共就是7x300=2100张，用于人工标注
        :return:
    """
    with open(label_file_path) as ls:
        lines = ls.readlines()
        sumer = defaultdict(list)
        for line in lines:
            filename = line.strip().split()[0]
            category = line.strip().split()[1]
            sumer[category].append(filename)
        if not os.path.exists(new_labeled_img_path):
            os.mkdir(new_labeled_img_path)
        for k, v in sumer.items():
            # 打乱，每个类别取前300个
            random.shuffle(v)
            random_300 = v[:300]
            for img in random_300:
                shutil.copyfile(os.path.join(original_data_path, img), os.path.join(new_labeled_img_path, img))


def img_crop_by_json(img_path):
    """
    先根据图片路径获取同名的json文件，然后根据json文件中的标注信息提取原图片中对应的矩形框（可能多个）
    :param img_path: 图片路径
    :return: 输出切割出来的所有矩形框（PIL.Image类型）,及矩形框在原图的坐标信息
    """
    img = Image.open(img_path)
    json_path = img_path[:-3] + 'json'
    js = json.load(open(json_path))
    rectangles = js['shapes']
    croped_rec = []
    lrud = []
    for rec in rectangles:
        point_1 = rec['points'][0]
        point_2 = rec['points'][1]
        left, right, up, down = min(point_1[0], point_2[0]), max(point_1[0], point_2[0]), \
                                min(point_1[1], point_2[1]), max(point_1[1], point_2[1])
        cropped = img.crop((left, up, right, down))
        lrud.append((left, right, up, down))
        croped_rec.append(cropped)
    return croped_rec, lrud


def get_all_class_label_infos():
    """
    获取所有数据集中的文件名和对应的类别信息
    :return: 返回字典形式存储的信息
    """
    with open(label_file_path) as lf:
        lines = lf.readlines()
        label_info = {}
        for line in lines:
            temp = line.strip().split()
            label_info[temp[0]] = temp[1]
    return label_info


def get_all_new_label_infos():
    """
    获取人工标注图片的文件名和对应的标注信息
    :return: 返回字典形式存储的信息
    """
    with open(new_label_file_path) as lf:
        lines = lf.readlines()
        new_label_info = {}
        for line in lines:
            temp_info = {}
            temp_line = line.strip().split()
            temp_info['class'] = temp_line[1]
            temp_info['rec_num'] = int(temp_line[2])
            boundings = []
            for i in range(temp_info['rec_num']):
                bounding = [float(temp_line[i * 4 + 3]), float(temp_line[i * 4 + 4]),
                            float(temp_line[i * 4 + 5]), float(temp_line[i * 4 + 6])]
                boundings.append(bounding)
            temp_info['boundings'] = boundings
            new_label_info[temp_line[0]] = temp_info
    return new_label_info


def labeled_info():
    """
    重新整理label信息，只整理其中的人工标注了的2100张图片，将标注信息整理成TXT
    每一行格式为：图片名 表情类别 标注数n 边框1左边界 边框1右边界 边框1上边界 边框1下边界 ... 边框n左边界 边框n右边界 边框n上边界 边n下边界
    :return:
    """
    croped_img_paths = os.listdir(new_labeled_img_path)
    class_label_info = get_all_class_label_infos()
    with open("new_label_infos.txt", 'w') as nli:
        for croped_img in croped_img_paths:
            if croped_img.endswith('.json'):
                continue
            if croped_img.endswith('.jpg'):
                _, lrud = img_crop_by_json(os.path.join(new_labeled_img_path, croped_img))
                new_lrud = []
                for x in lrud:
                    new_lrud.append(str(x[0]))
                    new_lrud.append(str(x[1]))
                    new_lrud.append(str(x[2]))
                    new_lrud.append(str(x[3]))
                nli.write(' '.join([croped_img, class_label_info[croped_img], str(len(lrud))] + new_lrud) + '\n')


def re_label_json():
    """
    人工标注时没有标注表情类别，重新加入表情类别到json中
    :return:
    """
    temp_path = 'temp'
    croped_img_paths = os.listdir(new_labeled_img_path)
    new_label_info = get_all_new_label_infos()
    for file in croped_img_paths:
        if file.endswith('.json'):
            js = json.load(open(os.path.join(new_labeled_img_path, file)))
            n = len(js['shapes'])
            for i in range(n):
                js['shapes'][i]['label'] = new_label_info[file[:-4]+'jpg']['class']
            json.dump(js, open(os.path.join(temp_path, file), 'w'), indent=1)



if __name__ == '__main__':
    #re_label_json()
    pass
