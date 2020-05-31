import os
import json
import numpy as np
import glob
import shutil
from sklearn.model_selection import train_test_split

np.random.seed(41)

'''
这部分代码是将人工筛查后的所有json文件进行转换，
转换成coco数据集的格式，
放到datasets文件夹中，
用于之后训练

2020.05.25 -- CaoHuiBin
'''

# 0为背景
#classname_to_id = {"0": 1}
classname_to_id = {"Surprise": 1, "Fear": 2, "Disgust": 3, "Happy": 4, "Sad": 5, "Angry": 6, "Neutral": 7}
# 这些id和之后模型预测的编号会有所不同，模型预测的话，对应关系是：
# {"watermark": 0, "commom1": 1, "flaw01": 2, "flaw02": 3, "flaw05": 4, "flaw07": 5, "other": 6}
# 输入虽然是0为背景，但是输出的预测信息是预测的矩形框的信息，预测的矩形框都是前景物体，所以不会输出背景类别，所以就把这些类别用0到6来表示了

class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, obj, path):
        image = {}
        from labelme import utils
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape):
        label = shape['label']
        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        #annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


if __name__ == '__main__':
    labelme_path = "all_img_json"
    saved_coco_path = "./datasets/"
    # 创建文件
    if not os.path.exists("%scoco/annotations/" % saved_coco_path):
        os.makedirs("%scoco/annotations/" % saved_coco_path)
    if not os.path.exists("%scoco/train2017/" % saved_coco_path):
        os.makedirs("%scoco/train2017" % saved_coco_path)
    if not os.path.exists("%scoco/val2017/" % saved_coco_path):
        os.makedirs("%scoco/val2017" % saved_coco_path)
    # 获取images目录下所有的json文件列表
    json_list_path = glob.glob(labelme_path + "/*.json")
    # 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
    train_path, val_path = train_test_split(json_list_path, test_size=0.3)
    print("train_n:", len(train_path), 'val_n:', len(val_path))

    # 把训练集转化为COCO的json格式
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json' % saved_coco_path)
    for file in train_path:
        shutil.copy(file.replace("json", "jpg"), "%scoco/train2017/" % saved_coco_path)
    for file in val_path:
        shutil.copy(file.replace("json", "jpg"), "%scoco/val2017/" % saved_coco_path)

    # 把验证集转化为COCO的json格式
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json' % saved_coco_path)

    # 不用纠结文件名中2017还是2019的问题，都是仿照coco数据集的文件来命名的