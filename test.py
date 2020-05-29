import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import tools

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from demo.predictor import VisualizationDemo

import torch
import numpy as np
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

'''
这部分代码是训练好模型后，进行测试集的测试，并统计输出混淆矩阵、precision、recall
代码参考https://blog.csdn.net/weixin_39916966/article/details/103299051

2020.03.07 -- CaoHuiBin
'''

# constants
WINDOW_NAME = "detections"

# 数据集路径
PROJECT_ROOT = '/home/xmu_stu/chb/facial_expression'  # 根据情况修改位置
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'datasets/coco')
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train2017')
VAL_PATH = os.path.join(DATASET_ROOT, 'val2017')
TRAIN_JSON = os.path.join(ANN_ROOT, 'instances_train2017.json')
VAL_JSON = os.path.join(ANN_ROOT, 'instances_val2017.json')

# inference
INPUT_IMG_PATH = VAL_PATH  # '/home/bboyhb/pyproject/测试数据/站点1/站点1-测试图片'
OUTPUT_IMG_PATH = os.path.join(PROJECT_ROOT, 'imgout')

# 数据集类别元数据
DATASET_CATEGORIES = [
    {"name": "Surprise", "id": 1, "color": [220, 20, 60]},
    {"name": "Fear", "id": 2, "color": [21, 142, 185]},
    {"name": "Disgust", "id": 3, "color": [0, 20, 60]},
    {"name": "Happy", "id": 4, "color": [219, 142, 0]},
    {"name": "Sad", "id": 5, "color": [220, 0, 60]},
    {"name": "Angry", "id": 6, "color": [219, 142, 185]},
    {"name": "Neutral", "id": 7, "color": [119, 142, 185]},
]
CATEGORIES_NAMES = [k["name"] for k in DATASET_CATEGORIES]

# 数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "train_2019": (TRAIN_PATH, TRAIN_JSON),
    "val_2019": (VAL_PATH, VAL_JSON),
}


def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   metadate=get_dataset_instances_meta(),
                                   json_file=json_file,
                                   image_root=image_root)


def get_dataset_instances_meta():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_dataset_instances(name, metadate, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco",
                                  **metadate)


# 注册数据集和元数据
def plain_register_dataset():
    DatasetCatalog.register("train_2019", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "train_2019"))
    MetadataCatalog.get("train_2019").set(thing_classes=CATEGORIES_NAMES,
                                          json_file=TRAIN_JSON,
                                          image_root=TRAIN_PATH)
    DatasetCatalog.register("val_2019", lambda: load_coco_json(VAL_JSON, VAL_PATH, "val_2019"))
    MetadataCatalog.get("val_2019").set(thing_classes=CATEGORIES_NAMES,
                                        json_file=VAL_JSON,
                                        image_root=VAL_PATH)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    args.config_file = os.path.join(PROJECT_ROOT, "configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TRAIN = ("train_2019",)
    cfg.DATASETS.TEST = ("val_2019",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MAX_SIZE_TEST = 400
    cfg.INPUT.MIN_SIZE_TEST = 160
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 50
    # cfg.MODEL.POST_NMS_TOPK_TEST = 10
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CATEGORIES_NAMES)  # 类别数
    # cfg.MODEL.WEIGHTS = "/home/Documents/pretrainedModel/Detectron2/R-50.pkl"  # 预训练模型权重
    cfg.MODEL.WEIGHTS = os.path.join(PROJECT_ROOT, 'output/model_final.pth')  # 最终权重，训练好的模型
    cfg.SOLVER.IMS_PER_BATCH = 8  # batch_size=2; iteration = 1434/batch_size = 717 iters in one epoch
    ITERS_IN_ONE_EPOCH = int(1434 / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1  # 12 epochs
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.GAMMA = 0.1  # 这些参数的说明见 train.py 文件
    cfg.SOLVER.STEPS = (30000,)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default=os.path.join(PROJECT_ROOT, "configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml"),
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def flaw_only(predict):
    '''
    预测结果中有正常元件，水印，瑕疵，在这里筛选出瑕疵信息，其他的去掉。
    :param predict: 模型的正常输出的预测结果，预测出了许多矩形框，包含了矩形框的位置和大小信息（用左上角和右下角坐标来表示），
                    矩形框预测的类别、分数，矩形框内部的mask（用只含bool类型的矩阵表示像素级别的mask），等等
    :return: 筛选完的预测结果
    '''
    cpu_device = torch.device("cpu")
    instance = predict["instances"].to(cpu_device)
    image_size = instance.image_size
    get_pred_classes = instance.get("pred_classes").numpy()
    pred_classes_index = []
    pred_classes = []
    for c in range(len(get_pred_classes)):
        if get_pred_classes[c] != 0 and get_pred_classes[c] != 1:
            pred_classes_index.append(c)
            pred_classes.append(get_pred_classes[c])
    pred_classes = torch.from_numpy(np.asarray(pred_classes))
    scores = tensor_transform(instance.get("scores"), pred_classes_index)
    pred_masks = tensor_transform(instance.get("pred_masks"), pred_classes_index)
    pred_boxes = Boxes(tensor_transform(instance.get("pred_boxes").tensor, pred_classes_index))
    return Instances(image_size=image_size, pred_boxes=pred_boxes, scores=scores, pred_classes=pred_classes,
                     pred_masks=pred_masks)


def tensor_transform(t, indexes):
    '''
    tensor根据indexes筛选留下的部分
    :param t: 要进行筛选的tensor，比如有这些元素： [A, B, C, D, E, F]
    :param indexes: 表示需要留下的部分的索引，比如[1,3,4,5]
    :return: 筛选完的, 根据上面的假设，会返回： [B, D, E, F]
    '''
    tensor2array = t.numpy()
    new_array = []
    for index in indexes:
        new_array.append(tensor2array[index])
    new_array = torch.from_numpy(np.asarray(new_array))
    return new_array


def predict_API():
    '''
    供外部调用，返回一个初始化完成的模型，可以直接用来预测。搭建后台服务器用来响应前端页面的时候会用到。
    :return:
    '''
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    register_dataset()
    demo = VisualizationDemo(cfg, instance_mode=ColorMode.SEGMENTATION)
    return demo


def determine_final_class(predict):
    '''
    根据分数最高来判定一张图片最后的预测类别（属于哪一类瑕疵）
    :param predict: 输入一个预测集合，一张图片中预测的很多个矩形框及其类别和分数信息
    :return: 只返回分数最高的那个瑕疵框的类别
    '''
    cpu_device = torch.device("cpu")
    instance = predict["instances"].to(cpu_device)
    get_pred_classes = instance.get("pred_classes").numpy()
    pred_classes_index = []
    pred_classes = []
    for c in range(len(get_pred_classes)):
        if get_pred_classes[c] != 0 and get_pred_classes[c] != 1:
            pred_classes_index.append(c)
            pred_classes.append(get_pred_classes[c])
    if len(pred_classes) == 0:
        return 7
    scores = tensor_transform(instance.get("scores"), pred_classes_index).numpy().tolist()
    maxscore = 0
    maxindex = 0
    for i in range(len(scores)):
        if scores[i] > maxscore:
            maxscore = scores[i]
            maxindex = i
    return pred_classes[maxindex]


def contain_class(predict, real):
    '''
    考虑到模型提取瑕疵效果比较好，于是尝试了一种‘不严谨’的统计方法，仅用来作为分析参考，
    该方法将一张图检测出的多个瑕疵保存成为一个集合，
    如果被测试的图片的真实类别属于该集合中，则认为预测正确。
    :param predict: 输入一个预测集合，一张图片中预测的很多个矩形框及其类别和分数信息
    :param real: 输入这张图片的类别真值
    :return: 如果认为预测正确了（也就是上述集合中包含真值），就返回正确的这个类别（也就是真值），
            否则还是返回分数最高的那个瑕疵框的类别
    '''
    cpu_device = torch.device("cpu")
    instance = predict["instances"].to(cpu_device)
    get_pred_classes = instance.get("pred_classes").numpy()
    pred_classes_index = []
    pred_classes = []
    for c in range(len(get_pred_classes)):
        if get_pred_classes[c] != 0 and get_pred_classes[c] != 1:
            pred_classes_index.append(c)
            pred_classes.append(get_pred_classes[c])
    if len(pred_classes) == 0:
        return 7
    if real in pred_classes:
        return real
    scores = tensor_transform(instance.get("scores"), pred_classes_index).numpy().tolist()
    maxscore = 0
    maxindex = 0
    for i in range(len(scores)):
        if scores[i] > maxscore:
            maxscore = scores[i]
            maxindex = i
    return pred_classes[maxindex]


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # 注册数据集
    register_dataset()

    model_demo = VisualizationDemo(cfg)

    predict_list = []
    real_list = []
    # for path in tqdm.tqdm(args.input, disable=not args.output):
    for imgfile in os.listdir(INPUT_IMG_PATH):

        # use PIL, to be consistent with evaluation
        img_fullName = os.path.join(INPUT_IMG_PATH, imgfile)
        img = read_image(img_fullName, format="BGR")
        start_time = time.time()
        predictions, visualized_output = model_demo.run_on_image_detection(img)
        # imgpath.json这个文件里面是图片文件名和类别真值的对应表
        # 模型输出的类别中 0和1 分别代表水印和正常元件，而 2，3，4，5，6 分别是那五种瑕疵，
        # 还缺少一个表示false的数字，所以我们用7来表示
        # 所以在imgpath.json文件中你只能看到2,3,4,5,6,7这6种数字，其实分别就代表了我们最终要分类的6个类别

        img_paths = tools.get_all_new_label_infos()
        pred = predictions.get('pred_classes').tolist()
        if len(pred) == 0:
            predict_list.append('1')
        else:
            predict_list.append(str(pred[0]+1))
        real_list.append(img_paths[imgfile]['class'])
        visualized_output.save(os.path.join('imgout', imgfile))
        # print(flaw_only(predictions))
        # log会在控制台输出预测所花的时间等信息
        # logger.info(
        #     "{}: detected {} instances in {:.2f}s".format(
        #         imgfile, len(predictions["instances"]), time.time() - start_time
        #     )
        # )
        # 这个是保存预测错误的图片，会在原图上额外绘画出识别的矩形框、mask、类别、分数等等，好让我们看看预测错误的图片长什么样并且是哪里预测错了
        # if determine_final_class(predictions) != img_paths[imgfile]:
        #     temp = ['watermark', 'commom1', '01', '02', '05', '07', 'other', 'false']
        #     tempstr = temp[img_paths[imgfile]] + '-' + temp[determine_final_class(predictions)]
        #     out_filename = os.path.join(OUTPUT_IMG_PATH, tempstr + imgfile)
        #     visualized_output.save(out_filename)

    # 根据上面两种函数，determine_final_class和contain_class两种方案来统计的情况，生成混淆矩阵、precision、recall
    cm = confusion_matrix(real_list, predict_list)
    ps = precision_score(real_list, predict_list, average=None)
    rc = recall_score(real_list, predict_list, average=None)
    print(cm)
    print(['{:.2%}'.format(x) for x in ps])
    print(['{:.2%}'.format(x) for x in rc])


'''
        #这个是保存输出的预测图片，会在原图上额外绘画出识别的矩形框、mask、类别、分数等等
        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(imgfile))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)
        #这个是实时显示输出的预测图片，会在原图上额外绘画出识别的矩形框、mask、类别、分数等等
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit
'''
