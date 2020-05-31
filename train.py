import os
import cv2
import logging
import torch
import shutil
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA

'''
这部分代码是训练并输出保存模型
代码参考https://blog.csdn.net/weixin_39916966/article/details/103299051

2020.05.25 -- CaoHuiBin
'''

# 数据集路径
PROJECT_ROOT = '/home/xmu_stu/chb/facial_expression' #主目录位置
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'datasets/coco')
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train2017')
VAL_PATH = os.path.join(DATASET_ROOT, 'val2017')
TRAIN_JSON = os.path.join(ANN_ROOT, 'instances_train2017.json')
VAL_JSON = os.path.join(ANN_ROOT, 'instances_val2017.json')


# 数据集类别元数据
DATASET_CATEGORIES = [
    #{"name": "0", "id": 1, "color": [220, 20, 60]},
    {"name": "1", "id": 1, "color": [220, 20, 60]},
    {"name": "2", "id": 2, "color": [21, 142, 185]},
    {"name": "3", "id": 3, "color": [0, 20, 60]},
    {"name": "4", "id": 4, "color": [219, 142, 0]},
    {"name": "5", "id": 5, "color": [220, 0, 60]},
    {"name": "6", "id": 6, "color": [219, 142, 185]},
    {"name": "7", "id": 7, "color": [119, 142, 185]},
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


# 查看数据集标注
def checkout_dataset_annotation(name="val_2019"):
    dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('show', vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() # 拷贝default config副本
    args.config_file = os.path.join(PROJECT_ROOT, "configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.merge_from_file(args.config_file)   # 从config file 覆盖配置
    cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置

    # 更改配置参数, "train_2019" "val_2019" 都是自己命名的
    cfg.DATASETS.TRAIN = ("train_2019",)
    cfg.DATASETS.TEST = ("val_2019",)
    cfg.DATALOADER.NUM_WORKERS = 2  # n线程
    cfg.INPUT.MAX_SIZE_TRAIN = 400 #400
    cfg.INPUT.MAX_SIZE_TEST = 400 #400
    cfg.INPUT.MIN_SIZE_TRAIN = (160,)
    cfg.INPUT.MIN_SIZE_TEST = 160
    #cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 20
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 50
    #cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 10
    # cfg.MODEL.POST_NMS_TOPK_TEST = 10
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CATEGORIES_NAMES)  # 类别数
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl" # 'output/model_final.pth'   # 预训练模型权重
    cfg.SOLVER.IMS_PER_BATCH = 2  # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size
    ITERS_IN_ONE_EPOCH = int(10738 / cfg.SOLVER.IMS_PER_BATCH) #int(1434 / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 20) - 1  # 12 epochs
    cfg.SOLVER.BASE_LR = 0.02  #学习率
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.GAMMA = 0.1   # GAMMA和STEPS是相互配合的，意味着iteration到达STEPS中的数量时，把学习率乘以GAMMA
    cfg.SOLVER.STEPS = (90000, 100000)    # 在这里就是当iteration到达120000时，学习率从0.02变成0.002，因为训练到后面会难以收敛所以要减小学习率使得loss可以到达局部最低点
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 10000
    cfg.SOLVER.WARMUP_ITERS = 10000   # 这里WARMUP的作用就是学习率在一开始的ITERS范围内（这里是iteration从0到1000）过程中是从零开始线性递增的
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = 50000#ITERS_IN_ONE_EPOCH - 1   #每隔多少iteration输出一次loss结果
    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH - 1

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    cfg = setup(args)
    print(cfg)

    # 注册数据集
    register_dataset()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    #最后将训练过程的loss等等数据（会输出到output文件夹中，训练过程会自动创建文件夹和文件）保存到metrics文件夹中方便以后画图分析loss
    dirs = os.listdir('./metrics')
    last = len(dirs)
    #下面三个根据需要来拷贝（尤其是模型，比较占用空间，没有需要的话不用去额外拷贝，但是下次训练的时候这个模型数据会被覆盖掉）
    if not os.path.exists('./metrics/' + str(last)):
        os.mkdir('./metrics/' + str(last))
    # 拷贝初始参数的文件
    if os.path.exists('./output/config.yaml'):
        shutil.copy('./output/config.yaml', './metrics/' + str(last))
    # 拷贝各个阶段输出的loss数据
    if os.path.exists('./output/metrics.json'):
        shutil.copy('./output/metrics.json', './metrics/' + str(last))
        os.remove('./output/metrics.json')
    # 拷贝输出的模型
    # if os.path.exists('./output/model_final.pth'):
    #     shutil.copy('./output/model_final.pth', './metrics/' + str(last))
