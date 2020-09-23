# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import os

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.data.datasets.coco import load_sem_seg

def prepareConfig(modelID, modelType, modelSavePath, dataFolder, clsnameList):

    if modelType == 'R50FCOSFASTER':
        config_file = './configs/FCOS-Faster-Detection/fcos_faster_rcnn_R_50_FPN_1x.yaml'
        register_pascal_voc(modelID, dataFolder, 'train', clsnameList)
    elif modelType == 'V39FPNFASTER':
        config_file = './configs/COCO-Detection/faster_rcnn_V_39_FPN_1x.yaml'
        register_pascal_voc(modelID, dataFolder, 'train', clsnameList)
    elif modelType == 'R50FPNFASTER':
        config_file = './configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml'
        register_pascal_voc(modelID, dataFolder, 'train', clsnameList)
    elif modelType == 'R50FPNRETINA':
        config_file = './configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml'
        register_pascal_voc(modelID, dataFolder, 'train', clsnameList)
    elif modelType == 'R50FPNFCOS':
        config_file = './configs/FCOS-Detection/fcos_R_50_FPN_1x.yaml'
        register_pascal_voc(modelID, dataFolder, 'train', clsnameList)
    elif modelType == 'D34FPNFCOS':
        config_file = './configs/FCOS-Detection/fcos_D_34_FPN_1x.yaml'
        register_pascal_voc(modelID, dataFolder, 'train', clsnameList)
    elif modelType == 'R50FPNMASK':
        config_file = 'configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml'
        register_coco_instances(modelID, {}, os.path.join(dataFolder, 'trainval.json'), os.path.join(dataFolder, 'images'))
    elif modelType == 'SEMANTIC':
        config_file = './configs/Misc/semantic_R_50_FPN_1x.yaml'
        DatasetCatalog.register(modelID, lambda:load_sem_seg("./datasets/tudou/masks", "./datasets/tudou/images", 'png', 'png'))
        MetadataCatalog.get(modelID).set(sem_seg_root="./datasets/tudou/masks", image_root="./datasets/tudou/images", evaluator_type="sem_seg")
        # DatasetCatalog.register(modelID, lambda:load_sem_seg("./datasets/chilun/masks", "./datasets/chilun/images"))
        # MetadataCatalog.get(modelID).set(sem_seg_root="./datasets/chilun/masks", image_root="./datasets/chilun/images", evaluator_type="sem_seg")

    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    # cfg.INPUT.MIN_SIZE_TRAIN = 2048
    # cfg.INPUT.MAX_SIZE_TRAIN = 2592
    # cfg.INPUT.MIN_SIZE_TEST = 2048
    # cfg.INPUT.MAX_SIZE_TEST = 2592

    cfg.DATASETS.TRAIN = (modelID,)
    cfg.DATASETS.TEST = (modelID,)
    # cfg.DATALOADER.NUM_WORKERS = 1
    # cfg.MODEL.WEIGHTS = './model/vovnet39_ese.pth'
    cfg.MODEL.WEIGHTS = './model/R-50.pkl'
    # cfg.MODEL.WEIGHTS = './model/dla34.pth'

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00125
    # cfg.SOLVER.BASE_LR = 0.000625
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(clsnameList)
    cfg.MODEL.FCOS.NUM_CLASSES = len(clsnameList)
    cfg.MODEL.RETINANET.NUM_CLASSES = len(clsnameList)

    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 9
    cfg.MODEL.SEM_SEG_HEAD.NORM = 'FrozenBN'

    cfg.OUTPUT_DIR = modelSavePath
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if os.path.exists(os.path.join(modelSavePath, 'tuning.yaml')):
        cfg.merge_from_file(os.path.join(modelSavePath, 'tuning.yaml'))

    cfg.freeze()
    #setup_logger(output=modelSavePath)

    return cfg
	
def trainNet(modelID, modelType, dataFolder, markValues, modelSavePath):
    modelSavePath = os.path.join(modelSavePath, 'model')
    
    cfg = prepareConfig(modelID, modelType, modelSavePath, dataFolder, markValues)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(False)
    with open(os.path.join(modelSavePath,'testConfig.yaml'), 'w') as f:
        f.write(cfg.dump())
    trainer.train()

if __name__ == "__main__":
    # clsnameList = ['1', '2', '4', '5', '6', '8']
    # dataFolder = './datasets/steelocr'
    # modelID = '1234'
    # trainNet(modelID, 'SEMANTIC', dataFolder,clsnameList, "./output/tudou")

    # print('successfully trained!')

    register_coco_instances('1', {}, './data/date/annotations/train.json', './data/date/images/')
    DatasetCatalog.get('1')
    mc = MetadataCatalog.get('1')
    print (mc.thing_classes)
