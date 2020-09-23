# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import json
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from predictor import VisualizationDemo
from detectron2.engine.defaults import DefaultPredictor

from fvcore.common.file_io import PathManager
import numpy as np

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])

def decode_segmap(label_mask):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    n_classes = 9
    label_colours = get_pascal_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    
    return rgb

def testNet(modelID, modelSavePath, imagePath):
    # clsnameList = loadClsname(modelSavePath)

    #cfg = prepareConfig(modelType, modelSavePath, len(clsnameList))
    resultimgSavePath = os.path.join(modelSavePath, 'res')
    modelSavePath = os.path.join(modelSavePath, 'model')
    
    cfg = get_cfg()
    cfg.MODEL.ROI_HEADS.CLASSES_NAME = ['1']
    cfg.MODEL.ROI_HEADS.COLORS_LIST = []
    cfg.merge_from_file(os.path.join(modelSavePath, 'config.yaml'))
    cfg.MODEL.WEIGHTS = os.path.join(modelSavePath, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.6
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.5

    cfg.DATASETS.TEST = ('coco_2017_val_panoptic_stuffonly',)
    #cfg.DATASETS.TEST = (modelID,)
    demo = VisualizationDemo(cfg)
    #predictor = DefaultPredictor(cfg)
    clsnameList = cfg.MODEL.ROI_HEADS.CLASSES_NAME
    detRes = []
    testImgPath = os.path.join(imagePath, 'JPEGImages')
    imgList = os.listdir(testImgPath)

    # file list
    # with PathManager.open('./datasets/chuangkaicaise/list.txt') as f:
    #     fileids = np.loadtxt(f, dtype=np.str)
    totaltime = 0
    totalcount = 0
    for i, imgName in enumerate(imgList):
        totalcount += 1
        currentImgDict = {}
        # use PIL, to be consistent with evaluation
        imgPath = os.path.join(testImgPath, imgName)
        currentImgDict['model_id'] = modelID
        currentImgDict['path'] = imgName
        currentImgDict['mark'] = []
        img = read_image(imgPath, format="BGR")
        #img = cv2.imread(imgPath)

        start = time.time()
        predictions, visualized_output= demo.run_on_image(img)
        print (predictions)
        # labelmask = predictions['sem_seg'].argmax(dim=0).to(torch.device('cpu')).numpy()
        # res = decode_segmap(labelmask)

        end = time.time()
        curtime = end - start
        print ('cost time:', curtime)
        if i != 0:
            totaltime += curtime

        visualized_output.save(os.path.join(resultimgSavePath, imgName[:-4] + '.jpg'))
        # cv2.imwrite(os.path.join(resultimgSavePath, imgName[:-4] + '.jpg'), res)
        break

    print ('average time:', totaltime / totalcount)

if __name__ == '__main__':
    detRes = testNet('1234', './output/ocrcolor_detr','./datasets/ocrcolor_detr/')
    # print (detRes)

    # path = './datasets/tudou/masks/'
    # import os
    # import cv2
    # import numpy as np
    # filelist = os.listdir(path)
    # a = cv2.imread(path + 'ansai_1008.png', 0)
    # a[np.where(a == 3)] = 2
    # a[np.where(a == 4)] = 3
    # a[np.where(a == 6)] = 4
    # a[np.where(a == 7)] = 5
    # a[np.where(a == 10)] = 6
    # a[np.where(a == 11)] = 7
    # a[np.where(a == 12)] = 8
    # print(np.unique(a))
    # cv2.imwrite(path + 'ansai_100.png', a)