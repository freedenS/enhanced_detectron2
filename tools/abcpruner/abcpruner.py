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
import torch
import random
import copy
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.data.datasets import register_sem_seg

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

class BeeGroup():
    def __init__(self):
        self.code = []
        self.fitness = 0
        self.rfitness = 0
        self.trail = 0

best_honey = BeeGroup()
NectraSource = []
EmployedBee = []
OnLooker = []
best_honey_state = None
food_number = 5
food_dimension = 16
food_limit = 5
honeychange_num = 2
max_preserve = 9
max_cycle = 10

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

def prepareConfig(dataFolder, clsnameList):
    config_file = './configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml'        

    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    cfg.DATASETS.TRAIN = ('train',)
    cfg.DATASETS.TEST = ('test',)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = './output/ocr/retina-smoothl1/model_final.pth'

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.000625
    cfg.SOLVER.MAX_ITER = 1000

    cfg.MODEL.BACKBONE.FREEZE_AT = 1

    cfg.MODEL.RETINANET.NUM_CLASSES = len(clsnameList)

    cfg.OUTPUT_DIR = './output/ocr/model'

    # cfg.freeze()

    return cfg

def calculateFitness(cfg, weights=None):
    global best_honey, best_honey_state
    trainer = Trainer(cfg)
    trainer.loadWeights(weights)
    # prevent loss to inf
    try:
        trainer.train()
    except Exception as e:
        print ('train fault')
    res = trainer.test(cfg, trainer._trainer.model)
    fitness = res['bbox']['AP50']
    print (f'current fitness:{fitness} best fitness:{best_honey.fitness}')
    if fitness > best_honey.fitness:
        best_honey_state = copy.deepcopy(trainer._trainer.model.state_dict())
        best_honey.code = copy.deepcopy(cfg.MODEL.ABCP.HONEY)
        best_honey.fitness = fitness
    return fitness

def initialize(cfg):
    global best_honey, NectraSource, EmployedBee, OnLooker

    for i in range(food_number):
        NectraSource.append(copy.deepcopy(BeeGroup()))
        EmployedBee.append(copy.deepcopy(BeeGroup()))
        OnLooker.append(copy.deepcopy(BeeGroup()))
        
        honey = [random.randint(1, max_preserve) for i in range(food_dimension)]
        NectraSource[i].code = honey
        cfg.MODEL.ABCP.HONEY = honey

        NectraSource[i].fitness = calculateFitness(cfg)
        NectraSource[i].rfitness = 0
        NectraSource[i].trail = 0

        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)
        EmployedBee[i].fitness = NectraSource[i].fitness
        EmployedBee[i].rfitness = NectraSource[i].rfitness
        EmployedBee[i].trail = NectraSource[i].trail

        OnLooker[i].code = copy.deepcopy(NectraSource[i].code)
        OnLooker[i].fitness = NectraSource[i].fitness
        OnLooker[i].rfitness = NectraSource[i].rfitness
        OnLooker[i].trail = NectraSource[i].trail

def sendEmployedBees(cfg):
    global NectraSource, EmployedBee
    for i in range(food_number):
        while 1:
            k = random.randint(0, food_number - 1)
            if k != i:
                break
        
        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)

        param2change = np.random.randint(0, food_dimension - 1, honeychange_num)
        R = np.random.uniform(-1, 1, honeychange_num)
        for j in range(honeychange_num):
            EmployedBee[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]] +
                R[j] * (NectraSource[i].code[param2change[j]] - NectraSource[k].code[param2change[j]]))
            EmployedBee[i].code[param2change[j]] = max(1, min(max_preserve, EmployedBee[i].code[param2change[j]]))

        cfg.MODEL.ABCP.HONEY = copy.deepcopy(EmployedBee[i].code)
        EmployedBee[i].fitness = calculateFitness(cfg)

        if EmployedBee[i].fitness > NectraSource[i].fitness:
            NectraSource[i].code = copy.deepcopy(EmployedBee[i].code)
            NectraSource[i].trail = 0
            NectraSource[i].fitness = EmployedBee[i].fitness
        else:
            NectraSource[i].trail = NectraSource[i].trail + 1

def calculateProbabilities():
    global NectraSource

    maxfit = best_honey.fitness

    for i in range(food_number):
        NectraSource[i].rfitness = (0.9 * (NectraSource[i].fitness / maxfit)) + 0.1

def sendOnlookerBees(cfg):
    global NectraSource, EmployedBee, OnLooker
    i = 0
    t = 0
    while t < food_number:
        R_choosed = random.uniform(0, 1)
        if R_choosed < NectraSource[i].rfitness:
            t += 1
            
            while 1:
                k = random.randint(0, food_number - 1)
                if k != i:
                    break

            OnLooker[i].code = copy.deepcopy(NectraSource[i].code)

            param2change = np.random.randint(0, food_dimension - 1, honeychange_num)
            R = np.random.uniform(-1, 1, honeychange_num)
            for j in range(honeychange_num):
                OnLooker[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]] +
                    R[j] * (NectraSource[i].code[param2change[j]] - NectraSource[k].code[param2change[j]]))
                OnLooker[i].code[param2change[j]] = max(1, min(max_preserve, OnLooker[i].code[param2change[j]]))

            cfg.MODEL.ABCP.HONEY = copy.deepcopy(OnLooker[i].code)
            OnLooker[i].fitness = calculateFitness(cfg)

            if OnLooker[i].fitness > NectraSource[i].fitness:
                NectraSource[i].code = copy.deepcopy(OnLooker[i].code)
                NectraSource[i].trail = 0
                NectraSource[i].fitness = OnLooker[i].fitness
            else:
                NectraSource[i].trail = NectraSource[i].trail + 1
        i += 1
        if i == food_number:
            i = 0

def sendScoutBees(cfg):
    global NectraSource, EmployedBee, OnLooker
    maxtrailindex = 0
    for i in range(food_number):
        if NectraSource[i].trail > NectraSource[maxtrailindex].trail:
            maxtrailindex = i
    if NectraSource[maxtrailindex].trail >= food_limit:
        for j in range(food_dimension):
            R = random.uniform(0, 1)
            NectraSource[maxtrailindex].code[j] = int(R * max_preserve)
            if NectraSource[maxtrailindex].code[j] == 0:
                NectraSource[maxtrailindex].code[j] += 1
        NectraSource[maxtrailindex].trail = 0
        cfg.MODEL.ABCP.HONEY = copy.deepcopy(NectraSource[maxtrailindex].code)
        NectraSource[maxtrailindex].fitness = calculateFitness(cfg)

def abcpruner():
    global best_honey, best_honey_state
    # ------------------------------------
    clsnameList = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    dataFolder = './data/voc'

    # register data
    register_pascal_voc('train', dataFolder, 'train', 2012, clsnameList)
    register_pascal_voc('val', dataFolder, 'val', 2012, clsnameList)
    MetadataCatalog.get('val').set(evaluator_type='pascal_voc')

    # prepare config
    basecfg = prepareConfig(dataFolder, clsnameList)
    print ('initializing-------------------------->')
    initialize(basecfg)

    for i in range(max_cycle):
        print (f'sendEmployedBees {i}----------------------->')
        sendEmployedBees(basecfg)
        print (f'calculateProbabilities {i}-------------------------->')
        calculateProbabilities()
        print (f'sendOnlookerBees {i}------------------------->')
        sendOnlookerBees(basecfg)
        print (f'sendScoutBees {i}---------------------------->')
        sendScoutBees(basecfg)

        print (f'search cycle {i} Best code:{best_honey.code} Best fitness:{best_honey.fitness}')
    
    basecfg.SOLVER.MAX_ITER = 9000
    basecfg.MODEL.ABCP.HONEY = copy.deepcopy(best_honey.code)
    with open('./output/ocr/model/testConfig.yaml', 'w') as f:
        f.write(basecfg.dump())
    print ("start fine tunning...")
    calculateFitness(basecfg, best_honey_state)

if __name__ == "__main__":
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    abcpruner()
