## Play Detectron2 on Your Own Data

This document helps you to play detectron2 with your own data easily.

If you need converting your data format or other data operations, refer to [tools/data/README.md](./tools/data/README.md).

### Training

Most trains can follow the step. 

If you want to modify some parameters, refer to this.

Sample data: [Google drive](https://drive.google.com/file/d/1oapspxvzrvNuncBID8m-sfBPqSIpg_ND/view?usp=sharing) [BaiduYun](https://pan.baidu.com/s/1UUvd3beGlm9pmOCky0oxOQ)(rt8u)

```
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances

config_file = './configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml'
register_coco_instances('1', {}, './data/date/annotations/train.json',
                                 './data/date/images')
cfg = get_cfg()
cfg.merge_from_file(config_file)
cfg.DATASETS.TRAIN = ('1',)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
trainer = DefaultTrainer(cfg)
trainer.train()
```

if you want to save your config file:

```
with open('config.yaml', 'w') as f:
    f.write(cfg.dump())
```

#### semantic segmentation

```
from detectron2.data.datasets import register_sem_seg
config_file = './configs/Misc/semantic_R_50_FPN_1x.yaml'
register_sem_seg('1', "gt_root", "image_root", "png", "jpg")
```

### Testing

Most tests can follow the step.

```
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.detection_utils import read_image

cfg = get_cfg()
cfg.merge_from_file('config.yaml')
cfg.MODEL.WEIGHTS = 'model_final.pth'
predictor = DefaultPredictor(cfg)
img = read_image('test.jpg')
predictions = predictor(img)
```


