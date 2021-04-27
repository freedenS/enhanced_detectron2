### Semantic segmentation

```
from detectron2.data.datasets import register_sem_seg
config_file = './configs/Misc/semantic_R_50_FPN_1x.yaml'
register_sem_seg('1', "gt_root", "image_root", "png", "jpg")
```

### Rotated-RCNN

annotation tool: [labelimg2](https://github.com/chinakook/labelImg2)

```
from detectron2.data.datasets import register_pascal_voc_rotated
config_file = './configs/Rotated-Detection/rotated_faster_rcnn_R_50_FPN_1x.yaml'
class_list = ['1', '2', '3', ...]
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)
register_pascal_voc_rotated('1', './path', 'train', 2020, class_list)
```

### FCOS

This implement is modified from https://github.com/aim-uofa/AdelaiDet

Usage:

```
set config_file = './configs/COCO-Detection/fcos_R_50_FPN_1x.yaml'
and follow training step of EASY_START.md
```

