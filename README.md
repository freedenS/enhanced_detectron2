# <center>Easy to play detectron2</center>

### What is Detectron2

Detectron2 is Facebook AI Research's next generation software system that implements state-of-the-art object detection algorithms. [Official address](https://github.com/facebookresearch/detectron2)

## enhanced detectron2

- detailed example and data help you to play every models in detectron2.
- add more data augmentation, visualization in training.
- add more models.
- add recent trick.
- deploy your model ([ORT](https://github.com/microsoft/onnxruntime), [TensorRT](https://github.com/NVIDIA/TensorRT), [libtorch](https://pytorch.org/get-started/locally/)).

## Installation

See [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

build from source.

## Quick Start

See [EASY_START.md](EASY_START.md)

## TODO

### activations

- [x] [SiLU](./detectron2/layers/README.md)
- [x] [Hardswish](./detectron2/layers/README.md)
- [x] [Swish](./detectron2/layers/README.md)
- [x] [Mish](./detectron2/layers/README.md)
- [x] [FReLU](./detectron2/layers/README.md)
- [ ] ...

### loss functions

- [x] [diou](./detectron2/layers/README.md)
- [x] [ciou](./detectron2/layers/README.md)
- [ ] ...

### attention modules

- [x] [CBAM](./detectron2/layers/README.md)
- [ ] ...

### visualization modules

- [x] [GradCam++](./detectron2/engine/README.md)
- [ ] ...

### backbone

- [x] [path aggregation network](./detectron2/modeling/backbone/README.md)
- [ ] ...

### tools

- [x] [voc split](./tools/data/README.md) split voc data to train,val and test set
- [x] [generate file list and label](./tools/data/README.md) generate two texts with image list and label list respectively
- [x] [voc to coco](./tools/data/README.md) modify from [voc2coco](https://github.com/yukkyo/voc2coco) convert voc to coco format
- [x] [analyze data](./tools/data/README.md) output the basic information of training data
- [x] [json_to_txt](./tools/data/README.md) convert coco to txt for some yolo models
- [x] [json_to_png](./tools/data/README.md) convert coco to png for semantic models
- [x] [coco_split.py](./tools/data/README.md) split coco data to train,val and test set
- [ ] ...

### models

- [x] [FCOS](./detectron2/modeling/meta_arch/README.md)
- [ ] CenterMask2
- [ ] DETR
- [ ] YOLO
- [ ] ...

### deployments

- [x] [FasterRCNN(C4)-tensorrt](https://github.com/wang-xinyu/tensorrtx/tree/master/rcnn)
- [x] [MaskRCNN(C4)-tensorrt](https://github.com/wang-xinyu/tensorrtx/tree/master/rcnn)
- [ ] RotatedRCNN(C4)-tensorrt
- [x] [RetinaNet-tensorrt](https://github.com/NVIDIA/retinanet-examples) refer to NVIDIA
- [ ] FCOS-tensorrt
- [ ] CenterMask2-tensorrt
- [x] [DETR-tensorrt](https://github.com/wang-xinyu/tensorrtx/tree/master/detr)
- [ ] ...

### model compress

- [x] [ABCPruner](https://github.com/freedenS/enhanced_detectron2/blob/abcpruner/tools/abcpruner/README.md)
- [ ] ...

### operators

- [x] [roialign(tensorrt)](https://github.com/wang-xinyu/tensorrtx/blob/master/rcnn/RoiAlignPlugin.h)
- [ ] ...

