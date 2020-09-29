# <center>Easy to play detectron2</center>

### What is Detectron2

Detectron2 is Facebook AI Research's next generation software system that implements state-of-the-art object detection algorithms. [Official address](https://github.com/facebookresearch/detectron2)

## enhanced detectron2

- There will be detailed example and data help you to play every models in detectron2.
- add more data augmentation, visualization in training.
- add more models.
- deploy your model ([ORT](https://github.com/microsoft/onnxruntime), [TensorRT](https://github.com/NVIDIA/TensorRT), [libtorch](https://pytorch.org/get-started/locally/)).

## Installation

See [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

build from source.

## Quick Start

See [EASY_START.md](EASY_START.md)

## TODO

### tools

- [x] [voc split](./tools/data/README.md)
- [x] [generate file list and label](./tools/data/README.md)
- [x] [voc to coco](./tools/data/README.md) modify from [voc2coco](https://github.com/yukkyo/voc2coco)
- [x] [analyze data](./tools/data/README.md)
- [x] [json_to_txt](./tools/data/README.md)
- [ ] ...

### models

- [ ] FCOS
- [ ] CenterMask2
- [ ] DETR
- [ ] YOLO
- [ ] ...

### deployments

- [x] RetinaNet-onnx
- [ ] RetinaNet-tensorrt
- [ ] FPN_Faster-onnx
- [ ] FPN-Mask-onnx
- [ ] FPN_Rotated_RCNN-onnx
- [ ] FCOS-onnx
- [ ] CenterMask2-onnx
- [ ] DETR-onnx
- [ ] ...

