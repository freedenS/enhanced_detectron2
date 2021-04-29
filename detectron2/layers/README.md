### activation

[activations.py](./activations.py) add some activations. such as ReLU,SiLU,Hardswish,Swish,Mish,FReLU all of them refer to https://github.com/ultralytics/yolov5/blob/master/utils/activations.py

Usage: just set the activation you want to use as follow.

```
backbone:
  cfg.MODEL.RESNETS.ACTIVATION='Mish'
fpn:
  cfg.MODEL.FPN.ACTIVATION='Mish'
roi box head:
  cfg.MODEL.ROI_BOX_HEAD.ACTIVATION='Mish'
retinanet head:
  cfg.MODEL.RETINANET.ACTIVATION='Mish'
fcos head:
  cfg.MODEL.FCOS.ACTIVATION='Mish'
```

### loss

[loss.py](./loss.py) add some loss functions. such as diou and ciou. Both of them refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/iou_loss.py

Usage: just set the loss you want to use as follow.

```
generalized rcnn
  cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE='ciou'
retinanet
  cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE='ciou'
# support in fcos is coming soon
```

### attention

[attentions.py](./attentions.py) add some attention modules. such as CBAM. Code is modified from https://github.com/luuuyi/CBAM.PyTorch and https://github.com/Jongchan/attention-module
Note: It will initialize when it's created.

Usage: just set the attention module you want to use as follow.

```
CBAM
  cfg.MODEL.BACKBONE.ATTENTION_MODULE="CBAM"
```

