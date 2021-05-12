### visualization

- GradCam++

Usage: just set cfg.VISUALIZED.LAYER_NAMES and follow [how to visualize](../../EASY_START.md#visualization).

```
cfg.VISUALIZED.LAYER_NAMES = [
  "backbone.bottom_up.res3.3.conv3",
  "backbone.bottom_up.res4.5.conv3",
  "backbone.bottom_up.res5.2.conv3",
  "backbone.fpn_output3",
  "backbone.fpn_output4",
  "backbone.fpn_output5",
  "backbone.top_block.p6",
  "backbone.top_block.p7",
  "head.cls_subnet.0",
  "head.cls_subnet.2",
  "head.cls_subnet.4",
  "head.cls_subnet.6",
  "head.bbox_subnet.0",
  "head.bbox_subnet.2",
  "head.bbox_subnet.4",
  "head.bbox_subnet.6"
] # example for retinanet

# how to get cfg.VISUALIZED.LAYER_NAMES
# model = build_model(cfg)
# for n, module in model.named_modules():
#    print(n)
```

**Note:**

**This module is experimental!**

**It is modified and different from https://github.com/yizt/Grad-CAM.pytorch** 

**It generates heatmaps with loss but [Grad-Cam.pytorch](https://github.com/yizt/Grad-CAM.pytorch) generates heatmaps with score.**

