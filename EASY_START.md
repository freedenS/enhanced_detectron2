## Play Detectron2 on Your Own Data

This document helps you to play detectron2 with your own data easily.

If you need converting your data format or other data operations, refer to [tools/data/README.md](./tools/data/README.md).

### Training

Most trains can follow the step. 

If you want to modify some parameters, refer to [config-references](https://detectron2.readthedocs.io/modules/config.html#config-references).

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
cfg.DATASETS.TEST = ('1',)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(False)
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

#### Rotated-RCNN

annotation tool: [labelimg2](https://github.com/chinakook/labelImg2)

```
from detectron2.data.datasets import register_pascal_voc_rotated
config_file = './configs/Rotated-Detection/rotated_faster_rcnn_R_50_FPN_1x.yaml'
class_list = ['1', '2', '3', ...]
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)
register_pascal_voc_rotated('1', './path', 'train', 2020, class_list)
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

### Deployment

#### RetinaNet

PyTorch>=1.5

```
import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.data.detection_utils import read_image
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

config_file = './output/date-retinanet/model/config.yaml'
model_file = './output/date-retinanet/model/model_final.pth'
cfg = get_cfg()
cfg.merge_from_file(config_file)
cfg.MODEL.WEIGHTS = model_file

torch_model = build_model(cfg)
DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
torch_model.eval()

inputs = read_image('./input.jpg', 'BGR')
transform_gen = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
inputs = transform_gen.get_transform(inputs).apply_image(inputs)
inputs = torch.as_tensor(inputs.astype('float32').transpose(2,0,1))
inputs = inputs.unsqueeze(0)

inputs_name = ['images']
outputs_name = ['boxes', 'scores', 'labels']
dynamic_axes = {'images':{2:'height', 3:'width'}, 'boxes':{0:'num'}, 'scores':{0:'num'}, 'labels':{0:'num'}}
with torch.no_grad():
    # ort
    torch.onnx.export(torch_model, inputs, './retinanet.onnx', opset_version=11,
                            input_names=inputs_name, output_names=outputs_name, dynamic_axes=dynamic_axes)
```

onnxruntime(gpu)>=1.3

```
import onnxruntime
sess = onnxruntime.InferenceSession('./retinanet.onnx')
oxres = sess.run(None, {sess.get_inputs()[0].name:inputs.numpy()})
```

