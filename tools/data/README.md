This directory contains a few scripts that process data.




* `voc_to_coco.py`

A script converted pascal voc format xmls to coco format json.

Usage:

```
python voc_to_coco.py --ann_dir ./path \
                      --ann_ids trainval.txt \
                      --labels labels.txt \
                      --output ./out.json
```

ann_dir: path to annotation files directory.

```
  path/
    1.xml
    2.xml
    ...
  trainval.txt
  labels.txt
  voc_to_coco.py
```

ann_ids: path to annotation files ids list. trainval.txt should like this:

```
1
2
...
```

labels: path to label list.

```
aeroplane
bicycle
bird
...
```

output: path to output json file.



* `generate_filelist_label.py`

A script generated file list for specified directory and label list (only for xml).

Usage:
```
python generate_filelist_label.py --data_dir ./path \
                         (--output_filelist trainval.txt \
                          --ext \
                          --label \
                          --output_labels labels.txt)
```

data_dir: path to generating files list directory.

output_filelist: path to file list.

ext: file list include extension or not.

label: generate label list or not (only for xml).

output_labels: path to label list.



- `analyze_data.py`

A script analyzed annotated data.

Usage:

```
python analyze_data.py --data_dir ./path \
                       --format coco
```

data_dir: path to annotated data.

```
coco:
  path/
    annotations/
      train.json
    images/
      1.jpg
      2.jpg
      ...
      
voc:
  path/
    ImageSets/
      Main/
        train.txt
    Annotations/
      1.xml
      2.xml
      ...
```

format: coco or voc



* `json_to_txt.py`

A script converted json to txt. It will generate a directory (including all the txt files)  named annotations  and a label.txt in current directory.

Usage:

```
python json_to_txt.py --json_file ./train.json
```

json_file: path to json file.



* `json_to_png.py`

A script converted json include segmentation to png for semantic model. It will create a new folder named 'masks' under current path and mask image in it.(note: it currently not support the instance which is splited into different groups)

Usage:

```
python json_to_png.py --json_file ./train.json
```

json_file: path to json file.



* `pascal_split.py`

A script to split pascal datasets. It will create a new folder named 'ImageSets' under current path and four files(trainval.txt,train.txt,val.txt,test.txt) in it.

Usage:

``` python
python pascal_split.py --data_dir ./path --trainval_ratio 0.9 --train_ratio 0.9
```

data_dir: path to annotations with pascal format.

trainval_ratio: the proportion of train-val samples in total samples

train_ratio: the proportion of train samples in train-val samples



* `coco_split.py`

A script to split coco datasets. It will create a new folder named 'ImageSets' under current path and four files(trainval.json,train.json,val.json,test.json) in it.

Usage:

```
python coco_split.py --json_file ./train.json --trainval_ratio 0.9 --train_ratio 0.9
```

json_file: path to json file.

trainval_ratio: the proportion of train-val samples in total samples

train_ratio: the proportion of train samples in train-val samples



* `load_coco_json.py`

A script to load annotations from a json file. 

Return:

dataset_dict: key is image_id.

```
{'0':     // imgid
  {
    'images':{"file_name": "*.jpg", "height": 480, "width": 640, "id": 0} ,
    'annotations':[{"area": 384, "iscrowd": 0, "image_id": 0, "bbox": [197,         155, 16, 24], "category_id": 6, "id": 1, "ignore": 0, "segmentation": []},
    {"area": 384, "iscrowd": 0, "image_id": 0, "bbox": [217, 155, 16, 24],         "category_id": 4, "id": 2, "ignore": 0, "segmentation": []}]
  }
}
```

categories: categories of training data.