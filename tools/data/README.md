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



- `pascal_split.py`

A script splited pascal data to train, val and test set.

Usage:

```
python voc_to_coco.py --data_dir ./path \
                      --trainval_ratio 0.9 \
                      --train_ratio 0.9
```

data_dir: path to annotation files directory.

```
  path/
    Annotations/
      1.xml
      2.xml
      ...
    ImageSets/
      Main/
```

trainval_ratio: ratio of train and val set.

train_ratio: ratio of train set.



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