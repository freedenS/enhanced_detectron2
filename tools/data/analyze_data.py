import os
import sys
import logging
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.data import MetadataCatalog
from fvcore.common.file_io import PathManager
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.build import print_instances_class_histogram

def load_voc_instances(dirname, split='train'):

    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    class_names = {}
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {}
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls not in class_names:
                class_names[cls] = len(class_names)
            
            instances.append(
                {"category_id": class_names[cls]}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts, list(class_names.keys())

def main():
    parser = argparse.ArgumentParser(
        description='This script support analyzing data.')
    parser.add_argument('--data_dir', type=str, default=None, required=True,
                        help='path to annotation files directory.')
    parser.add_argument('--format', type=str, default='coco',
                        help='format to analyze.(coco or voc)')

    args = parser.parse_args()

    logger = logging.getLogger('detectron2')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    if args.format == 'coco':
        dataset_dicts = load_coco_json(os.path.join(args.data_dir, 'annotations/train.json'), 
                                       os.path.join(args.data_dir, 'images'), '1')
        class_names = MetadataCatalog.get('1').thing_classes
    elif args.format == 'voc':
        dataset_dicts, class_names = load_voc_instances(args.data_dir)
    else:
        raise Exception("only support coco or voc format")
    
    print_instances_class_histogram(dataset_dicts, class_names)
    #TODO add more analysis


if __name__ == "__main__":
    main()