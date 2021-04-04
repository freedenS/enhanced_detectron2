import os
import io
import sys
import contextlib
import argparse
import cv2
import numpy as np
from load_coco_json import load_coco_json

def main():
    parser = argparse.ArgumentParser(
        description='This script support converting coco or voc to txt.')
    parser.add_argument('--json_file', type=str, default=None,
                        help='path to json files.')
    args = parser.parse_args()

    dataset_dicts, categories = load_coco_json(args.json_file)
    os.makedirs('./masks', exist_ok=True)
    for dd in dataset_dicts:
        dd = dataset_dicts[dd]
        height, width, annotations = dd["images"]["height"], dd["images"]["width"], dd["annotations"]
        res = np.zeros((height, width), dtype=np.uint8)
        
        for a in annotations:
            # TODO: ignore one instance which is splited more groups
            segmentation = a['segmentation'][0]
            s = []
            for i in range(int(len(segmentation) / 2)):
                s.append([segmentation[i*2], segmentation[i*2+1]])
            s = [s]
            s = np.array(s, dtype=np.int32)
            cv2.drawContours(res, s, -1, (a['category_id']), -1)
        cv2.imwrite("./masks/" + dd["images"]["file_name"][:-4] + ".png", res)

if __name__ == '__main__':
    main()
