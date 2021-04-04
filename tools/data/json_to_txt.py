import os
import io
import sys
import contextlib
import argparse
from load_coco_json import load_coco_json

def main():
    parser = argparse.ArgumentParser(
        description='This script support converting coco or voc to txt.')
    parser.add_argument('--json_file', type=str, default=None,
                        help='path to json files.')
    args = parser.parse_args()

    dataset_dicts, categories = load_coco_json(args.json_file)
    # write label
    with open('label.txt', 'w') as f:
        for i in categories:
            f.write(i['name'] + '\n')

    # wirte txt
    os.makedirs("./annotations", exist_ok=True)
    
    for data in dataset_dicts:
        data = dataset_dicts[data]
        filename = data['images']['file_name'][:-4]
        height, width = data['images']['height'], data['images']['width']
        with open('./annotations/'+filename+'.txt', 'w') as f:
            for i in data['annotations']:
                f.write("{} {} {} {} {}\n".format(i['category_id']-1, 
                round((i['bbox'][0] + i['bbox'][2] / 2) / width, 6), 
                round((i['bbox'][1] + i['bbox'][3] / 2) / height, 6), 
                round(i['bbox'][2] / width, 6), 
                round(i['bbox'][3] / height, 6)))

if __name__ == '__main__':
    main()