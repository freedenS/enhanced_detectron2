import os
import random
import argparse
from load_coco_json import load_coco_json
import json

def dataset_split(dataset_dicts, categories, train, val):         
    total_samples = len(dataset_dicts.keys())
    train_samples = int(total_samples * train)
    val_samples = int(total_samples * val)
    test_samples = total_samples - train_samples - val_samples
    test_dict = {'images':[], 'annotations':[], 'categories':categories}
    val_dict = {'images':[], 'annotations':[], 'categories':categories}
    train_dict = {'images':[], 'annotations':[], 'categories':categories}
    trainval_dict = {'images':[], 'annotations':[], 'categories':categories}
    
    for i in range(max(train_samples, val_samples, test_samples)):
        if len(test_dict['images']) != test_samples:
            pop_id = list(dataset_dicts.keys())[random.randint(0, len(dataset_dicts.keys()) - 1)]
            selected_samples = dataset_dicts.pop(pop_id)
            test_dict['images'].append(selected_samples['images'])
            test_dict['annotations'] += selected_samples['annotations']
        if len(val_dict['images']) != val_samples:
            pop_id = list(dataset_dicts.keys())[random.randint(0, len(dataset_dicts.keys()) - 1)]
            selected_samples = dataset_dicts.pop(pop_id)
            val_dict['images'].append(selected_samples['images'])
            val_dict['annotations'] += selected_samples['annotations']
            trainval_dict['images'].append(selected_samples['images'])
            trainval_dict['annotations'] += selected_samples['annotations']
        if len(train_dict['images']) != train_samples:
            pop_id = list(dataset_dicts.keys())[random.randint(0, len(dataset_dicts.keys()) - 1)]
            selected_samples = dataset_dicts.pop(pop_id)
            train_dict['images'].append(selected_samples['images'])
            train_dict['annotations'] += selected_samples['annotations']
            trainval_dict['images'].append(selected_samples['images'])
            trainval_dict['annotations'] += selected_samples['annotations']
        
    os.makedirs('./ImageSets', exist_ok=True)
    with open('./ImageSets/test.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_dict, indent=4))
    with open('./ImageSets/val.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(val_dict, indent=4))
    with open('./ImageSets/train.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(train_dict, indent=4))
    with open('./ImageSets/trainval.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(trainval_dict, indent=4))

def main():
    parser = argparse.ArgumentParser(
        description='This script support spliting voc data to train, val and test set.')
    parser.add_argument('--json_file', type=str, default=None,
                        help='path to annotation files.')
    parser.add_argument('--trainval_ratio', type=float, default=None,
                        help='ratio of train and val set.')
    parser.add_argument('--train_ratio', type=float, default=None,
                        help='ratio of train set.')
    args = parser.parse_args()

    dataset_dicts, categories = load_coco_json(args.json_file)
    
    test = 1-args.trainval_ratio
    train = args.trainval_ratio * args.train_ratio
    val = args.trainval_ratio - train
    dataset_split(dataset_dicts, categories, train, val)

if __name__ == '__main__':
    main()