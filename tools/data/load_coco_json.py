import json

def load_coco_json(json_file):
    with open(json_file, 'r') as f:
        dataset_dict = json.load(f)
    
    imglist = dataset_dict['images']
    annotations = dataset_dict['annotations']
    categories = dataset_dict['categories']
    
    dataset_dict = {}
    for i in imglist:
        dataset_dict[i['id']] = {}
        cur_anno = {'images':i}
        img_anno = []
        for j in annotations:
            if j["image_id"] == i['id']:
                img_anno.append(j)
        cur_anno['annotations'] = img_anno
        dataset_dict[i['id']] = cur_anno
    return dataset_dict, categories
