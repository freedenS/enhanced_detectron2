import os
import io
import sys
import contextlib
import argparse
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

def main():
    parser = argparse.ArgumentParser(
        description='This script support converting coco or voc to txt.')
    parser.add_argument('--json_file', type=str, default=None,
                        help='path to json files.')
    args = parser.parse_args()

    dataset_dicts, thing_classes = load_coco_json(args.json_file)
    # write label
    with open('label.txt', 'w') as f:
        for i in thing_classes:
            f.write(i + '\n')

    # wirte txt
    os.makedirs("./annotations")
    
    for data in dataset_dicts:
        filename = os.path.splitext(data['file_name'])[0]
        height, width = data['height'], data['width']
        with open('./annotations/'+filename+'.txt', 'w') as f:
            for i in data['annotations']:
                f.write("{} {} {} {} {}\n".format(i['category_id']-1, 
                round((i['bbox'][0] + i['bbox'][2] / 2) / width, 6), 
                round((i['bbox'][1] + i['bbox'][3] / 2) / height, 6), 
                round(i['bbox'][2] / width, 6), 
                round(i['bbox'][3] / height, 6)))

def load_coco_json(json_file):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        print("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    
    imgs = coco_api.loadImgs(img_ids)
    
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))

    print("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"]

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = img_dict["file_name"]
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        print(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts, thing_classes


if __name__ == '__main__':
    main()