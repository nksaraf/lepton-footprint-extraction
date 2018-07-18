import json
import os
from tqdm import tqdm
from pycocotools.coco import COCO


def run():
    annofile = "data/mapping_challenge/annotation.json"
    if not os.path.exists(annofile):
        print("{} does not exist!".format(annofile))

    out_dir = "data/mapping_challenge/annotations/"
    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    imgset_file = "data/mapping_challenge/index.txt"
    if imgset_file:
        imgset_dir = os.path.dirname(imgset_file)
        if not os.path.exists(imgset_dir):
            os.makedirs(imgset_dir)

    # initialize COCO api.
    coco = COCO(annofile)

    img_ids = coco.getImgIds()
    img_names = []
    for img_id in tqdm(img_ids):
        # get image info
        img = coco.loadImgs(img_id)
        file_name = img[0]["file_name"]
        name = os.path.splitext(file_name)[0]
        if out_dir:
            # get annotation info
            anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = coco.loadAnns(anno_ids)
            # save annotation to file
            img_anno = dict()
            img_anno["image"] = img[0]
            img_anno["annotation"] = anno
            anno_file = "{}/{}.json".format(out_dir, name)
            with open(anno_file, "w") as f:
                json.dump(img_anno, f, sort_keys=False, indent=None, ensure_ascii=False)
        if imgset_file:
            img_names.append(name)
    if img_names:
        img_names.sort()
        with open(imgset_file, "w") as f:
            f.write("\n".join(img_names))