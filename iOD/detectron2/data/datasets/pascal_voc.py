# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET
from random import shuffle

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import pdb


__all__ = ["register_pascal_voc"]


# fmt: off
# CLASS_NAMES = [
#     "person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light",
#     "traffic sign", "train",
# ]
CLASS_NAMES = [
    "car_daytime", "person_daytime", "car_night", "person_night",
]
# CLASS_NAMES = [
#     "car_night", "person_night", "car_daytime", "person_daytime",
# ]
# CLASS_NAMES = [
#     "car", "person", "car", "person",
# ]
# CLASS_NAMES = [
#     "car_night", "person_night",
# ]

# CLASS_NAMES = [
#     "car_daytime", "person_daytime",
# ]
# fmt: on


def load_voc_instances(dirname: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # shuffle(CLASS_NAMES)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        # tree = ET.parse(anno_file)
        try:
            tree = ET.parse(anno_file)
        except OSError as e:
            print(anno_file)
            continue

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls not in  CLASS_NAMES:
                continue
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_pascal_voc(name, dirname, split, year):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, year=year, split=split
    )
