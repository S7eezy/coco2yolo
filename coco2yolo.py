__author__ = 'steezy'
__version__ = '1.0'
"""
pyCOCOtools API manipulation for YOLOv5 custom dataset generation
"""

from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import requests
from zipfile import ZipFile
from tqdm import tqdm
import os
import random
import cv2

"""
COCO Categories list
"""


class coco2yolo:
    def __init__(self, COCO_CATEGORIES_QUERY, CUSTOM_DATASET=False, COCO_ANNOTATIONS="instances_train2017.json",
                 NUM_SAMPLES=1500, OUTPUT_DIRECTORY="./output", CONVERT_YOLO=True):
        """
        Download any amount of random COCO images within specific categories.

        :param COCO_CATEGORIES_QUERY: (list) COCO categories to download (put abstract category to artificially increment cat_id index in your model)
        :param CUSTOM_DATASET: (bool) toggle custom classes indexes. Label indexes will be based on COCO_CATEGORIES_QUERY indexes
        :param COCO_ANNOTATIONS: (str) path to COCO annotation json
        :param NUM_SAMPLES: (int) number of images to download per category
        :param OUTPUT_DIRECTORY: (str) path to output directory
        :param CONVERT_YOLO: (bool) convert bboxes to YOLOv5 format
        """

        self.max_samples = NUM_SAMPLES
        self.output_path = OUTPUT_DIRECTORY
        self.categories_query = COCO_CATEGORIES_QUERY
        self.convert_yolo = CONVERT_YOLO
        self.custom_dataset = CUSTOM_DATASET
        self.coco_annotations = COCO_ANNOTATIONS

        self.categories_query_ids = []
        self.sampled_images = []
        self.annotations = None

        self.__build__()
        self.getCatIds()

    def __build__(self):
        # Check if query isn't empty
        if len(self.categories_query) < 1:
            exit("---------------------------------\nINVALID COCO_CATEGORIES_QUERY\n\nCorrect example : coco2yolo(COCO_CATEGORIES_QUERY=['car', 'handspinner', 'truck'])\n---------------------------------")

        # Download Annotations if local file doesn't exist
        try:
            self.annotations = COCO(annotation_file=self.coco_annotations)
        except FileNotFoundError:
            self.downloadAnnotations(url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip", annotation="instances_train2017", path=self.coco_annotations)

        # Init categories list by name
        self.categories_list = [cat['name'] for cat in self.annotations.loadCats(self.annotations.getCatIds())]

        # Make query ids list and replace custom classes by None to keep index
        if self.custom_dataset:
            for cat in self.categories_query:
                if cat in self.categories_list:
                    self.categories_query_ids.append(self.getCatId(catName=cat))
                else:
                    self.categories_query_ids.append(None)

    def downloadAnnotations(self, url, annotation, path):
        # Check if path is correct
        if not path.endswith(".json"):
            exit("---------------------------------\nINVALID COCO_ANNOTATIONS\n\nCorrect example : COCO_ANNOTATIONS='path/to/instances_train2017.json'\nIf you want this program to download annotations for you, simply remove COCO_ANNOTATIONS argument to load it as default.\n---------------------------------")

        print(f"Annotations not found: Downloading {url}...")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(url.split("/")[-1], 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        print("Extracting annotations...")
        with ZipFile(url.split("/")[-1], 'r') as zipObj:
            zipObj.extract(f"annotations/{annotation}.json")
            print(f"annotations/{annotation}.json was extracted successfully !")
        print(f"Deleting {url.split('/')[-1]}...")
        os.replace(f"annotations/{annotation}.json", path)
        os.rmdir("annotations")
        os.remove(url.split("/")[-1])
        self.annotations = COCO(annotation_file=path)

    def getCatName(self, catId):
        return self.annotations.loadCats(self.annotations.getCatIds(catIds=catId))[0]["name"]

    def getCatId(self, catName):
        return self.annotations.getCatIds(catNms=catName)[0]

    def getCatIds(self):
        """
        Select image IDs related to query categories
        """
        for category in self.categories_query:
            if category in self.categories_list:
                try:
                    query_id = self.getCatId(category)

                    """
                    Create ./output/{category} directories
                    """
                    os.makedirs(os.path.join(self.output_path, category, "labels"), exist_ok=True)
                    os.makedirs(os.path.join(self.output_path, category, "images"), exist_ok=True)

                    """
                    Start downloading images within a specific category
                    """
                    img_ids = self.annotations.getImgIds(catIds=self.getCatId(category))
                    self.COCO_Download(img_ids, query_id)
                except IndexError:
                    continue

    def COCO_Download(self, img_ids, query_id):
        """
        Download {COCO_MAX_SAMPLES} random images within {query_id} category
        """
        desc = f"Sampling '{self.getCatName(query_id)}' images"

        sampling = True
        num_samples = 0
        with tqdm(total=self.max_samples, desc=desc) as pbar:
            while sampling:
                """
                Select a random image ID in a specific category
                """
                next_id = random.randint(0, len(img_ids) - 1)
                im = img_ids[next_id]

                """
                Check for duplicate before downloading
                """
                if im not in self.sampled_images:
                    num_samples += 1
                    self.sampled_images.append(img_ids[next_id])
                    img_info = self.annotations.loadImgs([im])[0]
                    img_url = img_info["coco_url"]
                    image = Image.open(requests.get(img_url, stream=True).raw)
                    image.save(
                        fp=os.path.join(self.output_path, self.getCatName(query_id), "images", f"{im}.jpg"))

                    """
                    Replace the initial COCO label coordinates (xmin, ymin, xmax, ymax) by YOLO coordinates (x_center, y_center, width, height)
                    """
                    if self.convert_yolo:
                        self.Convert_YOLO(image=image, img_id=im, fp=os.path.join(self.output_path, self.getCatName(query_id), "images", f"{im}.jpg"))

                    del image
                    pbar.update(1)
                else:
                    continue

                """
                Check if sampling limitation is reached
                """
                if num_samples >= self.max_samples:
                    sampling = False
            pbar.close()

    def Convert_YOLO(self, image, img_id, fp):
        """
        Convert COCO coordinates to YOLO coordinates
        Initial COCO label coordinates : xmin, ymin, w, h (Pixels)
        Output YOLO label coordinates : x_center, y_center, width, height (%)
        """
        ann_ids = self.annotations.getAnnIds(imgIds=[img_id], iscrowd=None)
        annotations = self.annotations.loadAnns(ann_ids)

        image_data = []
        image_p = cv2.imread(fp)
        imgHeight, imgWidth, _ = image_p.shape
        for ann in annotations:
            if ann["category_id"]:
                """
                Extract coordinates from a single annotation
                """
                x = float(ann['bbox'][0])
                y = float(ann['bbox'][1])
                w = float(ann['bbox'][2])
                h = float(ann['bbox'][3])

                x_center = x + w / 2
                y_center = y + h / 2

                """
                Convert pixels location to image ratio location
                """
                ImgW, ImgH = image.size
                x_center /= ImgW
                y_center /= ImgH
                w /= ImgW
                h /= ImgH

                """
                Find the corresponding model category ID and save one bounding box
                """
                category_id = ann["category_id"]
                if not self.custom_dataset:
                    image_data.append([category_id, float(x_center), float(y_center), float(w), float(h)])
                else:
                    if category_id in self.categories_query_ids:
                        image_data.append([self.categories_query_ids.index(category_id), float(x_center), float(y_center), float(w), float(h)])


if __name__ == "__main__":
    coco2yolo(COCO_CATEGORIES_QUERY=['car', 'person', 'bicycle'], COCO_ANNOTATIONS="instances_train2017.json", NUM_SAMPLES=1500, OUTPUT_DIRECTORY="./output", CONVERT_YOLO=True)
