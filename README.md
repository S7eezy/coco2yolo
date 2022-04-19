# coco2yolo

#### Custom YoloV5 database creation with custom classes support

Based on **PyCOCOtools API** and **COCO database**

<img src="https://cocodataset.org/images/coco-logo.png" height="100" />
<img src="https://miro.medium.com/max/1400/1*bSLNlG7crv-p-m4LVYYk3Q.png" height="100" />

This project was meant to be used for **YoloV5** database creation, but you can easily adapt it for any other model format.


# Installation
`pip install -r requirements.txt`

# Usage

#### Example
Call coco2yolo class with desired arguments

In this example, our data model has a custom classes and the following classes indexes apply in labels: 

`0: Face mask, 1: car, 2: person, 3: bicycle`

```py
from coco2yolo import coco2yolo

coco2yolo(
    CUSTOM_DATASET=True,
    COCO_CATEGORIES_QUERY=['face mask', 'car', 'person', 'bicycle'],
    COCO_ANNOTATIONS="instances_train2017.json",
    NUM_SAMPLES=1500,
    OUTPUT_DIRECTORY="./output",
    CONVERT_YOLO=True
    )
```

#### Parameters

```properties
(list) COCO_CATEGORIES_QUERY : COCO categories to download (put abstract category to artificially increment cat_id index in your model)
(bool) CUSTOM_DATASET        : toggle custom classes indexes. Label indexes will be based on COCO_CATEGORIES_QUERY indexes
(str)  COCO_ANNOTATIONS      : path to COCO annotation json
(int)  NUM_SAMPLES           : number of images to download per category
(str)  OUTPUT_DIRECTORY      : path to output directory
(bool) CONVERT_YOLO          : convert bboxes to YOLOv5 format (from pixels location to image ratio location)

Important note : no label will be generated if you dont export to YoloV5 Format
```

#### Output

```jsonpath
../path/to/output/

        car/
            images/
                29327.jpg/
                ..
                8721.jpg/
            labels/
                29327.txt/
                ..
                8721.txt/
        
        person/
            images/
                761.jpg/
                ..
                91272.jpg/
            labels/
                761.txt/
                ..
                91272.txt/
```