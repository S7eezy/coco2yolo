from coco2yolo import coco2yolo

if __name__ == "__main__":
    coco2yolo(CUSTOM_DATASET=True,
              COCO_CATEGORIES_QUERY=['face mask', 'car', 'person', 'motorcycle'],
              COCO_ANNOTATIONS="instances_train2017.json",
              NUM_SAMPLES=1500,
              OUTPUT_DIRECTORY="./output",
              CONVERT_YOLO=True)
