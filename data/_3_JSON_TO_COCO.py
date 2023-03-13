import pandas as pd
import os
import shutil
import json
from tqdm import tqdm
from numpyencoder import NumpyEncoder


SOURCE_JSON_LIST = []

TOTAL_COCO_JSON = {
    "images": [],    
    "annotations": [],
    "categories": []
}

def load_json(p):
    with open(p, 'r', encoding='utf-8') as r:
        v_json = json.load(r)
    return v_json

def dump_json(p, v):
    with open(p, 'w', encoding='utf-8') as w:
        json.dump(v, w, cls=NumpyEncoder, ensure_ascii=False, indent="\t")

def getJsonList(path):
    f = open(path, "r")
    while True:
        line = f.readline().strip()
        if not line: break
        SOURCE_JSON_LIST.append(line)

def covertCOCO(filePath):
    # global ANNOTATION_ID_INT
    sourceJson = load_json(filePath)

    IMAGE_INFO = dict()       
        
    IMAGE_INFO['file_name'] = sourceJson['dataset']['dataset.name']+'.jpg'
    IMAGE_INFO['height'] = sourceJson['dataset']['dataset.height']
    IMAGE_INFO['width'] = sourceJson['dataset']['dataset.width']

    # image_id = str(sourceJson['dataset']['dataset.id']).replace('P', '')
    image_id = len(TOTAL_COCO_JSON['images'])

    IMAGE_INFO['id'] = int(image_id)
    IMAGE_INFO['license'] = 0
    IMAGE_INFO['flickr_url'] = ''
    IMAGE_INFO['coco_url'] = ''
    IMAGE_INFO['date_captured'] = ''

    TOTAL_COCO_JSON["images"].append(IMAGE_INFO)


    annotation_list = sourceJson['annotation']  

    for i, annotation in enumerate(annotation_list):
        anno_x = list()
        anno_y = list()

        for k in range(len(annotation['annotation_point'])):
            if k % 2 == 0:
                anno_x.append(annotation['annotation_point'][k])
            if k % 2 == 1:
                anno_y.append(annotation['annotation_point'][k])

        # x_point = annotation['annotation.point'][0]
        # y_point = annotation['annotation.point'][1]
        # width = annotation['annotation.point'][2]
        # height = annotation['annotation.point'][3]

        x_point = min(anno_x)
        y_point = min(anno_y)
        width = max(anno_x) - min(anno_x)
        height = max(anno_y) - min(anno_y)

        ANNOTATION_INFO = dict() 
        seg_list = list()
        seg_list.append(annotation['annotation_point'])
        
        ANNOTATION_INFO['segmentation'] = seg_list
        ANNOTATION_INFO['area'] = width * height
        ANNOTATION_INFO['iscrowd'] = 0
        ANNOTATION_INFO['image_id'] = int(image_id)
        ANNOTATION_INFO['bbox'] = [x_point, y_point, width, height]
        ANNOTATION_INFO['category_id'] = selectCategory(sourceJson['metadata.clothes']['metadata.clothes.type'])
        ANNOTATION_INFO['id'] = len(TOTAL_COCO_JSON["annotations"])        

        TOTAL_COCO_JSON["annotations"].append(ANNOTATION_INFO)

def generateCategory():
    CATEGORY_INFO = dict()
    CATEGORY_INFO['id'] = 1
    CATEGORY_INFO['name'] = 'blouse'
    CATEGORY_INFO['supercategory'] = ''
    TOTAL_COCO_JSON['categories'].append(CATEGORY_INFO)

    CATEGORY_INFO = dict()
    CATEGORY_INFO['id'] = 2
    CATEGORY_INFO['name'] = 'cardigan'
    CATEGORY_INFO['supercategory'] = ''
    TOTAL_COCO_JSON['categories'].append(CATEGORY_INFO)

    CATEGORY_INFO = dict()
    CATEGORY_INFO['id'] = 3
    CATEGORY_INFO['name'] = 'coat'
    CATEGORY_INFO['supercategory'] = ''
    TOTAL_COCO_JSON['categories'].append(CATEGORY_INFO)

    CATEGORY_INFO = dict()
    CATEGORY_INFO['id'] = 4
    CATEGORY_INFO['name'] = 'jacket'
    CATEGORY_INFO['supercategory'] = ''
    TOTAL_COCO_JSON['categories'].append(CATEGORY_INFO)

    CATEGORY_INFO = dict()
    CATEGORY_INFO['id'] = 5
    CATEGORY_INFO['name'] = 'jumper'
    CATEGORY_INFO['supercategory'] = ''
    TOTAL_COCO_JSON['categories'].append(CATEGORY_INFO)

    CATEGORY_INFO = dict()
    CATEGORY_INFO['id'] = 6
    CATEGORY_INFO['name'] = 'shirt'
    CATEGORY_INFO['supercategory'] = ''
    TOTAL_COCO_JSON['categories'].append(CATEGORY_INFO)

    CATEGORY_INFO = dict()
    CATEGORY_INFO['id'] = 7
    CATEGORY_INFO['name'] = 'sweater'
    CATEGORY_INFO['supercategory'] = ''
    TOTAL_COCO_JSON['categories'].append(CATEGORY_INFO)

    CATEGORY_INFO = dict()
    CATEGORY_INFO['id'] = 8
    CATEGORY_INFO['name'] = 't-shrt'
    CATEGORY_INFO['supercategory'] = ''
    TOTAL_COCO_JSON['categories'].append(CATEGORY_INFO)

    CATEGORY_INFO = dict()
    CATEGORY_INFO['id'] = 9
    CATEGORY_INFO['name'] = 'vest'
    CATEGORY_INFO['supercategory'] = ''
    TOTAL_COCO_JSON['categories'].append(CATEGORY_INFO)

    CATEGORY_INFO = dict()
    CATEGORY_INFO['id'] = 10
    CATEGORY_INFO['name'] = 'pants'
    CATEGORY_INFO['supercategory'] = ''
    TOTAL_COCO_JSON['categories'].append(CATEGORY_INFO)

    CATEGORY_INFO = dict()
    CATEGORY_INFO['id'] = 11
    CATEGORY_INFO['name'] = 'skirt'
    CATEGORY_INFO['supercategory'] = ''
    TOTAL_COCO_JSON['categories'].append(CATEGORY_INFO)

    CATEGORY_INFO = dict()
    CATEGORY_INFO['id'] = 12
    CATEGORY_INFO['name'] = 'onepiece(dress)'
    CATEGORY_INFO['supercategory'] = ''
    TOTAL_COCO_JSON['categories'].append(CATEGORY_INFO)

    CATEGORY_INFO = dict()
    CATEGORY_INFO['id'] = 13
    CATEGORY_INFO['name'] = 'onepiece(jumpsuite)'
    CATEGORY_INFO['supercategory'] = ''
    TOTAL_COCO_JSON['categories'].append(CATEGORY_INFO)


def selectCategory(clothes_category):
    category_Id = 0
    if (clothes_category.find('blouse') != -1):
        category_Id = 1    

    elif (clothes_category.find('cardigan') != -1):
        category_Id = 2

    elif (clothes_category.find('coat') != -1):
        category_Id = 3

    elif (clothes_category.find('jacket') != -1):
        category_Id = 4

    elif (clothes_category.find('jumper') != -1):
        category_Id = 5

    elif (clothes_category.find('t-shirt') != -1):
        category_Id = 8
    
    elif (clothes_category.find('sweater') != -1):
        category_Id = 7

    elif (clothes_category.find('shirt') != -1):
        category_Id = 6            

    elif (clothes_category.find('vest') != -1):
        category_Id = 9

    elif (clothes_category.find('pants') != -1):
        category_Id = 10

    elif (clothes_category.find('skirt') != -1):
        category_Id = 11

    elif (clothes_category.find('onepiece(dress)') != -1):
        category_Id = 12

    elif (clothes_category.find('onepiece(jumpsuite)') != -1):
        category_Id = 13
    
    return category_Id


ROOT = './_2_JSON_LIST/'
TYPE = 'test'
TARGET_PREFIX = './_2_JSON_LIST/'
TARGET_ANNOTATION_FOLDER = "./annotations"

getJsonList(ROOT + TYPE + '_LIST.txt')
pbar = tqdm(SOURCE_JSON_LIST)
for filePath in pbar:
    covertCOCO(filePath)
    # check(filePath)
generateCategory()
dump_json(TARGET_ANNOTATION_FOLDER + '/' + TYPE + '.json', TOTAL_COCO_JSON)


# IMAGE FILE COPY ##
pbar = tqdm(SOURCE_JSON_LIST)
for filePath in pbar:    
    jsonFileName = filePath.split('/')[-1]
    imageFileName = jsonFileName.split('.')[-2]
    # category = filePath.split('/')[2]
    # detail = filePath.split('/')[4]

    print(jsonFileName)
    print()

    prefix = './원천데이터'

    # imagePath = prefix + category + '/' + detail + '/' + imageFileName + '.jpg'
    imagePath = prefix + '/' + imageFileName + '.jpg'

    if TYPE == 'train':
        # shutil.copy(imagePath, './mmdetection/data/nia_CH/1_Train/' + imageFileName + '.jpg')
        shutil.copy(imagePath, './images/train/' + imageFileName + '.jpg')
    elif TYPE == 'test':
        # shutil.copy(imagePath, './mmdetection/data/nia_CH/2_Test/' + imageFileName + '.jpg')
        shutil.copy(imagePath, './images/test/' + imageFileName + '.jpg')
    elif TYPE == 'val' :
        # shutil.copy(imagePath, './mmdetection/data/nia_CH/3_Val/' + imageFileName + '.jpg')
        shutil.copy(imagePath, './images/val/' + imageFileName + '.jpg')
# IMAGE FILE COPY ##


print(len(SOURCE_JSON_LIST))
print('OK...')