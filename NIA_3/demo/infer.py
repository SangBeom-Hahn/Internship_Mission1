import cv2
import matplotlib.pyplot as plt
# config 파일을 설정하고, 다운로드 받은 pretrained 모델을 checkpoint로 설정. 
# config 파일과 pretrained 모델을 기반으로 Detector 모델을 생성. 
from mmdet.apis import init_detector, inference_detector


config_file = 'configs/NIA31/mask_rcnn_r50_fpn_1x_coco_na31.py'
checkpoint_file = 'checkpoints/NIA_3-1.pth'


model = init_detector(config_file, checkpoint_file, device='cpu')
img = 'demo/gaurdigan_demo.jpg'
out_file = "demo/result.jpg"
# inference_detector의 인자로 string(file경로), ndarray가 단일 또는 list형태로 입력 될 수 있음. 
results = inference_detector(model, img)

from mmdet.apis import show_result_pyplot
# inference 된 결과를 원본 이미지에 적용하여 새로운 image로 생성(bbox 처리된 image)
# Default로 score threshold가 0.3 이상인 Object들만 시각화 적용. show_result_pyplot은 model.show_result()를 호출. 
# show_result_pyplot(model, img, results)

show_result_pyplot(
    model,
    img,
    results,
    palette='coco',
    score_thr=0.3,
    out_file=out_file)

type(results), len(results) # (list, 80)

print(results)

# print(model.__dict__)

# print(model.cfg.pretty_text)