from mmdet.apis import init_detector, inference_detector

config_file = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = './checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
result = inference_detector(model, './demo/demo.jpg')
print(result)