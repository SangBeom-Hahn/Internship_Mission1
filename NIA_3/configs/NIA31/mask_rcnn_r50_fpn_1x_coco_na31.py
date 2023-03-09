_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_nia31.py',
    '../_base_/datasets/kfashion.py',
    '../_base_/schedules/schedule_1x_nia31.py', '../_base_/default_runtime.py'
]

# load_from = 'checkpoints/latest.pth'