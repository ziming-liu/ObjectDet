import os

import cv2

from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

#config_file = 'configs/pgnet/visdrone_bl_full_cascade_rcnn_r18_fpn_1x_s2x.py'
#checkpoint_file = 'work_dirs/visdrone_bl_full_cascade_rcnn_r18_fpn_1x_s2x/epoch_12.pth'

config_file = 'configs/pgnet/visdrone_bl_full_pg2streamcascade_rcnn_r18_fpn_1x_1333.py'
checkpoint_file = 'work_dirs/visdrone_sota_full_pg2streamcascade_rcnn_r18_fpn_1x_1333/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'demo/0000269_00001_d_0000348.jpg'# or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
#show_result(img, result, model.CLASSES)
# or save the visualization results to image files
path = "/home/share2/ziming/2sfpnv2/"#+img.split('.')[0].split('/')[1]
if not os.path.exists(path):
    os.makedirs(path)
show_result(img, result, model.CLASSES,show=False, out_file=os.path.join(path,'result.jpg'))