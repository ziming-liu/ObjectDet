import os

import cv2

from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = 'configs/pgnet/visdrone_bl_full_pg2streamcascade_rcnn_r18_fpn_1x.py'
checkpoint_file = 'work_dirs/visdrone_sota_full_pg2streamcascade_rcnn_r18_fpn_1x/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'demo/0000277_04401_d_0000560.jpg' # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
#show_result(img, result, model.CLASSES)
# or save the visualization results to image files
path = "/home/share2/ziming/"+img.split('.')[0].split('/')[1]
if not os.path.exists(path):
    os.makedirs(path)
show_result(img, result, model.CLASSES,show=False, out_file=os.path.join(path,'result.jpg'))