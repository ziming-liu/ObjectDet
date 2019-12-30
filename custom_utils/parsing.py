import random
import numpy as np
import xml.dom.minidom
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import os
def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  max(b1_x1, b2_x1)
    inter_rect_y1 =  max(b1_y1, b2_y1)
    inter_rect_x2 =  min(b1_x2, b2_x2)
    inter_rect_y2 =  min(b1_y2, b2_y2)
    #Intersection area
    inter_width = inter_rect_x2 - inter_rect_x1 + 1
    inter_height = inter_rect_y2 - inter_rect_y1 + 1
    if inter_width > 0 and inter_height > 0:#strong condition
        inter_area = inter_width * inter_height
        #Union Area
        b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area)
    else:
        iou = 0
    return iou

def aug_txt(img_idx, txt_path='./', img_path='./'):
    class_name = ['ignored regions','pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','others']
    img = Image.open(os.path.join(img_path,img_idx+".jpg"))
    width, height = img.size
    fin = open(os.path.join(txt_path,img_idx+'.txt'), 'r')
    #get all the gts
    bboxes = []
    lines = []
    for line in fin.readlines():
        line = line.split(',')
        lines.append(line)
        bboxes.append([int(line[0]), int(line[1]),int(line[0])+int(line[2])-1, int(line[1])+int(line[3])-1])
    fin.close()

    #generate new gt
    specific_class_idxs = [3,6,7,8]#index of pedestrian, awning-tricycle
    threshold = 0.2
    sample_num_per_sample = 300
    with open(os.path.join(txt_path,img_idx+'.txt'), 'a') as fout:
        print("info: {}".format(img_idx))
        for line in lines:
            class_name = int(line[5])
            if class_name in specific_class_idxs:
                bbox_left, bbox_top, bbox_width, bbox_height = int(line[0]), int(line[1]), int(line[2]), int(line[3])
                expand_bbox_width, expand_bbox_height = bbox_width+10, bbox_height+10
                for i in range(sample_num_per_sample):
                    new_bbox_left = random.randint(0, width-expand_bbox_width)
                    new_bbox_top = random.randint(0, height-expand_bbox_height)
                    bbox1 =  [new_bbox_left, new_bbox_top, new_bbox_left+expand_bbox_width-1, new_bbox_top+expand_bbox_height-1]
                    ious = [bbox_iou(bbox1, bbox) for bbox in bboxes]
                    #print(max(ious))
                    if max(ious) <= threshold:
                        #write into txt file
                        fout.write(str(bbox1[0]+5)+','+str(bbox1[1]+5)+','+str(bbox_width)+','+str(bbox_height)+','+'0,'+str(class_name)+',0,0'+'\n')
                        #update bboxes list
                        bboxes.append(bbox1)
                        #update image
                        region=img.crop( (bbox_left-5, bbox_top-5, bbox_left-5+expand_bbox_width-1, bbox_top-5+expand_bbox_height-1))
                        img.paste(region, ( bbox1[0], bbox1[1]) )
                        img.save(os.path.join(img_path,img_idx+".jpg"))

def parsing(path):
    img_path = os.path.join(path,"images")
    ann_path = os.path.join(path,"annotations")
    items = os.listdir(img_path)
    items.sort()
    for ii,item in enumerate(items):
        id = item.split('.')[0]
        aug_txt(id,ann_path,img_path)
if __name__ == '__main__':
    import fire
    fire.Fire()
