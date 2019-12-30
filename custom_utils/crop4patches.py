import numpy
import os
import json
import cv2
import csv
import os.path as osp
import mmcv
import numpy as np

def isgood(w,h):
    if w<2 or h<2:
        return  False
    if w /h >10.0 or h/w >10.0:
        return False
    return True

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

def save_newanno(message, path):
    with open(path,'a') as ann: # 追加模式
        ann.write(message)
        ann.write('\n')


def crop4patches(img_prefix,img_writen,istrain=True):

    if not os.path.exists(img_writen+'annotations/'):
        os.makedirs(img_writen+'annotations/')
    if not os.path.exists(img_writen+'images/'):
        os.makedirs(img_writen+'images/')
    img_infos = []
    img_file = img_prefix+'images/'
    all_imgs_files = os.listdir(img_file)
    for img_file in all_imgs_files:
        img_id = img_file.split('.')[0]
        anno_name ='annotations/{}.txt'.format(img_id)
        img_name = 'images/{}.jpg'.format(img_id)
        #p rint(filename)
        print("dealing with {}".format(img_name))
        img_path = osp.join(img_prefix, img_name)
        anno_path = osp.join(img_prefix,anno_name)

        img = cv2.imread(img_path)
        h,w,c = img.shape
        print("h {}".format(h))
        print("w {}".format(w))

        patch_width = int(w) // 2
        patch_height = int(h) // 2
        bboxes = []
        bboxes.append(np.array([0,0,patch_width,patch_height]))
        bboxes.append(np.array([0,patch_height,patch_width,h]))
        bboxes.append(np.array([patch_width,0,w,patch_height]))
        bboxes.append(np.array([patch_width,patch_height,w,h]))
        padw = (w-patch_width)//2
        padh = (h-patch_height)//2
        if istrain:
            bboxes.append(np.array([padw,padh,w-padw,h-padh]))
        bboxes = np.array(bboxes)
        img_patches = mmcv.imcrop(img,bboxes,scale=1.0)
        for jj in range(len(img_patches)):
            if istrain:
                assert  (len(img_patches)) == 5
            else:
                assert  (len(img_patches)) == 4
            cv2.imwrite(img_writen+'images/{}_{}.jpg'.format(img_id,jj+1),img_patches[jj])

        with open(anno_path,'r') as ann:
            note = ann.readlines()
            # 计算中心 patch的标注
            if istrain:
                for item in note:
                    values_str = item.split(',')#list()
                    bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,\
                    truncation,occulusion = int(values_str[0]),int(values_str[1]),\
                    int(values_str[2]),int(values_str[3]),int(values_str[4]),int(values_str[5]),\
                    int(values_str[6]),int(values_str[7])
                    # in central patch
                    if bbox_left>padw and bbox_top>padh and bbox_left<w-padw and bbox_top < h-padh:

                        if bbox_left+bbox_width>w-padw or bbox_top+bbox_height>h-padh:
                            if bbox_iou((bbox_left,bbox_top,bbox_left+bbox_width,bbox_top+bbox_height),\
                                        (bbox_left,bbox_top,min(w-padw,bbox_left+bbox_width),min(h-padh,bbox_top+bbox_height))) > 0.5:
                                message = str(bbox_left-padw)+','+str(bbox_top-padh)+','+str(min(w-padw,bbox_left+bbox_width)-bbox_left)+','+str(min(h-padh,bbox_top+bbox_height)-bbox_top)\
                                +','+str(score)+','+str(object_category)+','+str(1)+','+str(occulusion)
                                path = img_writen+'annotations/{}_{}.txt'.format(img_id,5)
                                save_newanno(message,path)
                                continue
                            else:
                                continue
                        else:
                            message = str(bbox_left-padw)+','+str(bbox_top-padh)+','+str(min(w-padw,bbox_left+bbox_width)-bbox_left)+','+str(min(h-padh,bbox_top+bbox_height)-bbox_top)\
                            +','+str(score)+','+str(object_category)+','+str(truncation)+','+str(occulusion)
                            path = img_writen+'annotations/{}_{}.txt'.format(img_id,5)
                            #print("5loc {}".format(message))
                            save_newanno(message,path)
                            continue


            for item in note:
                values_str = item.split(',')#list()
                bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,\
                truncation,occulusion = int(values_str[0]),int(values_str[1]),\
                int(values_str[2]),int(values_str[3]),int(values_str[4]),int(values_str[5]),\
                int(values_str[6]),int(values_str[7])

                if bbox_left < patch_width and bbox_top < patch_height:# zuoshang
                    if bbox_left+bbox_width> patch_width or bbox_top+bbox_height > patch_height:# outline
                        if bbox_iou((bbox_left,bbox_top,bbox_left+bbox_width,bbox_top+bbox_height),\
                                    (bbox_left,bbox_top,min(patch_width,bbox_left+bbox_width),min(patch_height,bbox_top+bbox_height))) > 0.5:
                            #save
                            message = str(bbox_left-0)+','+str(bbox_top-0)+','+str(min(patch_width,bbox_left+bbox_width)-bbox_left)+','+str(min(patch_height,bbox_top+bbox_height)-bbox_top)\
                            +','+str(score)+','+str(object_category)+','+str(1)+','+str(occulusion)
                            path = img_writen+'annotations/{}_{}.txt'.format(img_id,1)
                            save_newanno(message,path)
                            continue
                        else:# dont save
                            continue
                    else: # 完整直接save
                        message = str(bbox_left-0)+','+str(bbox_top-0)+','+str(min(patch_width,bbox_left+bbox_width)-bbox_left)+','+str(min(patch_height,bbox_top+bbox_height)-bbox_top)\
                        +','+str(score)+','+str(object_category)+','+str(truncation)+','+str(occulusion)
                        path = img_writen+'annotations/{}_{}.txt'.format(img_id,1)
                        save_newanno(message,path)
                        #print("1loc {}".format(message))
                        continue
                #zuoxia
                if bbox_left< patch_width and bbox_top >= patch_height:
                    if bbox_top+bbox_height > h:# 原本标注错误
                        raise IOError
                    if bbox_left+bbox_width > patch_width:# outline
                        if bbox_iou((bbox_left,bbox_top,bbox_left+bbox_width,bbox_top+bbox_height),\
                (bbox_left,bbox_top,min(patch_width,bbox_left+bbox_width),min(patch_height,bbox_top+bbox_height))) > 0.5:
                            #save
                            message = str(bbox_left-0)+','+str(bbox_top-patch_height)+','+str(min(patch_width,bbox_left+bbox_width)-bbox_left)+','+str(min(h,bbox_top+bbox_height)-bbox_top)\
                            +','+str(score)+','+str(object_category)+','+str(1)+','+str(occulusion)
                            path = img_writen+'annotations/{}_{}.txt'.format(img_id,2)
                            save_newanno(message,path)
                            continue
                        else:# dont save
                            continue
                    else:
                        #save
                        message = str(bbox_left-0)+','+str(bbox_top-patch_height)+','+str(min(patch_width,bbox_left+bbox_width)-bbox_left)+','+str(min(h,bbox_top+bbox_height)-bbox_top)\
                        +','+str(score)+','+str(object_category)+','+str(truncation)+','+str(occulusion)
                        path = img_writen+'annotations/{}_{}.txt'.format(img_id,2)
                        save_newanno(message,path)
                        #print("2loc {}".format(message))
                        continue
                #youshang
                if bbox_left >= patch_width and bbox_top < patch_height:
                    if bbox_left + bbox_width > w:
                        raise IOError
                    if bbox_top + bbox_height > patch_height:# outline
                        if bbox_iou((bbox_left,bbox_top,bbox_left+bbox_width,bbox_top+bbox_height),\
                                    (bbox_left,bbox_top,min(patch_width,bbox_left+bbox_width),min(patch_height,bbox_top+bbox_height))) > 0.5:
                            #save
                            message = str(bbox_left-patch_width)+','+str(bbox_top-0)+','+str(min(w,bbox_left+bbox_width)-bbox_left)+','+str(min(patch_height,bbox_top+bbox_height)-bbox_top)\
                            +','+str(score)+','+str(object_category)+','+str(1)+','+str(occulusion)# must trucncation
                            path = img_writen+'annotations/{}_{}.txt'.format(img_id,3)
                            save_newanno(message,path)
                            continue
                        else:# dont save
                            continue
                    else:
                        #save
                        message = str(bbox_left-patch_width)+','+str(bbox_top-0)+','+str(min(w,bbox_left+bbox_width)-bbox_left)+','+str(min(patch_height,bbox_top+bbox_height)-bbox_top)\
                        +','+str(score)+','+str(object_category)+','+str(truncation)+','+str(occulusion)
                        path = img_writen+'annotations/{}_{}.txt'.format(img_id,3)
                        save_newanno(message,path)
                        #print("3loc {}".format(message))
                        continue
                # youxia
                if bbox_left >= patch_width and bbox_top >= patch_height:
                    if bbox_left+bbox_width>w or bbox_height+bbox_top>h:
                        raise  IOError
                    # 第四个区域不会有 outline
                    message = str(bbox_left-patch_width)+','+str(bbox_top-patch_height)+','+str(bbox_width)+','+str(bbox_height)\
                    +','+str(score)+','+str(object_category)+','+str(truncation)+','+str(occulusion)
                    path = img_writen+'annotations/{}_{}.txt'.format(img_id,4)
                    save_newanno(message,path)
                    #print("4loc {}".format(message))
                    continue
        #check if the image has no annotaion , delet it
        for jj in range(len(img_patches)):
            if istrain:
                assert  (len(img_patches)) == 5
            else:
                assert  (len(img_patches)) == 4
            if not os.path.exists(img_writen+'annotations/{}_{}.txt'.format(img_id,jj+1)):
                os.remove(img_writen+'images/{}_{}.jpg'.format(img_id,jj+1))
                #path = img_writen+'annotations/{}_{}.txt'.format(img_id,jj+1)
                #with open(path,'w') as ann: # 追加模式
                 #   pass
                #print("empty {}".format('annotations/{}_{}.jpg'.format(img_id,jj+1)))

    new_list = os.listdir(img_writen+'images/')
    new_list_show = []
    new_list_show.extend(new_list[:100])
    new_list_show.extend(new_list[500:600])
    for ii,item in enumerate(new_list_show):
        showimg = cv2.imread(img_writen+'images/'+item)
        id = item.split('.')[0]
        annotation  = img_writen+'annotations/'+id+'.txt'
        #if not os.path.exists(annotation):
        #    continue
        with open(annotation,'r') as ann:
            note = ann.readlines()
            bboxes = []
            for jj in note:
                values_str = jj.split(',')#list()
                bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,\
                truncation,occulusion = int(values_str[0]),int(values_str[1]),\
                int(values_str[2]),int(values_str[3]),int(values_str[4]),int(values_str[5]),\
                int(values_str[6]),int(values_str[7])
                bboxes.append(np.array([bbox_left,bbox_top,bbox_left+bbox_width,bbox_top+bbox_height]))
            bboxes = np.array(bboxes)
            print('/home/share2/VisDrone2019/vispatch/'+item)
            if istrain:
                mmcv.imshow_bboxes(showimg,bboxes,show=False,out_file='/home/share2/VisDrone2019/TASK1/trainpatch/'+item)
            else:
                mmcv.imshow_bboxes(showimg,bboxes,show=False,out_file='/home/share2/VisDrone2019/TASK1/valpatch/'+item)








if __name__ == '__main__':
    import fire
    fire.Fire()
    #img_prefix = '/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-val/'
    #img_writen= '/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-val-patches/'
    #crop4patches(img_prefix=img_prefix,img_writen=img_writen,istrain=False)
