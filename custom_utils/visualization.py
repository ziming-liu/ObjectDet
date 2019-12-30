import mmcv
import cv2
import os.path as osp
import os
import numpy as np
def visualization(root):
    images_path = osp.join(root,'images')
    anns_path =  osp.join(root,'annotations')
    items = os.listdir(images_path)
    for imgname in items:
        print("img:{}... ".format(imgname))
        img_id = imgname.split('.')[0]
        ann_path = osp.join(anns_path,img_id+'.txt')
        img_path = osp.join(images_path, imgname)
        img = cv2.imread(img_path)
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        with open(ann_path, 'r') as anno:
            all_info = anno.readlines()

            for item in all_info:
                values_str = item.split(',')  # list()
                bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, \
                truncation, occulusion = int(values_str[0]), int(values_str[1]), \
                                         int(values_str[2]), int(values_str[3]), float(values_str[4]), int(
                    values_str[5]), \
                                         int(values_str[6]), int(values_str[7])
                label = object_category
                # if not self.test_mode:#train set and val/test set , has different label annotation
                if label == 0 or label == 11:
                    continue
                else:
                    label = label  # 标注从1 开始
                # if self.test_mode:
                #    label = label+1
                if bbox_height < 2.0 or bbox_width < 2.0:
                    continue
                if bbox_height / bbox_width > 6.0 or bbox_width / bbox_height > 6.0:
                    continue
                # xmin  ymin  xmax  ymax
                bbox = [bbox_left, bbox_top, bbox_left + bbox_width - 1, bbox_top + bbox_height - 1]
                # 截断的 过于模糊的都会可能会造成loss波动太大， 去掉

                if score == 1 and truncation == 0 and (occulusion == 0 or occulusion == 1):
                    bboxes.append(bbox)
                    labels.append(label)
                elif score == 0 or truncation == 1 or occulusion == 2:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    raise IndexError
        assert len(labels) + len(labels_ignore) == len(bboxes) + len(bboxes_ignore), 'label is not equal to bboxes'
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2)#坐标是否需要减一？？
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) #坐标是否需要减一？？
            labels_ignore = np.array(labels_ignore)
        #assert bboxes.shape[0]>0,print(bboxes.shape)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        bboxes_sum = np.concatenate([bboxes, bboxes_ignore], axis=0).astype(np.float32)
        labels_sum = np.concatenate([labels, labels_ignore], axis=0).astype(np.int64)
        CLASSES = ['ignore region','pedestrian',
                   'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
                   'awning-tricycle', 'bus', 'motor', 'other' ]  # 12 in total
        mmcv.imshow_det_bboxes(img, bboxes_sum, labels_sum, CLASSES, show=False,
                               out_file=osp.join(root,'visualization', img_id + '.jpg'))

if __name__ == '__main__':
    import fire
    fire.Fire()