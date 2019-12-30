import pickle
import json
import os
import sys
import subprocess
import numpy as np
from argparse import ArgumentParser

import mmcv
import numpy as np
import sys
sys.path.append('..')
import datasets
from core import eval_map

#save_path = '/home/share2/VisDrone2019/TASK1/DET_results-test-challenge/'
#pkl_path = 'out_frcnn_x101.pkl'
#json_path = 'img_size_anno.json'
def pkl2txt(save_path,pkl_path,dataset,json_path):
    print("go in")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result = pickle.load(open(pkl_path,'rb'))
    img_infos = dataset.load_annotations(json_path)
    img_list = img_infos
    num_imgs = len(result)
    num_classes = len(result[0])
    print("num class: {}".format(num_classes))
    assert(num_imgs == len(img_list))
    for ii in range(len(dataset)):
        #If rescale is False, then returned bboxes and masks will fit the scale
        #of imgs[0].
        scale_factor = dataset.prepare_test_img(ii)['img_meta'][0][0]['scale_factor']
        print(scale_factor)
        #['img_meta']['scale_factor']

        img_id = img_list[ii]['id']
        print("dealing with img: {}".format(img_id))
        with open(os.path.join(save_path,img_id+'.txt'),'w') as f:
            det_result_i = []
            for cls in range(num_classes):
                bbox_i = result[ii][cls]
                #print(type(bbox_i))
                #print(bbox_i.shape)
                #print(bbox_i[:,1].shape)
                #print(bbox_i[:,2].shape)
                #print(bbox_i[:,3].shape)
                bbox_i[:,:4] = bbox_i[:,:4] / scale_factor
                bbox_i[:,2]= bbox_i[:,2]-bbox_i[:,0]+1
                bbox_i[:,3]= bbox_i[:,3]-bbox_i[:,1]+1
                label = np.ones((result[ii][cls].shape[0],1)) * np.array([cls+1])
                label = label.astype(np.int64)
                det_result_i.append(np.concatenate([bbox_i,label],1))
            det_result_i = np.concatenate(det_result_i,0).astype(np.float32)
            ignore =  np.ones((det_result_i.shape[0],2)) * np.array([-1])
            ignore = ignore.astype(np.int64)
            det_result_i = np.concatenate([det_result_i,ignore],1)
            assert det_result_i.shape[1] == 8
            row,col = det_result_i.shape
            for ii in range(row):
                for jj in range(col):
                    f.write(str(det_result_i[ii][jj]))
                    if jj==7:
                        f.write('\r\n')
                    else:
                        f.write(',')


"""
    for i in range(num_imgs):
        img_id = img_list[i]['id']
        print("dealing with img: {}".format(img_id))
        file = open(save_path + img_id+'.txt', 'w')
        for j in range(num_classes):
            if result[i][j].tolist() == None:
                continue
            for out in result[i][j].tolist():
                out.extend([j+1,-1,-1])
                file.write(str(out).strip('[]')+'\n')
        file.close()
"""
if __name__=="__main__":
    """
    example::
    ./custom_utils/pkl2txt.py /home/share2/VisDrone2019/TASK1/outfile/faster_rcnn_r50_fpn_1x_visdrone_dubug/predict_anno/     /home/share2/VisDrone2019/TASK1/outfile/out_faster_x50_epoch8.pkl val

    """
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('save_path', help='save file path')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    json_path = '/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-val/img_size_anno.json'

    pkl2txt(args.save_path,args.result, test_dataset,json_path)


    """
    save_path =  sys.argv[1]
    pkl_path  = sys.argv[2]
    mode = sys.argv[3]
    if mode =='val':
        json_path = '/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-val/img_size_anno.json'
    elif mode=='test':
        json_path = '/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-test-challenge/img_size_anno.json'
    else:
        raise IOError
    print("Starting")
    pkl2txt(save_path,pkl_path,json_path)
    #results = pickle.load(open(pkl_path,'rb'))




    print("imgs numbers ")
    print(len(results))
    print("image class ")
    print(len(results[0]))
    print("the value of img a class 1")
    print(type(results[0][0]))
    print(results[0][0])
    """
