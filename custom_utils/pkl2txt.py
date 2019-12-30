import pickle
import json
import os
import sys
import subprocess
import numpy as np
#save_path = '/home/share2/VisDrone2019/TASK1/DET_results-test-challenge/'
#pkl_path = 'out_frcnn_x101.pkl'
#json_path = 'img_size_anno.json'
def pkl2txt(save_path,pkl_path,json_path):
    #print("go in")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result = pickle.load(open(pkl_path,'rb'))
    img_infos_dict = json.load(open(json_path, 'rb'))
    img_list = img_infos_dict['img_infos']
    num_imgs = len(result)
    num_classes = len(result[0])
    print("num class: {}".format(num_classes))
    assert(num_imgs == len(img_list))
    for imgidx in range(num_imgs):
        img_id = img_list[imgidx]['id']
        print("dealing with img: {}".format(img_id))
        with open(os.path.join(save_path,img_id+'.txt'),'w') as f:
            det_result_i = []
            for cls in range(num_classes):
                bbox_i = result[imgidx][cls]
                bbox_i[:,2],bbox_i[:,3] = bbox_i[:,2]-bbox_i[:,0]+1,bbox_i[:,3]-bbox_i[:,1]+1
                label = np.ones((result[imgidx][cls].shape[0],1)) * np.array([cls+1])
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
                    if jj==4:# score
                        f.write(str(det_result_i[ii][jj]))
                    else:
                        f.write(str(int(det_result_i[ii][jj])))
                    if jj==7:
                        f.write('\n')
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
    """
    print("imgs numbers ")
    print(len(results))
    print("image class ")
    print(len(results[0]))
    print("the value of img a class 1")
    print(type(results[0][0]))
    print(results[0][0])
    """
