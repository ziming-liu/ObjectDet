import json
import os
import cv2
import numpy as np
from .pkl2txt import *
from tqdm import tqdm
import mmcv

def dropObjectsInIgr(gt, det, imgHeight, imgWidth):
    idxFr = np.where(gt[:, 5] != 0)[0]
    curgt = gt[idxFr]
    idxIgr = np.where(gt[:, 5] == 0)[0]
    igrRegion = np.maximum(gt[idxIgr, :4]-1,0)
    if len(igrRegion) > 0:
        igrMap = np.zeros((imgHeight, imgWidth))
        numIgr = len(igrRegion)
        for j in range(numIgr):
            x1 = igrRegion[j, 1]
            x2 = min(imgHeight, igrRegion[j, 1] + igrRegion[j, 3] + 2)
            y1 = igrRegion[j, 0]
            y2 = min(imgWidth,igrRegion[j,0] + igrRegion[j,2] + 2)
            igrMap[x1:x2, y1:y2] = 1
        intIgrMap = createIntImg(np.double(igrMap))
        idxLeftGt = []
        imgWidth = imgWidth-1
        imgHeight = imgHeight-1
        for i in range(len(curgt)):
            pos = np.maximum(np.round(curgt[i,:4])-1,0)#????????????????????-1?
            x = max(min(imgWidth, pos[0]),0)
            y = max(min(imgHeight, pos[1]),0)
            w = pos[2]+1
            h = pos[3]+1
            tl = intIgrMap[y, x]
            tr = intIgrMap[y, min(imgWidth, x + w)]
            bl = intIgrMap[max(0, min(imgHeight, y + h)), x]
            br = intIgrMap[max(0, min(imgHeight, y + h)), min(imgWidth, x + w)]
            igrVal = tl + br - tr - bl
            if igrVal / (h * w) < 0.5:
                idxLeftGt.append(i)
        curgt = curgt[idxLeftGt]
        idxLeftDet = []
        for i in range(len(det)):
            pos = np.maximum(np.round(det[i, :4])-1, 0).astype(np.int32)
            x = max(min(imgWidth, pos[0]),0)
            y = max(min(imgHeight, pos[1]),0)
            w = pos[2]+1
            h = pos[3]+1
            tl = intIgrMap[y, x]
            tr = intIgrMap[y, min(imgWidth, x + w)]
            bl = intIgrMap[max(0, min(imgHeight, y + h)), x]
            br = intIgrMap[max(0, min(imgHeight, y + h)), min(imgWidth, x + w)]
            igrVal = tl + br - tr - bl
            if igrVal / (h * w) < 0.5:
                idxLeftDet.append(i)
        det = det[idxLeftDet]
    return curgt,det
def createIntImg(img):
    height, width = img.shape
    #intImg = img
    for i in range(1,height):
        img[i, 0] = img[i, 0] + img[i-1, 0]
    for j in range(1,width):
        img[0, j] = img[0, j] + img[0, j-1]
    for i in range(1,height):
        for j in range(1,width):
            img[i, j] = img[i, j] + img[i-1, j] + img[i, j-1] - img[i-1, j-1]
    return img
def VOCap(rec,prec):
    mrec = np.insert(rec, 0, 0)
    mrec = np.append(mrec,1)
    mpre = np.insert(prec, 0, 0)
    mpre = np.append(mpre, 0)
    for i in range(len(mpre)-2,-1,-1):
        mpre[i] = max(mpre[i],mpre[i+1])
    i = np.where(mrec[1:]!=mrec[:-1])[0]+1
    ap = sum((mrec[i]-mrec[i-1])*mpre[i])
    return ap
def compOas(dt, gt, ig):
    m = len(dt)
    n = len(gt)
    oa = np.zeros((m, n))
    de = dt[:, 0:2]+dt[:,2:4]
    da = dt[:, 2]*dt[:, 3]
    ge = gt[:, 0:2]+gt[:,2:4]
    ga = gt[:, 2]*gt[:, 3]
    for i in range(m):
        for j in range(n):
            w = min(de[i, 0], ge[j, 0]) - max(dt[i, 0], gt[j, 0])
            if w <= 0:
                continue
            h = min(de[i, 1], ge[j, 1]) - max(dt[i, 1], gt[j, 1])
            if h <= 0:
                continue
            t = w * h
            if ig[j]:
                u = da[i]
            else:
                u = da[i] + ga[j] - t
            oa[i, j] = t / u
    return oa
def evalRes( gt0, dt0, thr):
    mul=0
    if len(gt0)==0:
        gt0=np.zeros((0, 5))
    if len(dt0)==0:
        dt0=np.zeros((0, 5))
    assert (dt0.shape[1] == 5)
    nd = dt0.shape[0]
    assert (gt0.shape[1] == 5)
    ng = gt0.shape[0]
    ord = np.argsort(-dt0[:, 4],kind='mergesort')
    dt0 = dt0[ord]
    ord = np.argsort(gt0[:, 4],kind='mergesort')
    gt0 = gt0[ord]
    gt = gt0
    dt = dt0
    dt = np.concatenate((dt,np.zeros((nd, 1))),axis=1)
    gt[:, 4]=-gt[:, 4]
    oa = compOas(dt[:,:4], gt[:,:4], gt[:, 4] == -1)
    for d in range(nd):
        bstOa =thr
        bstg = 0
        bstm = 0
        for g in range(ng):
            m = gt[g, 4]
            if m == 1 and mul==0:
                continue
            if bstm!=0 and m == -1:
                break
            if oa[d, g] < bstOa:
                continue
            bstOa = oa[d, g]
            bstg = g
            if m == 0:
                bstm = 1
            else:
                bstm = -1
        g = bstg
        m = bstm
        if m == -1:
            dt[d, 5] = m
        elif m == 1:
            gt[g, 4] = m
            dt[d, 5] = m
    return gt,dt
def calcAccuracy(numImgs, allgt, alldet):
    AP = np.zeros((10, 10))
    AR = np.zeros((10, 10, 4))
    evalClass = []
    for idClass in range(1,11):
        for idImg in range(numImgs):
            gt = allgt[idImg]
            if True in (gt[:, 5] == idClass):
                evalClass.append(idClass-1)
        x = 0
        for thr in np.arange(0.5,1,0.05):
            x = x + 1
            y = 0
            for maxDets in [1,10,100,500]:
                y = y + 1
                gtMatch = np.zeros((0,))
                detMatch = np.zeros((0,2))
                for idImg in range(numImgs):
                    gt = allgt[idImg]
                    det = alldet[idImg]
                    idxGtCurClass = np.where(gt[:, 5] == idClass)[0]
                    idxDetCurClass = np.where(det[:min(len(det), maxDets), 5] == idClass)[0]
                    gt0 = gt[idxGtCurClass, :5]
                    dt0 = det[idxDetCurClass, :5]
                    gt1, dt1 = evalRes(gt0, dt0, thr)
                    gtMatch = np.concatenate((gtMatch, gt1[:, 4]), axis=0)
                    detMatch = np.concatenate((detMatch, dt1[:, 4:6]), axis=0)
                idrank = np.argsort(-detMatch[:, 0],kind = 'mergesort')
                tp = np.cumsum(detMatch[idrank, 1] == 1)
                rec = tp / np.maximum(1, len(gtMatch))
                if len(rec) > 0:
                    AR[idClass-1, x-1, y-1] = max(rec) * 100
            fp = np.cumsum(detMatch[idrank, 1] == 0)
            prec = tp / np.maximum(0, (fp + tp))
            AP[idClass-1, x-1] = VOCap(rec, prec) * 100
    AP_all = np.mean(AP[evalClass,:])
    AP_50 = np.mean(AP[evalClass,0])
    AP_75 = np.mean(AP[evalClass, 5])
    AR_1 = np.mean(AR[evalClass,:, 0])
    AR_10 = np.mean(AR[evalClass,:, 1])
    AR_100 = np.mean(AR[evalClass, :, 2])
    AR_500 = np.mean(AR[evalClass, :, 3])
    return AP_all, AP_50, AP_75, AR_1, AR_10, AR_100, AR_500

def patchtxt2imgtxt(save_patches_path,save_path,gtPath,imgPath):
    """
    save_patches_path: the path to store txt result of patches img
    save_path: the path to stroe the txt result of origin img, which is obtained
    from patches
    gtPath: the path of origin img that store the annotations
    """
    #print(os.listdir(gtPath))
    origin_ann = os.listdir(gtPath)
    origin_ann.sort()
    #print(origin_ann)
    result_patches_txt = os.listdir(save_patches_path)
    result_patches_txt.sort()
    #print(result_patches_txt)
    for ii,name_ann_origin in enumerate(origin_ann):
        patch_anns = result_patches_txt[ii*4:(ii+1)*4]
        img_name = name_ann_origin.split('.')[0] + '.jpg'
        img = cv2.imread(imgPath+img_name)
        h,w,c = img.shape
        h_patch, w_patch = h//2, w//2
        zero_point = [[0,0],
                      [0,h_patch],
                      [w_patch,0],
                      [w_patch,h_patch]]

        with open(os.path.join(save_path,name_ann_origin),'a') as ann_ff:

            for t in range(4):
                # promise the right corresponding
                assert int(patch_anns[t].split('.')[0].split('_')[-1]) == t+1
                # read parch pred anno and transforms and write into origin img pred anno
                with open(os.path.join(save_patches_path,patch_anns[t]),'r') as patch_ff:
                    content = patch_ff.readlines()
                    for item in content:
                        values_str = item.split(',')#list()
                        bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,\
                        truncation,occulusion = int(values_str[0]),int(values_str[1]),\
                        int(values_str[2]),int(values_str[3]),float(values_str[4]),int(values_str[5]),\
                        int(values_str[6]),int(values_str[7]) # float is score
                        bbox_left = bbox_left + zero_point[t][0]
                        bbox_top  = bbox_top  + zero_point[t][1]
                        message = str(bbox_left)+','+str(bbox_top)+','+values_str[2]+\
                        ','+values_str[3]+','+values_str[4]+','+values_str[5]+','+values_str[6]+\
                        ','+values_str[7]
                        ann_ff.write(message)


def eval_visdrone_det(work_dir,pkl_path,dataset,is_patch=False,isImgDisplay=False):
    """
    is_patch: if we use patches img to test , and the result should be merged.
    isImgDisplay: if we show the imgs
    """
    save_path = os.path.join(work_dir, 'result_txt/')
    save_patches_path = os.path.join(work_dir,'result_patches_txt/')
    json_path = dataset['ann_file']
    if isImgDisplay:
        vis_path = os.path.join(work_dir,'visulization_result/')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_patches_path):
        os.makedirs(save_patches_path)

    #isImgDisplay = False

    if is_patch:
        # 处理一下地址， 用原图的标注去test，而不是patches图片的
        print("patches img path is {}".format(dataset['img_prefix']))
        datasetPathlist = dataset['img_prefix'].split('/')
        if datasetPathlist[-1]=='':
        #assert datasetPathlist[-1] == ''
            origin_prefix = datasetPathlist[-2].split('-')[:-1]
            datasetPathlist[-2] = '-'.join(origin_prefix)
        else:
            origin_prefix = datasetPathlist[-1].split('-')[:-1]
            datasetPathlist[-1] = '-'.join(origin_prefix)
        datasetPath = '/'.join(datasetPathlist)
        print("origin img path path is {}".format(datasetPath))
        assert dataset['img_prefix']!=datasetPath
    else:
        datasetPath = dataset['img_prefix']
    #datasetPath = '/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-val/'  #
    #resPath = '/home/share2/VisDrone2019/TASK1/outfile/faster_rcnn_r50_fpn_1x_visdrone_dubug/predict_anno/'  #
    #json_path =datasetPath +  'img_size_anno.json'
    gtPath = datasetPath + 'annotations/'
    imgPath = datasetPath + 'images/'

    if is_patch:
        pkl2txt(save_patches_path,pkl_path,json_path)
        if not os.listdir(save_path):# 结果文件为空才转换
            patchtxt2imgtxt(save_patches_path,save_path,gtPath,imgPath)
        else:
            print("result txt has been existing,remove it first")
            exit()
    else:
        pkl2txt(save_path,pkl_path,json_path)
    resPath = save_path
    #img_infos_dict = json.load(open(json_path, 'rb'))
    #img_list = img_infos_dict['img_infos']
    #nameImgs = [img['id'] for img in img_list]
    #width_list = [img['width'] for img in img_list]
    #height_list = [img['height'] for img in img_list]
    for root, dirs, nameImgs in os.walk(gtPath):
        numImgs=len(nameImgs)
    allgt = []
    alldet = []
    for i in tqdm(range(numImgs)):
        oldgt = []
        olddet = []
        gt_file = open(gtPath + nameImgs[i])
        det_file = open(resPath + nameImgs[i])
        for line in gt_file.readlines():
            line = line.strip()
            line=line.split(',')
            line = [int(i) for i in line]
            oldgt.append(line)
        for line in det_file.readlines():
            line = line.strip()
            line=line.split(',')
            line = [float(i) for i in line]
            olddet.append(line)


        if not oldgt:
            oldgt =  np.zeros((0, 8))
        else:
            oldgt = np.array(oldgt)
        if not olddet:
            olddet = np.zeros((0,8))
        else:
            olddet = np.array(olddet)
        assert olddet.shape[-1] ==8
        assert oldgt.shape[-1] ==8, oldgt.shape
        img = cv2.imread(imgPath + nameImgs[i][:-4] + '.jpg')
        height, width,_ = img.shape
        newgt, det = dropObjectsInIgr(oldgt, olddet, height, width)
        gt = newgt
        gt[newgt[:, 4] == 0, 4] = 1
        gt[newgt[:, 4] == 1, 4] = 0
        # if show the det result ++4.28.2019
        if isImgDisplay:

            gt_bboxes = []
            gt_labels = []
            for kk in range(len(gt)):
                gt_bboxes.append(gt[kk][:4])
                gt_labels.append(gt[kk][5])
            if gt_bboxes:
                gt_bboxes = np.array(gt_bboxes)
                gt_labels = np.array(gt_labels)
                gt_bboxes[:,2:] = gt_bboxes[:,:2] + gt_bboxes[:,2:]
                gt_img = img
                mmcv.imshow_det_bboxes(gt_img,gt_bboxes,gt_labels,show=False,out_file=vis_path+ nameImgs[i][:-4] + '_gt.jpg')

            bboxes = []
            labels = []
            for kk in range(len(det)):
                bboxes.append(det[kk][:4])
                labels.append(det[kk][5])
            if bboxes:
                bboxes = np.array(bboxes)
                labels = np.array(labels)
                bboxes[:,2:] = bboxes[:,:2] + bboxes[:,2:]
                det_img = img
                mmcv.imshow_det_bboxes(det_img,bboxes,labels,show=False,out_file=vis_path+ nameImgs[i][:-4] + '.jpg')
        allgt.append(gt)
        alldet.append(det)
    allgt = np.array(allgt)
    alldet = np.array(alldet)
    # displayImage(imgPath, numImgs, nameImgs, allgt, alldet, isImgDisplay)
    AP_all, AP_50, AP_75, AR_1, AR_10, AR_100, AR_500 = calcAccuracy(numImgs, allgt, alldet)
    print(AP_all, AP_50, AP_75, AR_1, AR_10, AR_100, AR_500)
