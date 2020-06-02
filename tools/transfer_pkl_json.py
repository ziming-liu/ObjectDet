'''
@Author: ziming
@Date: 2020-01-18 13:13:23
@LastEditTime : 2020-01-18 13:28:32
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \objdet_old\tools\transfer_pkl_json.py
'''
import os
import mmcv
import fire
import numpy
import json
def transfer(pkl_path,json_path):
    results = mmcv.load(pkl_path)
    print(type(results))
    for img_idx in range(len(results)):
        img = results[img_idx]
        for clss_idx in range(len(img)):
            clss = img[clss_idx]
            row,col = clss.shape
            #for row_idx in range(row):
            results[img_idx][clss_idx] = results[img_idx][clss_idx].tolist()
    mmcv.dump(results,json_path)

if __name__ == "__main__":
    fire.Fire()