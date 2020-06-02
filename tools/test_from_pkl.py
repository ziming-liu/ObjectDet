'''
@Author: your name
@Date: 2020-01-17 18:37:05
@LastEditTime : 2020-01-18 20:17:11
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \objdet_old_machine\tools\test_from_pkl.py
'''
import argparse
import os
import os.path as osp
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

import os
import sys
sys.path.append("..")
from custom_utils import eval_visdrone_det

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    #parser.add_argument('config', help='test config file path')
    #parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input', help='input result file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--is_coco', action='store_true', help='coco results')
    parser.add_argument('--is_patch',action='store_true', help='patch to test or not')
    parser.add_argument('--tmpdir', default='./tmp/', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    assert type(args.input) is list()
    N = len(args.input)
    for ii in range(N):
        if args.input[ii] is not None and not args.input[ii].endswith(('.pkl', '.pickle')):
            raise ValueError('The input file must be a pkl file.')
    
    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    

    rank, _ = get_dist_info()
    if args.input and rank == 0:
        print('\n results is {}'.format(args.input))
        root_Path = '/'.join(args.out[0].split('/')[:-1])
        if not os.path.exists(root_Path):
            os.makedirs(root_Path)
        #mmcv.dump(outputs, args.out)
        outputs = list()
        print("{} models ".format(N))
        for jj in range(N):
            input = mmcv.load(args.input[jj])
            print("{} images".format(len(input)))
            for zz in range(len(input)):
                if jj==0:
                    outputs.append(input[zz])
                else:
                    assert len(outputs[zz]) == len(input[zz])
                    outputs[zz].extend(input[zz])
        mmcv.dump(outputs, args.out) 
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if not args.is_coco:
                #  test VisDrone2019
                if args.eval == ['bbox']:
                    print("eval {}".format(args.eval))
                    test_dataset = cfg.data.test
                    eval_visdrone_det(cfg.work_dir, args.out, test_dataset,args.is_patch,args.show)
            else:
                if eval_types == ['proposal_fast']:
                    result_file = args.out
                    coco_eval(result_file, eval_types, dataset.coco)
                else:
                    if not isinstance(outputs[0], dict):
                        result_file = args.out + '.json'
                        results2json(dataset, outputs, result_file)
                        coco_eval(result_file, eval_types, dataset.coco)
                    else:
                        for name in outputs[0]:
                            print('\nEvaluating {}'.format(name))
                            outputs_ = [out[name] for out in outputs]
                            result_file = args.out + '.{}.json'.format(name)
                            results2json(dataset, outputs_, result_file)
                            coco_eval(result_file, eval_types, dataset.coco)



if __name__ == '__main__':
    main()
