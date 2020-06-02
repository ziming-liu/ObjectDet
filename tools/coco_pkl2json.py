
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
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
        root_Path = '/'.join(args.input.split('/')[:-1])
        if not os.path.exists(root_Path):
            os.makedirs(root_Path)
        #mmcv.dump(outputs, args.out)
        outputs = mmcv.load(args.input)
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

""" 
def coco_pkl2json(pkl_path,):
    root_Path = pkl_path.split('.')[0]
    result_file = root_Path + '.json'
    outputs = mmcv.load(pkl_path)
    if not isinstance(outputs[0], dict):
    #result_file = args.out + '.json'
        results2json(dataset, outputs, result_file)
        coco_eval(result_file, eval_types, dataset.coco)
    else:
        raise TypeError("not support multi dataset")
    
if __name__ == "__main__":
    import fire
    fire.Fire()

"""