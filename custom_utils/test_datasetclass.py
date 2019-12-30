

from mmdet.datasets.visdrone import VisDroneDataset


from torch.utils.data import Dataset,DataLoader
from mmdet.datasets import get_dataset
from functools import partial

from mmcv.runner import get_dist_info
from mmcv.parallel import collate
from torch.utils.data import DataLoader
from mmdet.datasets import build_dataloader

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

train_dataset = get_dataset(dict(
type= 'VisDroneDataset',
ann_file='/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-train/img_size_anno.json',
img_prefix='/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-train',
img_scale=(1333, 800),img_norm_cfg=img_norm_cfg,
multiscale_mode='value',
size_divisor=32,
proposal_file=None,
flip_ratio=0,
with_mask=False,
with_label=False,
test_mode=False))
data_loaders = [
    build_dataloader(
        train_dataset,
        2,
        2,
        3,
        dist=False)
]
for ii,data in enumerate(data_loaders[0]):
    print(data)
