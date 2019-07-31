from .resnet import ResNet, make_res_layer
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .hrnet import HRNet
from .deep_shallow_net import DeShNet
from .deshnet import DSNet
from .bideshnet import BiDSNet
__all__ = ['ResNet', 'make_res_layer', 'ResNeXt', 'SSDVGG', 'HRNet','BiDSNet', 'DeShNet','DSNet']
