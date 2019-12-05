<<<<<<< HEAD



from torchsummary import summary
from mmdet.models import IPN_kite
from mmdet.models import ResNet
kite = IPN_kite(depth=34,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    style='pytorch',)
kite = kite.cuda()
summary(kite,(3,1333,800))


=======



from torchsummary import summary
from mmdet.models import IPN_kite
from mmdet.models import ResNet
kite = IPN_kite(depth=34,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    style='pytorch',)
kite = kite.cuda()
summary(kite,(3,1333,800))


>>>>>>> 7c6038648571c1cf4ef9ee1adce503f36c4eac1d
