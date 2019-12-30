import os
import shutil
import numpy
path = '/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-train-patches-balance/annotations/'
root = '/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-train-patches-balance/'
all = os.listdir(path)
counts = {}
for i in range(12):
    counts[i] = 0
flag = {}
for i in range(12):
    flag[i] = 0
prob = {1:0.23,2:0.08,3:0.03,4:0.42,5:0.07,6:0.04,7:0.01,8:0.01,9:0.02,10:0.09}
times = {1:1,2:1,3:2,4:1,5:1,6:1,7:2,8:10,9:2,10:1}

for item in all:
    item_path = path+item
    id = item.split('.')[0]
    print("img {}".format(id))
    img_path = os.path.join(root,'images',id+'.jpg')
    ann_path = os.path.join(root,'annotations',id+'.txt')
    with open(item_path,'r') as f:
        gtbboxes = f.readlines()
        for i in range(12):
            flag[i] = 0
        for line in gtbboxes:
            cls = line.split(',')[5]
            cls_ = int(cls)
            flag[cls_] = 1
        max_copy_num = -1
        for i in range(12):
            if i ==0 or i==11:
                continue
            if flag[i] == 1:
                copy_num = times[i] - 1
                if copy_num>max_copy_num:
                    max_copy_num = copy_num

                counts[i] += 1
        for k in range(max_copy_num):
            dist_img_path = os.path.join(root, 'images', id + '_cp' + str(k) + '.jpg')
            dist_ann_path = os.path.join(root, 'annotations', id + '_cp' + str(k) + '.txt')
            shutil.copy(img_path, dist_img_path)
            shutil.copy(ann_path, dist_ann_path)

        #counts.append(len(gtbboxes))

#counts_np = numpy.array(counts)
#idx = numpy.argsort(counts_np)
#print(idx)
#print(counts)
for k,v in zip(counts.keys(),counts.values()):
    print("{} have {} \n".format(k,v))



#print("the max num is \n")
#print(max(counts))
