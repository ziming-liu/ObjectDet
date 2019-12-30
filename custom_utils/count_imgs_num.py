import os
import numpy
path = '/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-train/annotations/'
all = os.listdir(path)
counts = {}
for i in range(12):
    counts[i] = 0
flag = {}
for i in range(12):
    flag[i] = 0

for item in all:
    item_path = path+item
    with open(item_path,'r') as f:
        gtbboxes = f.readlines()
        for i in range(12):
            flag[i] = 0
        for line in gtbboxes:
            cls = line.split(',')[5]
            cls_ = int(cls)
            flag[cls_] = 1
        for i in range(12):
            if flag[i] == 1:
                counts[i] += 1

        #counts.append(len(gtbboxes))

#counts_np = numpy.array(counts)
#idx = numpy.argsort(counts_np)
#print(idx)
#print(counts)
for k,v in zip(counts.keys(),counts.values()):
    print("{} have {} \n".format(k,v))



#print("the max num is \n")
#print(max(counts))
