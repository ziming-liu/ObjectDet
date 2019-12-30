import os
import numpy
path = '/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-train-balance/annotations/'
all = os.listdir(path)
counts = {}
for i in range(12):
    counts[i] = 0


for item in all:
    item_path = path+item
    with open(item_path,'r') as f:
        gtbboxes = f.readlines()
        for line in gtbboxes:
            cls = line.split(',')[5]
            cls_ = int(cls)
            counts[cls_] +=1

        #counts.append(len(gtbboxes))

#counts_np = numpy.array(counts)
#idx = numpy.argsort(counts_np)
#print(idx)
#print(counts)
sum  = 0
for k,v in zip(counts.keys(),counts.values()):
    if k==0 or k==11:
        continue
    print("c{} have {} \n".format(k,v))
    sum += v
for k,v in zip(counts.keys(),counts.values()):
    if k==0 or k==11:
        continue
    print("c{} prob {} \n".format(k,v/sum))




#print("the max num is \n")
#print(max(counts))
