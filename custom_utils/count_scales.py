import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def count_scales(ann_path,img_path):
    all_ann = os.listdir(ann_path)
    count_small = 0
    count_mid = 0
    count_large = 0
    count_very_small = 0
    count_extremly_small = 0
    all_fraction =[]
    for ii, ann_name in enumerate(all_ann):
        id = ann_name.split('.')[0]
        img_name = id + '.jpg'
        imreadpath = os.path.join(img_path,img_name)
        img = cv2.imread(imreadpath)
        h,w,_ = img.shape
        area_img = h*w
        single_path  = os.path.join(ann_path,ann_name)
        with open(single_path,'r') as ff:
            annotaions_items = ff.readlines()
            for item in annotaions_items:
                values_str = item.split(',')#list()
                bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,\
                truncation,occulusion = int(values_str[0]),int(values_str[1]),\
                int(values_str[2]),int(values_str[3]),float(values_str[4]),int(values_str[5]),\
                int(values_str[6]),int(values_str[7]) # float is score
                area = bbox_height * bbox_width
                fraction = area / area_img
                all_fraction.append(fraction)
                if area<100:
                    count_extremly_small = count_extremly_small +1
                if area>=100 and area<225:
                    count_very_small = count_very_small +1
                if area < 322 and area >=0:
                    count_small = count_small+1
                elif area>=322 and area < 962:
                    count_mid = count_mid +1
                elif area >= 962:
                    count_large = count_large +1
                else:
                    print("area is {}".format(area))
                    raise IOError
    print("count_small     :         area < 100:  edge:10      | {}".format(count_extremly_small))
    print("count_very_small:    100<=area < 225:  edge:15      | {}".format(count_very_small))

    print("count_small     :         area < 322:  edge:17.9    | {}".format(count_small))
    print("count_mid       :   322 <=area< 962 :  edge:17.9-31 | {}".format(count_mid))
    print("count_large     :   962<= area      :  edge:31.0    | {}".format(count_large))

    std = [(i+1)*0.0001 for i in range(1000)]
    num_each_scale = [0 for i in range(1000)]
    col =  [0 for i in range(1000)]
    num_obj = len(all_fraction)
    for jj in range(len(std)):
        for f in all_fraction:
            if f>=std[jj]-0.0001 and f<std[jj]:
                num_each_scale[jj] = num_each_scale[jj] +1

        if jj==0:
            col[jj] = num_each_scale[jj] / num_obj
        else:
            col[jj] = num_each_scale[jj] / num_obj + col[jj-1]
    x = np.array(std)
    y = np.array(col)
    print("fraction of diff scales")
    with open("./scaleinfo.txt",'w') as ff:
        ff.write("scale of xlabel: \n")
        ff.write(str(std))
        ff.write("scale of y label: \n")
        ff.write(str(col))
        ff.write("numeachscale: \n")
        ff.write(str(num_each_scale))
        ff.write("all_fraction: \n")
        ff.write(str(all_fraction))
    print(x)
    print(y)
    plt.figure(figsize=(10,5))
    plt.plot(x,y,color="red",linewidth=2)
    plt.xlabel("relative scale")
    plt.ylabel("fraction of some scale")
    plt.savefig("./scale_vis_0.0001.jpg")


if __name__ == '__main__':
    import fire
    fire.Fire(count_scales)
