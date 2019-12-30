
import numpy
import os
import json
import cv2
import csv
import os.path as osp
def load_annotations(sequence_file,img_prefix):
    """
    for the visdrone, anno_file is the 'path' where the annotation items are.
    """
    videos = os.listdir(sequence_file)
    videos.sort()
    img_infos = []
    for video in videos:
        video_path = os.path.join(sequence_file,video)
        video_name = video

        all_imgs_files = os.listdir(video_path)
        all_imgs_files.sort()

        #print(all_anno_files)
        #img_ids = mmcv.list_from_file(ann_file)
        for img_file in all_imgs_files:
            #print(anno_file)
            img_id = img_file.split('.')[0]
            #print(img_id)
            filename ='sequences/'+video_name+ '/{}.jpg'.format(img_id)
            #print(filename)

            img_path = osp.join(img_prefix, filename)
            print(img_path)
            img = cv2.imread(img_path)
            h,w,c = img.shape
            width = int(w)
            height = int(h)

            img_infos.append(
                dict(id=img_id, video_id=video_name,filename=filename, width=width, height=height))

    return img_infos

def get_json_video(imgs_path,json_path):
    result = dict()
    img_infos = load_annotations(imgs_path, json_path)
    result['img_infos'] = img_infos
    with open(json_path + 'img_size_anno.json', 'w') as dst_file:
        json.dump(result, dst_file)
    #with open(json_path + 'img_size_anno.csv', 'w') as dst_file:
    #    csv.writer(dst_file, img_infos)
if __name__ == '__main__':
    """
    generate an json annotaion of for video's img size , width and bbox_height
    """
    import fire
    fire.Fire()
    #imgs_path = '/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-val-patches/images/'
    #imgs_path = '/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-val-patches/'
