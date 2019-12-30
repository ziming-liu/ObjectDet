import os


def get_mini_dataset(origin_set,new_set):
    origin_num = 0
    mini_num = 0
    origin_img = os.path.join(origin_set,"images")
    origin_ann = os.path.join(origin_set,"annotations")
    new_img = os.path.join(new_set,"images")
    new_ann = os.path.join(new_set,"annotations")
    if  not os.path.exists(new_img):
        os.makedirs(new_img)
    if  not os.path.exists(new_ann):
        os.makedirs(new_ann)

    items = os.listdir(origin_img)
    items.sort()
    assert len(items) > 0
    for ii,item in enumerate(items):
        id = item.split('.')[0]
        print("id :: {} ".format(id))
        ann_item = id+'.txt'
        origin_num = origin_num + 1
        if ii%10 ==0:
            mini_num = mini_num + 1
            sourceFile = os.path.join(origin_img,item)
            targetFile = os.path.join(new_img,item)
            open(targetFile, "wb").write(open(sourceFile, "rb").read())
            sourceFile = os.path.join(origin_ann, ann_item)
            targetFile = os.path.join(new_ann, ann_item)
            open(targetFile, "wb").write(open(sourceFile, "rb").read())

    print("origin num {}".format(origin_num))
    print("mini num {}".format(mini_num))
if __name__ == '__main__':
    import fire
    fire.Fire()