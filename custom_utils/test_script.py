import os
def patchtxt2imgtxt(save_patches_path,save_path,gtPath):
    """
    save_patches_path: the path to store txt result of patches img
    save_path: the path to stroe the txt result of origin img, which is obtained
    from patches
    gtPath: the path of origin img that store the annotations
    """
    print(os.listdir(gtPath))
    origin_ann = os.listdir(gtPath)
    origin_ann.sort()
    print(origin_ann)
    result_patches_txt = os.listdir(save_patches_path)
    result_patches_txt.sort()
    print(result_patches_txt)
    for ii,name_ann_origin in enumerate(origin_ann):
        patch_anns = result_patches_txt[ii*4:(ii+1)*4]
        with open(os.path.join(save_path,name_ann_origin),'a') as ann_ff:

            for t in range(4):
                # promise the right corresponding
                assert int(patch_anns[t].split('.')[0].split('_')[-1]) == t+1
                with open(os.path.join(save_patches_path,patch_anns[t]),'r') as patch_ff:
                    content = patch_ff.readlines()
                    ann_ff.writelines(content)
if __name__ == '__main__':
    #patchtxt2imgtxt("origin/","new/","gt/")
    path = '/home/share2/VisDrone2019/TASK1/VisDrone2019-DET-val-patches/'
    print(path)
    datasetPathlist = path.split('/')
    assert datasetPathlist[-1] == ''
    origin_prefix = datasetPathlist[-2].split('-')[:-1]
    datasetPathlist[-2] = '-'.join(origin_prefix)
    datasetPath = '/'.join(datasetPathlist)
    print(datasetPath)
