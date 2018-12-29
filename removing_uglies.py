import os
import cv2
import numpy as np

def removing_uglies(check_dirs, ugly_dir):
    match = False
    for check_dir in check_dirs:
        print('[DEBUG] Num ',check_dir ,len(os.listdir(check_dir)))
        for img_name in os.listdir(check_dir):
            for ugly_img_name in os.listdir(ugly_dir):
                try:
                    ugly = cv2.imread(ugly_dir+'/'+ugly_img_name)
                    img_path = check_dir+'/'+img_name
                    img = cv2.imread(img_path)
                    if img is None:
                        os.remove(img_path)
                        print('[INFO] Deleted',img_path)
                        continue
                    if ugly.shape == img.shape and not(np.bitwise_xor(ugly,img).any()):
                        os.remove(img_path)
                        print('[INFO] Deleted',img_path)
                except Exception as e:
                    print('[ERROR] ', str(e))

if __name__ == '__main__':
    ''' Place ugly images in directory 'uglies' '''

    ugly_dir = 'raw_datasets/uglies'
    check_dirs=['raw_datasets/neg', 'raw_datasets/pos']
    removing_uglies(check_dirs, ugly_dir)

    print('[INFO] Removed ugly images Done!')
