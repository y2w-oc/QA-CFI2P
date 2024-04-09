import os
import os.path
import numpy as np
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    root_path = '/home/yao/workspace/Raw'
    save_path = '../../kitti'
    seq_list = range(0, 22)

    is_save_img = True
    is_save_pc = False
    
    calib_raw = root_path + '/' + 'data_odometry_calib/dataset/sequences/.'
    calib_save = save_path + '/calib'
    cmd = "cp -r " + calib_raw + " " + calib_save
    os.system(cmd)


    for seq in seq_list:
        img2_folder = os.path.join(root_path, 'data_odometry_color', 'sequences', '%02d' % seq, 'image_2')
        img3_folder = os.path.join(root_path, 'data_odometry_color', 'sequences', '%02d' % seq, 'image_3')
        sample_num = round(len(os.listdir(img2_folder)))

        img2_folder_npy = os.path.join(save_path, 'data_odometry_color_npy', 'sequences', '%02d' % seq, 'image_2')
        img3_folder_npy = os.path.join(save_path, 'data_odometry_color_npy', 'sequences', '%02d' % seq, 'image_3')
        if os.path.isdir(img2_folder_npy) == False:
            os.makedirs(img2_folder_npy)
        if os.path.isdir(img3_folder_npy) == False:
            os.makedirs(img3_folder_npy)


        for i in range(sample_num):
            print('working on seq %d - image %d' % (seq, i))

            # ----------- png -------------
            img2_path = os.path.join(img2_folder, '%06d.png' % i)
            img3_path = os.path.join(img3_folder, '%06d.png' % i)

            img2 = cv2.imread(img2_path)
            img2 = img2[:, :, ::-1]  # HxWx3

            img3 = cv2.imread(img3_path)
            img3 = img3[:, :, ::-1]  # HxWx3

            if is_save_img:
                np.save(os.path.join(img2_folder_npy, '%06d.npy' % i), img2)
                np.save(os.path.join(img3_folder_npy, '%06d.npy' % i), img3)

