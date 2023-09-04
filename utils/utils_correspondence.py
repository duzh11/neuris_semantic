# find correspondence (x,y) in another pic
import matplotlib.pyplot as plt

import numpy as np
import cv2
import os


def find_correspondence(scene_name, source_id, target_id, x, y):
    data_dir = f'/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/dataset/indoor/{scene_name}'

    image_source = cv2.imread(os.path.join(data_dir, 'image', f'{source_id}.png'))
    depth_source = cv2.imread(os.path.join(data_dir, 'depth', f'{source_id}.png'), cv2.IMREAD_UNCHANGED)/1000
    image_target = cv2.imread(os.path.join(data_dir, 'image', f'{target_id}.png'))
    intrinsic = np.loadtxt(os.path.join(data_dir, 'intrinsic_color_crop1248_resize640.txt'))
    # c2w
    c2w_source=np.loadtxt(os.path.join(data_dir, 'pose', f'{source_id}.txt'))
    c2w_target=np.loadtxt(os.path.join(data_dir, 'pose', f'{target_id}.txt'))
    # w2c
    w2c_target = np.linalg.inv(c2w_target)
    # source to target
    coor= np.array([depth_source[y, x]*x, depth_source[y, x]*y, depth_source[y, x], 1])
    source_c = np.linalg.inv(intrinsic)@coor
    target_c = (w2c_target@(c2w_source@source_c))

    result = intrinsic@target_c
    (target_x, target_y) = (result/result[2])[:2]
    print('correspondence:', (round(target_x), round(target_y)))
    
    plt.close('all')
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image_source[...,::-1])
    ax[0].plot(x, y, 'ro', markersize=5)

    ax[1].imshow(image_target[...,::-1])
    ax[1].plot(target_x, target_y, 'ro', markersize=5)

    plt.tight_layout()
    # plt.show()
    
    return round(target_x), round(target_y)

if __name__=='__main__':
    scene_name='scene0616_00'
    source_id='0000'
    target_id='0010'
    x, y= 200, 60
    find_correspondence(scene_name, source_id, target_id, 2*x, 2*y)
    plt.show()








