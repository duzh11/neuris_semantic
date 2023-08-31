# find correspondence (x,y) in another pic
import matplotlib.pyplot as plt

import numpy as np
import cv2
import os


def read_file(file):
    data = open(file)

    lines = [[float(w) for w in line.strip().split()] for line in data]
    assert len(lines) == 4
    return np.array(lines).astype(np.float32)  

def find_correspondence(scene_name, source_id, target_id, x, y):
    data_dir = f'/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/dataset/indoor/{scene_name}'

    image_source = cv2.imread(os.path.join(data_dir, 'image', f'{source_id}.png'))
    image_target = cv2.imread(os.path.join(data_dir, 'image', f'{target_id}.png'))

    intrinsic = read_file(os.path.join(data_dir, 'intrinsic_color_crop1248_resize640.txt'))[:3,:3]
    # c2w
    c2w_source=read_file(os.path.join(data_dir, 'pose', f'{source_id}.txt'))
    c2w_target=read_file(os.path.join(data_dir, 'pose', f'{target_id}.txt'))
    # w2c
    w2c_target = np.linalg.inv(c2w_target)
    # source to target
    source_c = np.linalg.inv(intrinsic)@np.array([x, y, 1])
    source_c_h = np.append(source_c, 1)
    target_c = (w2c_target@(c2w_source@source_c_h))[:3]

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
    source_id='0130'
    target_id='0140'
    x, y= 284, 141
    find_correspondence(scene_name, source_id, target_id, 2*x, 2*y)
    plt.show()








