import os, sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

dataset_dir = '/home/du/Proj/Dataset/ScanNet'
num_class = 40
scannnet_dir = f'{dataset_dir}/scannet_frames/scannet_frames_25k/'
scan_file = f'{dataset_dir}/ScanNet/Tasks/Benchmark/scannetv2_val.txt'

with open(scan_file, 'r') as f:
    scan_lis = f.readlines()
scan_lis = [x.strip() for x in scan_lis]

label_num_all = np.zeros(num_class)
for scan in tqdm(scan_lis, desc='propcessing scene...'):
    scan_dir = f'{scannnet_dir}{scan}/'
    sem_dir = glob(f'{scan_dir}/label/*.png')
    for sem_file in sem_dir:
        sem = cv2.imread(sem_file, cv2.IMREAD_UNCHANGED)
        label_lis, label_num = np.unique(sem, return_counts=True)
        for idx in range(len(label_lis)):
            if label_lis[idx]==0:
                continue
            label_num_all[label_lis[idx]-1] += label_num[idx]

label_prob = label_num_all/label_num_all.sum()

label_weights_0 = 1-label_prob

label_prob_inverse = 1/label_prob
label_weights_1 = label_prob_inverse/label_prob_inverse.sum()

plt.bar([i+1 for i in range(num_class)], label_prob)
plt.bar([i+1 for i in range(num_class)], label_weights_0)
plt.bar([i+1 for i in range(num_class)], label_weights_1)
plt.legend(['prob', '1-prob', 'prob_inverse'])
plt.show()

print('1')