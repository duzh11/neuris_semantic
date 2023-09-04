from pathlib import Path
from glob import glob
import numpy as np
import cv2
import os

scene = 'scene0050_00'
target_idx= 0
end_idx= -1

Data_dir = os.path.join('../Data/dataset/indoor', scene)
H , W=240 , 320

out_path = f'{Data_dir}/vedio'
Path(out_path).mkdir(exist_ok=True, parents=True)

# sdf，semantic
img_path = os.path.join(Data_dir, 'image')
pred_path = os.path.join(Data_dir, 'semantic_pred_vis')
mask_path = os.path.join(Data_dir, 'semantic_consistency_0.3_vis')

img_lis= glob(f'{img_path}/*.png')
img_lis.sort()
semantic_lis = glob(f'{pred_path}/*.png')
semantic_lis.sort(key=lambda x:int((x.split('/')[-1]).split('.')[0]))
mask_lis = glob(f'{mask_path}/*.png')
mask_lis.sort(key=lambda x:int((x.split('/')[-1]).split('.')[0]))

images = []
N_image=len(img_lis)
if end_idx==-1:
    end_idx=N_image-1

for i in range(target_idx, end_idx):
    img = cv2.resize(cv2.imread(img_lis[i]), (W, H))
    semantic = cv2.resize(cv2.imread(semantic_lis[i]), (W, H))
    maks = cv2.resize(cv2.imread(mask_lis[i]), (W, H))
    img_cat=(255 * np.ones((H, 10, 3))).astype('uint8')
    lis=[img, img_cat, semantic, img_cat, maks]
    images.append(np.concatenate(lis, axis=1))

(height, width, _) = images[0].shape

video_name = f'{out_path}/img_semantic_mask_0.3.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
videowriter = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

for image in images:
    videowriter.write(image.astype(np.uint8))

videowriter.release()