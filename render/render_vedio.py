from pathlib import Path
from glob import glob
import numpy as np
import cv2
import os
import sys

scan = sys.argv[1]
method_name = sys.argv[2]
target_idx= int(sys.argv[3])
end_idx= int(sys.argv[4])

H , W=240 , 320

exps_dir = f'/home/du/Proj/3Dv_Reconstruction/NeuRIS/exps/indoor/neus/{method_name}/{scan}'

out_path = f'{exps_dir}/rendering/vedio'
Path(out_path).mkdir(exist_ok=True, parents=True)

# sdf，semantic
mesh_path = os.path.join(exps_dir, 'rendering', 'mesh')
render_semantic = sorted(glob(os.path.join(exps_dir, 'semantic/fine', '00160000_*_reso2.png')))

images = []
N_image=len(render_semantic)
if end_idx==-1:
    end_idx=N_image-1

for i in range(target_idx, end_idx):
    mesh=cv2.resize(cv2.imread(os.path.join(mesh_path, f'render_{i}.jpg')), (W, H))
    semantic=cv2.resize(cv2.imread(render_semantic[i]), (W, H))
    img_cat=(255 * np.ones((H, 10, 3))).astype('uint8')
    lis=[mesh, img_cat, semantic]
    images.append(np.concatenate(lis, axis=1))

(height, width, _) = images[0].shape

video_name = f'{out_path}/vedio.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
videowriter = cv2.VideoWriter(video_name, fourcc, 15, (width, height))

for image in images:
    videowriter.write(image.astype(np.uint8))

videowriter.release()
