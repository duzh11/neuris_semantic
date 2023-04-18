import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn import preprocessing 

WALL_SEMANTIC_ID = 80
FLOOR_SEMANTIC_ID = 160

scene_num='0616_00'
img_num1='133'
img_num2='1330'
semantic_pred='/home/du/Proj/NeuRIS/exps/indoor/neus/scene'+scene_num+'/test1_semantic/semantic_npz/'
semantic_gt='/home/du/Proj/NeuRIS/Data/dataset/indoor/scene'+scene_num+'/semantic_deeplab/'
pred_name='00160000_'+img_num1+'_reso2.npz'
gt_name=img_num2+'.png'
f1=os.path.join(semantic_pred,pred_name)
f2=os.path.join(semantic_gt,gt_name)
npz1 = np.load(f1)['arr_0']
npz2 = cv2.imread(f2)

# wall_mask = npz2 == WALL_SEMANTIC_ID
# floor_mask = npz2 == FLOOR_SEMANTIC_ID
# bg_mask = ~(wall_mask | floor_mask)
# npz2[wall_mask] = 1
# npz2[floor_mask] = 2
# npz2[bg_mask] = 0

norm1=np.linalg.norm(npz1, axis=-1, ord=2,keepdims=True)
norm2=npz2

img_pred=((norm1+1e-6)*255).astype(np.uint8)
img_gt=((norm2+1e-6)*255).astype(np.uint8)

plt.figure("pred")
plt.imshow(img_pred)
plt.axis('off')
# plt.savefig(depth_pred+'00160000_'+img_num1+'_reso1.png',bbox_inches='tight',pad_inches = 0)

plt.figure("gt")
plt.imshow(img_gt)
plt.axis('off')
# plt.savefig(depth_gt+img_num2+'_render.png', bbox_inches='tight',pad_inches = 0)

plt.show()

# cv2.imwrite(depth_pred+'00160000_0140_reso1.png',img_pred)
# cv2.imwrite(depth_gt+'0140.png',img_gt)

print('complete')

