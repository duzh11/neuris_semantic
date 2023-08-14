import cv2
import os
import numpy as np

img_dir='/home/du/Proj/NeuRIS/Data/dataset/indoor/scene0580_00/image'
semantic_gt_dir='/home/du/Proj/NeuRIS/Data/dataset/indoor/scene0580_00/semantic_deeplab'
semantic_dir='/home/du/Proj/NeuRIS/exps/indoor/neus/scene0580_00/test3_semantic/semantic_render'
img_number='0110'
img_name=img_number+'.png'
semantic_gt_name=str(int(img_number))+'.png'
semantic_name='00160000_'+img_number+'_reso2.png'
img_dir=os.path.join(img_dir,img_name)
semantic_gt_file=os.path.join(semantic_gt_dir,semantic_gt_name)
semantic_file=os.path.join(semantic_dir,semantic_name)

img=cv2.imread(img_dir)
semantic_gt=cv2.imread(semantic_gt_file)
semantic=cv2.imread(semantic_file)

print(type(img))
print('img shape:',img.shape)
print('semantic_gt shape:', semantic_gt.shape)
print('semantic shape:', semantic.shape)

resolution=(img.shape)[0]/(semantic.shape)[0]
print('resolution:',resolution)
img_ds = cv2.pyrDown(img)
semantic_gt_ds = cv2.pyrDown(semantic_gt)
print('down sampling:', img_ds.shape)

img_ds_name=img_number+'_res2.png'
img_ds_dir=os.path.join(img_dir,img_ds_name)
cv2.imwrite(img_ds_dir,img_ds)

alpha=0.7

refuse=np.zeros_like(semantic)
for i in range((img_ds.shape)[0]):
    for j in range((img_ds.shape)[1]):
        if semantic[i][j][0]==0:
            refuse[i][j]=img_ds[i,j]
        if semantic[i][j][0]==80:
            refuse[i][j]=(1-alpha)*np.float128(img_ds[i,j])+alpha*np.float128([232, 199, 174])
        if semantic[i][j][0]==160:
            refuse[i][j]=(1-alpha)*np.float128(img_ds[i,j])+alpha*np.float128([152, 223, 138])
semantic_refuse_name='00160000_'+img_number+'_reso2_refuse.png'
semantic_refuse_file=os.path.join(semantic_dir,semantic_refuse_name)
print(semantic_refuse_file)
cv2.imwrite(semantic_refuse_file, refuse)

refuse_gt=np.zeros_like(semantic)
for i in range((img_ds.shape)[0]):
    for j in range((img_ds.shape)[1]):
        if semantic_gt_ds[i][j][0]==0:
            refuse[i][j]=img_ds[i,j]
        if semantic_gt_ds[i][j][0]==80:
            refuse[i][j]=(1-alpha)*np.float128(img_ds[i,j])+alpha*np.float128([232, 199, 174])
        if semantic_gt_ds[i][j][0]==160:
            refuse[i][j]=(1-alpha)*np.float128(img_ds[i,j])+alpha*np.float128([152, 223, 138])
semantic_gt_ds_refuse_name='00160000_'+img_number+'_reso2_refuse_GT.png'
semantic_gt_ds_refuse_file=os.path.join(semantic_dir,semantic_gt_ds_refuse_name)
print(semantic_gt_ds_refuse_file)
cv2.imwrite(semantic_gt_ds_refuse_file, refuse)

print('complete')