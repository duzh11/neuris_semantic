import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

Data_dir = '../Data/dataset/indoor'
scene_name = 'scene0616_00'
eva=0.5

# read data
semantic_dir = os.path.join(Data_dir, scene_name, 'semantic_pred')
semantic_vis_dir = os.path.join(Data_dir, scene_name, 'semantic_pred_vis')
img_dir = os.path.join(Data_dir, scene_name, 'image')
depth_dir = os.path.join(Data_dir, scene_name, 'depth')
pose_dir = os.path.join(Data_dir, scene_name, 'pose')
intrinsic_dir = os.path.join(Data_dir, scene_name, 'intrinsic_color_crop1248_resize640.txt')

semantic_lis = glob(f'{semantic_dir}/*.png')
semantic_lis.sort(key=lambda x:int((x.split('/')[-1]).split('.')[0]))
semantic_vis_lis = glob(f'{semantic_vis_dir}/*.png')
semantic_vis_lis.sort(key=lambda x:int((x.split('/')[-1]).split('.')[0]))
img_lis = sorted(glob(f'{img_dir}/*.png'))
depth_lis = sorted(glob(f'{depth_dir}/*.png'))
pose_lis = sorted(glob(f'{pose_dir}/*.txt'))

semantic_np = np.stack([cv2.imread(seg_name)[:,:,0] for seg_name in semantic_lis])
semantic_vis_np = np.stack([cv2.imread(seg_vis_name) for seg_vis_name in semantic_vis_lis])
img_np = np.stack([cv2.imread(img_name) for img_name in img_lis])
depth_np = np.stack([cv2.imread(depth_name, cv2.IMREAD_UNCHANGED)/1000 for depth_name in depth_lis])
pose_np = np.stack([np.loadtxt(pose_name) for pose_name in pose_lis])
intrinsic_np = np.loadtxt(intrinsic_dir)

def compute_projection(H, W, intrinsic, depth, pose_source, pose_target):
        # acquiring camera coordinates
        x = np.linspace(0, W-1, W)
        y = np.linspace(0, H-1, H)
        x_mesh, y_mesh = np.meshgrid(x, y)
        u, v =torch.from_numpy(x_mesh).type(torch.cuda.FloatTensor).flatten(), torch.from_numpy(y_mesh).type(torch.cuda.FloatTensor).flatten()
        # reproject (u,v) to word
        depth = torch.from_numpy(depth).type(torch.cuda.FloatTensor)
        I = torch.zeros(4, H*W).type(torch.cuda.FloatTensor)
        I[0, :] = u * depth
        I[1, :] = v * depth
        I[2, :] = depth
        I[3, :] = 1
        world_coordinates = torch.mm(torch.from_numpy(pose_source).type(torch.cuda.FloatTensor), torch.mm(
                torch.from_numpy(np.linalg.inv(intrinsic)).type(torch.cuda.FloatTensor), I))
        target_points = torch.mm(torch.from_numpy(intrinsic).type(torch.cuda.FloatTensor),
                torch.mm( torch.from_numpy(np.linalg.inv(pose_target)).type(torch.cuda.FloatTensor), world_coordinates))
        # normalized
        target_points = target_points.transpose(0, 1)[:, :3]
        target_points[:, 0] /= target_points[:, 2]
        target_points[:, 1] /= target_points[:, 2]
        target_points = torch.round(target_points[:, :2])

        return target_points

save_dir = os.path.join(Data_dir, scene_name, 'semantic_consistency')
os.makedirs(save_dir, exist_ok=True)

N_img, H, W=semantic_np.shape
for source_idx in tqdm(range(N_img), desc='computing projection'):
        semantic_source = torch.from_numpy(semantic_np[source_idx, :].flatten()).cuda()
        consistency = torch.zeros_like(semantic_source)
        count = torch.zeros_like(semantic_source)

        # 将source_img投影到target_img
        pose_source = pose_np[source_idx]
        depth = depth_np[source_idx].flatten()
        for target_idx in range(N_img):
                pose_target = pose_np[target_idx]
                semantic_target = torch.from_numpy(semantic_np[target_idx, :].flatten()).cuda()
                projected_points = compute_projection(H, W, intrinsic_np, depth, pose_source, pose_target)
                
                # validating
                if target_idx==2000:
                        plt.close('all')
                        img_source, img_target = img_np[source_idx], img_np[target_idx]
                        x_source, y_source =400, 120
                        (x_target, y_target) = projected_points[y_source*W+x_source-1,:]
                        print(f'correspondence: ({x_target}, {y_target})')
                        fig, ax = plt.subplots(1, 2)
                        ax[0].imshow(img_source[...,::-1])
                        ax[0].plot(x_source, y_source, 'ro', markersize=5)

                        ax[1].imshow(img_target[...,::-1])
                        ax[1].plot(x_target.cpu().numpy(), y_target.cpu().numpy(), 'ro', markersize=5)

                        plt.tight_layout()
                        plt.show()
                        print('validate')

                # filtering
                valid_mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < W) & (
                        projected_points[:, 1] >= 0) & (projected_points[:, 1] < H)
                
                valid_mask = valid_mask.flatten()
                projected_points_flat = (projected_points[:, 1] * W + projected_points[:, 0])
                
                projected_points_filter = projected_points_flat[valid_mask]
                semantic_target_filter = semantic_target[valid_mask]
                semantic_source_filter = semantic_source[valid_mask]
                
                # compute semantic consistency
                semantic_projection = semantic_target[projected_points_filter.long()]
                similarity = (semantic_source_filter==semantic_projection)
                consistency[valid_mask] += similarity
                count[valid_mask] += valid_mask[valid_mask]
        
        # 认为出现次数出现一定次数以上为正确语义
        global_similarity = consistency/count
        consistency_mask = global_similarity>eva
        semantic_similarity = global_similarity.reshape(H,W)
        np.savez(os.path.join(save_dir, f'{source_idx}.npz'), semantic_similarity.cpu().numpy())
        
        # 可视化
        visualize=True
        if visualize:
                save_vis_dir = os.path.join(Data_dir, scene_name, f'semantic_consistency_{eva}_vis')
                os.makedirs(save_vis_dir, exist_ok=True)
                semantic_vis = semantic_vis_np[source_idx]
                consistency_mask_np = consistency_mask.cpu().numpy().reshape(H, W)
                semantic_vis[~consistency_mask_np]=np.array([0,0,0])
                cv2.imwrite(os.path.join(save_vis_dir, f'{source_idx}.png'), semantic_vis)

                

        
        








                      
                      
        


    

    
    
    





    

