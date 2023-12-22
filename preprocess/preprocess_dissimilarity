from glob import glob
from tqdm import tqdm
from lu_vp_detect import VPDetection
import numpy as np
import matplotlib as mpl
import os
import cv2

img_dir = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/dataset/indoor'
scene_lis = ['test']

def cosine_similarity(x1, x2, dim=2, eps=1e-6):
   dot_product = np.sum(x1 * x2, axis=dim)
   x1_norm = np.sqrt(np.sum(x1 ** 2, axis=dim))+eps
   x2_norm = np.sqrt(np.sum(x2 ** 2, axis=dim))+eps
   cosine_sim = dot_product / (x1_norm * x2_norm)
   return cosine_sim

for scene_name in tqdm(scene_lis, desc='processing scene...'):
    for data_mode in ['train']:
        intrinsics = np.loadtxt(os.path.join(img_dir, scene_name, 'intrinsic_color_crop1248_resize640.txt'))
        fx, fy = intrinsics[0,0], intrinsics[1,1]
        cx, cy = intrinsics[0,2], intrinsics[1,2]
        principal_point = cx, cy
        focal_length = fx
        seed = 2023
        vpd = VPDetection(60, principal_point, focal_length, seed)
        
        img_file = sorted(glob(os.path.join(img_dir, scene_name, 'image', data_mode, '*.png')))
        normal_file = sorted(glob(os.path.join(img_dir, scene_name, 'normal_w', data_mode, 'pred_normal','*.npz')))
        depth_file = sorted(glob(os.path.join(img_dir, scene_name, 'depth', data_mode, '*.png')))
        pose_file = sorted(glob(os.path.join(img_dir, scene_name, 'pose', data_mode, '*.txt')))

        align_normal_dir = os.path.join(img_dir, scene_name, 'align_normal', data_mode)
        os.makedirs(align_normal_dir, exist_ok=True)
        dissimilalrity_dir = os.path.join(img_dir, scene_name, 'dissimilarirty', data_mode)
        os.makedirs(dissimilalrity_dir, exist_ok=True)
        for idx in tqdm(range(len(img_file)), desc='computing dissimilarity...'):
            img = cv2.imread(img_file[idx])
            w, h, _ = img.shape
            normal = np.load(normal_file[idx])['arr_0']
            
            ########################
            # aligned normal
            vps = vpd.find_vps(img) 
            vps = np.vstack([vps, -vps]).astype(float) # 6x3
            norm_flatten = normal.reshape(-1,3) # Nx3
            # compute_mmap
            repeats = (norm_flatten.shape[0],1,1)
            vps_6 = np.transpose(np.tile(vps, repeats), (1,0,2)) # 6xNx3
            repeats = (6,1,1)
            norm_flatten_6 = np.tile(norm_flatten, repeats) # 6xNx3
            cos_sim = cosine_similarity(vps_6, norm_flatten_6)
            index = np.argmax(cos_sim, axis=0)  # N
            align_normal = vps[index,:].reshape(w, h, 3)  # wxhx3
            # vis
            align_normal_vis=(((align_normal + 1) * 0.5).clip(0,1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(align_normal_dir, os.path.basename(img_file[idx])), align_normal_vis)
            #########################

            #########################
            # acquiring camera coordinates
            pose = np.loadtxt(pose_file[idx])
            depth = cv2.imread(depth_file[idx], cv2.IMREAD_UNCHANGED)/1000
            x, y = np.linspace(0, w-1, w), np.linspace(0, h-1, h)
            x_mesh, y_mesh = np.meshgrid(x, y)
            depth = depth.flatten()
            u, v = x_mesh.flatten(), y_mesh.flatten()
            # reprojecting points 
            I = np.zeros((4, h*w))
            I[0, :] = u * depth
            I[1, :] = v * depth
            I[2, :] = depth
            I[3, :] = 1
            world_coordinates = pose @ (np.linalg.inv(intrinsics) @ I) # 4xN
            # computing depth
            points = np.transpose(np.expand_dims(world_coordinates[:3, :], axis=-1), (1, 0, 2)) # Nx3x1
            norm_flatten = np.expand_dims(norm_flatten, axis=1) # Nx1x3    
            D = (- norm_flatten @ points) # wxh
            # D[depth<1e-3] = 0
            # D[D<0] = -D[D<0] 
            D = D.reshape(w,h)      
            #########################

            # img
            img=img.astype(float)
            rgb_down = np.sqrt(np.sum((img[1:,:,:] - img[:-1,:,:]) ** 2, axis=-1))
            rgb_right= np.sqrt(np.sum((img[:,1:,:] - img[:,:-1,:]) ** 2, axis=-1))
            dst_rgb = rgb_down[:,:-1] + rgb_right[:-1,:]
            dst_rgb = (dst_rgb - np.min(dst_rgb))/(np.max(dst_rgb) - np.min(dst_rgb)+1e-6) # normalize
            # normal
            normal = align_normal
            normal_down = np.sqrt(np.sum((normal[1:,:,:] - normal[:-1,:,:])**2, axis=-1))
            normal_right= np.sqrt(np.sum((normal[:,1:,:] - normal[:,:-1,:])**2, axis=-1))
            dst_normal = normal_down[:,:-1] + normal_right[:-1,:]
            dst_normal = (dst_normal - np.min(dst_normal))/(np.max(dst_normal) - np.min(dst_normal)+1e-6) # normalize
            # D
            D_down = abs(D[1:, :] - D[:-1, :])
            D_right= abs(D[:, 1:] - D[:, :-1])
            dst_D= D_down[:, :-1] + D_right[:-1, :]
            dst_D = (dst_D - np.min(dst_D))/(np.max(dst_D) - np.min(dst_D)+1e-6) # normalize
            # normalD
            normalD_down = normal_down+D_down
            normalD_right = normal_right+D_right
            dst_normalD = normalD_down[:,:-1] + normalD_right[:-1,:]
            dst_normalD = (dst_normalD - np.min(dst_normalD))/(np.max(dst_normalD) - np.min(dst_normalD)+1e-6) # normalize

            dst = 0.4*dst_rgb+0.6*dst_normalD

            dissimilalrity_vis = mpl.colormaps['viridis'](dst)*255
            dissimilalrity_vis = dissimilalrity_vis.astype(np.uint8)[:,:,:-1][...,::-1]
            cv2.imwrite(os.path.join(dissimilalrity_dir, os.path.basename(img_file[idx])), dissimilalrity_vis)




            
        
            
            





            
        
        