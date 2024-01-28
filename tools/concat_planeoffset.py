import os, sys
sys.path.append(os.getcwd())
import cv2
import matplotlib
import numpy as np
from glob import glob
from tqdm import tqdm
import utils.utils_image as ImageUtils

img_dir = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/dataset/indoor'
exps_dir = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/exps/indoor/neus'

from confs.path import lis_name_scenes
method_name_lis = ['deeplab_ce_plane/ce_stop_planedepth10', 'deeplab_ce_plane/ce_stop_planedepth30', 'deeplab_ce_plane/ce_stop_planedepth100']
iter_num = '00160000'

def compute_planeoffset(h, w, 
                          intrinsics, 
                          pose, 
                          depth,
                          normal):
    # acquiring camera coordinates
    x, y = np.linspace(0, w-1, w), np.linspace(0, h-1, h)
    x_mesh, y_mesh = np.meshgrid(x, y)
    u, v = x_mesh.flatten(), y_mesh.flatten()
    
    # reprojecting points 
    I = np.zeros((4, h*w))
    I[0, :] = u * depth.flatten()
    I[1, :] = v * depth.flatten()
    I[2, :] = depth.flatten()
    I[3, :] = 1

    world_coordinates = pose @ (np.linalg.inv(intrinsics) @ I) # 4xN
    
    # computing depth
    points = np.transpose(np.expand_dims(world_coordinates[:3, :], axis=-1), (1, 0, 2)) # Nx3x1
    normal = np.expand_dims(normal.reshape(-1, 3), axis=1) # Nx1x3    
    planeoffset = (- normal @ points) # wxh
    return planeoffset.reshape(w, h)

for method_name in method_name_lis:
    for scene_name in tqdm(lis_name_scenes, desc='processing scene...'):
        for data_mode in ['train']:
            img_file = sorted(glob(os.path.join(img_dir, scene_name, 'image', data_mode, '*.png')))
            pose_file = sorted(glob(os.path.join(img_dir, scene_name, 'pose', data_mode, '*.txt')))
            intrinsics = np.loadtxt(os.path.join(img_dir, scene_name, 'intrinsic_color_crop1248_resize640.txt'))
            # pred
            normal_pred_file = sorted(glob(os.path.join(exps_dir, method_name, scene_name, 'normal', data_mode, 'fine',f'{iter_num}_*.npz')))
            depth_pred_file = sorted(glob(os.path.join(exps_dir, method_name, scene_name, 'depth', data_mode, 'fine',f'{iter_num}_*.npz')))
            planeoffset_pred_dir = os.path.join(exps_dir, method_name, scene_name, 'planeoffset', data_mode, 'fine')
            os.makedirs(planeoffset_pred_dir, exist_ok=True)
            # GT
            normal_GT_file = sorted(glob(os.path.join(img_dir, scene_name, 'normal', data_mode, 'pred_normal_world','*.npz')))
            depth_GT_file = sorted(glob(os.path.join(img_dir, scene_name, 'depth', data_mode, '*.png')))
            planeoffset_GT_dir = os.path.join(img_dir, scene_name, 'planeoffset', data_mode)
            os.makedirs(planeoffset_GT_dir, exist_ok=True)
            
            ### computing planeoffset
            normal_pred = np.load(normal_pred_file[0])['arr_0']
            w, h, _ = normal_pred.shape
            for idx in tqdm(range(len(img_file)), desc='Computing Planeoffset...'):
                pose = np.loadtxt(pose_file[idx])
                img = cv2.imread(img_file[idx])
                ### pred
                normal_pred = np.load(normal_pred_file[idx])['arr_0']
                depth_pred = np.load(depth_pred_file[idx], cv2.IMREAD_UNCHANGED)['arr_0']                
                planeoffset_pred = compute_planeoffset(h, w, 
                                                    intrinsics, 
                                                    pose, 
                                                    depth_pred,
                                                    normal_pred)
                
                depth_pred_map = cv2.convertScaleAbs(depth_pred*50)
                depth_pred_map_jet = cv2.applyColorMap(depth_pred_map, cv2.COLORMAP_JET)
                planeoffset_pred_map = cv2.convertScaleAbs(planeoffset_pred*25)
                planeoffset_pred_map_jet = cv2.applyColorMap(planeoffset_pred_map, cv2.COLORMAP_JET)

                ### GT
                normal_GT = np.load(normal_GT_file[idx])['arr_0']
                depth_GT = cv2.imread(depth_GT_file[idx], cv2.IMREAD_UNCHANGED)/1000
                normal_GT = cv2.resize(normal_GT, (h ,w), interpolation=cv2.INTER_NEAREST)
                depth_GT = cv2.resize(depth_GT, (h ,w), interpolation=cv2.INTER_NEAREST)
                planeoffset_GT = compute_planeoffset(h, w, 
                                                    intrinsics, 
                                                    pose, 
                                                    depth_GT,
                                                    normal_GT )
                
                depth_GT_map = cv2.convertScaleAbs(depth_GT*50)
                depth_GT_map_jet = cv2.applyColorMap(depth_GT_map, cv2.COLORMAP_JET)
                planeoffset_GT_map = cv2.convertScaleAbs(planeoffset_GT*25)
                planeoffset_GT_map_jet = cv2.applyColorMap(planeoffset_GT_map, cv2.COLORMAP_JET)
                planeoffset_GT_map_jet[depth_GT<0.01, :]=[0,0,0]
                cv2.imwrite(os.path.join(planeoffset_GT_dir, os.path.basename(img_file[idx])), planeoffset_GT_map_jet)

                # planeoffset diff
                depth_diff = np.abs(depth_GT - depth_pred)
                planeoffset_diff = np.abs(planeoffset_GT - planeoffset_pred)
                colormap_func = matplotlib.cm.get_cmap("jet")
                depth_diff_vis = colormap_func(depth_diff)[:, :, :3]
                planeoffset_diff_vis = colormap_func(planeoffset_diff)[:, :, :3]

                depth_diff_vis[depth_GT<0.01, :]=[0,0,0]
                planeoffset_diff_vis[depth_GT<0.01, :]=[0,0,0]

                lis_img = [cv2.resize(img, (h, w), interpolation=cv2.INTER_NEAREST), \
                           depth_GT_map_jet, depth_pred_map_jet, depth_diff_vis[..., ::-1]*255, \
                           planeoffset_GT_map_jet, planeoffset_pred_map_jet, planeoffset_diff_vis[..., ::-1]*255 ]
                
                lis_img = np.stack(lis_img, axis=0)
                ImageUtils.write_image_lis(os.path.join(planeoffset_pred_dir, 'concat_'+os.path.basename(img_file[idx])), 
                                           lis_img, 
                                           cat_mode='horizontal')


                
