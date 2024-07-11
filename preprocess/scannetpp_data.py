import os, shutil, logging
import cv2
import json
import torch, plyfile
import numpy as np
import open3d as o3d

from glob import glob
from tqdm import tqdm

import preprocess.utils.colmap as ColmapUtils
import utils.utils_io as IOUtils
import utils.utils_nyu as NyuUtils
import utils.utils_geometry as GeometryUtils

from utils.utils_image import read_image

class ScannetppData:
    def __init__(self, dir_scan, dir_neus, scene_name, 
                camera_device='dslr'):
    
        self.dir_scan_data = f'{dir_scan}/data/{scene_name}/{camera_device}'
        self.mesh_path = f'{dir_scan}/data/{scene_name}/scans'
        self.dir_scan_pth = f'{dir_scan}/semantics-nyu40/pth/{scene_name}.pth'
        self.dir_neus = dir_neus

        # distorted intrinsics, colmap pose, point cloud
        cameras, images, points3D = ColmapUtils.read_model(f'{self.dir_scan_data}/colmap', ".txt")
        # write points3D to ply
        xyz_list, rgb_list = [], []
        for point_id, point_data in points3D.items():
            xyz_list.append(point_data.xyz)
            rgb_list.append(point_data.rgb)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(xyz_list))
        pcd.colors = o3d.utility.Vector3dVector(np.array(rgb_list) / 255.0)  # Open3D 要求颜色值在 [0, 1] 范围内 
        o3d.io.write_point_cloud(f'{dir_neus}/point_cloud_colmap.ply', pcd)

        # Laser scan
        pc_path = os.path.join(dir_scan + f'data/{scene_name}/scans/pc_aligned.ply')

        # Save camera pose
        poses_w2c_dict = {image.name: image.world_to_camera for image_id, image in images.items()}

        train_test_split = json.load(open(f'{dir_neus}/train_test_lists.json'))
        train_split, test_split = sorted(train_test_split['train']), sorted(train_test_split['test'])
        self.poses_w2c_train = np.array([poses_w2c_dict[img_name] for img_name in train_split])
        self.poses_w2c_test =  np.array([poses_w2c_dict[img_name] for img_name in test_split])
        
        self.poses_c2w_train = GeometryUtils.get_poses_inverse(self.poses_w2c_train)
        self.poses_c2w_test = GeometryUtils.get_poses_inverse(self.poses_w2c_test)
        # copy pose files
        path_target_train, path_target_test = f"{dir_neus}/pose/train/", f"{dir_neus}/pose/test/"
        IOUtils.ensure_dir_existence(path_target_train)
        IOUtils.ensure_dir_existence(path_target_test)
        
        for idx, pose_c2w in enumerate(self.poses_c2w_train):
            np.savetxt(path_target_train+f"{idx:04d}.txt", pose_c2w)
        for idx, pose_c2w in enumerate(self.poses_c2w_test):
            np.savetxt(path_target_test+f"{idx:04d}.txt", pose_c2w)
        
    @staticmethod
    def select_data(dir_neus, dir_scan_data, dir_scan_pth, train_test_split,
                    cropped_size = (1536, 1152),
                    resize_size = (640, 480) ):
        # Crop and resize images, depths, semantics
        img_lis = os.listdir(f'{dir_scan_data}/undistorted_images/')
        (H, W, _) = read_image(f'{dir_scan_data}/undistorted_images/'+img_lis[0]).shape
        (W_cropped, H_cropped) = cropped_size
        crop_width_half = (W-W_cropped)//2
        crop_height_half = (H-H_cropped) //2
        assert (W-W_cropped)%2 ==0 and (H- H_cropped) %2 == 0
        
        # Slecting images, depths, semantics
        train_split, test_split = sorted(train_test_split['train']), sorted(train_test_split['test'])
        for img_name in tqdm(img_lis, desc='Selecting data'):
            # resize
            if img_name in train_split:
                mode='train'
                img_idx = train_split.index(img_name)
            elif img_name in test_split:
                mode='test'
                img_idx = test_split.index(img_name)
            else:
                logging.warning(f"Image {img_name} not in train or test split, skipped.")
                continue
            
            # RGB
            path_src = f'{dir_scan_data}/undistorted_images/' + img_name
            img = cv2.imread(path_src)
            img_crop = img[crop_height_half: H-crop_height_half, crop_width_half: W-crop_width_half, :]
            assert img_crop.shape[0] == cropped_size[1]
            img_resize = cv2.resize(img_crop, resize_size, interpolation=cv2.INTER_NEAREST)

            path_target = f"{dir_neus}/image/{mode}/"
            IOUtils.ensure_dir_existence(path_target)
            cv2.imwrite(path_target+f"{(img_idx):04d}.png", img_resize)
            
            # Depth
            path_src = f'{dir_scan_data}/undistorted_depths/' + img_name.split('.')[0] + '.png'
            depth = cv2.imread(path_src, cv2.IMREAD_UNCHANGED)
            depth_crop = depth[crop_height_half: H-crop_height_half, crop_width_half: W-crop_width_half]
            depth_resize = cv2.resize(depth_crop, resize_size, interpolation=cv2.INTER_NEAREST)

            path_target = f"{dir_neus}/depth/{mode}/"
            IOUtils.ensure_dir_existence(path_target)
            cv2.imwrite(path_target+f"{(img_idx):04d}.png", depth_resize)
            
            # Depth-vis map
            depth_vis = cv2.convertScaleAbs(depth_resize*50.0/1000)
            depth_vis_jet = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            path_target = f"{dir_neus}/depth_vis/{mode}/"
            IOUtils.ensure_dir_existence(path_target)
            cv2.imwrite(path_target+f"{(img_idx):04d}.png", depth_vis_jet)

            # Semantic
            path_src = f'{dir_scan_data}/undistorted_semantics-nyu40/' + img_name.split('.')[0] + '.png'
            semantic = cv2.imread(path_src, cv2.IMREAD_UNCHANGED)
            semantic_crop = semantic[crop_height_half: H-crop_height_half, crop_width_half: W-crop_width_half]
            semantic_resize = cv2.resize(semantic_crop, resize_size, interpolation=cv2.INTER_NEAREST)
            ## preprocessed semantic labels: void label
            semantic_relabel=semantic_resize.astype(float)
            semantic_relabel[semantic_resize==65535] = -1
            semantic_relabel = (semantic_relabel+1).astype(np.uint8)

            path_target = f"{dir_neus}/semantic/{mode}/semantic_GT/"
            IOUtils.ensure_dir_existence(path_target)
            cv2.imwrite(path_target+f"{(img_idx):04d}.png", semantic_relabel)
            
            # Semantic-vis map
            colour_map_np = NyuUtils.nyu40_colour_code
            semantic_vis = colour_map_np[semantic_relabel]

            path_target = f"{dir_neus}/semantic/{mode}/semantic_GT_vis/"
            IOUtils.ensure_dir_existence(path_target)
            cv2.imwrite(path_target+f"{(img_idx):04d}.png", semantic_vis[...,::-1])
        
        # GT mesh
        mesh_path = os.path.dirname(dir_scan_data) + '/scans/'
        plydata = plyfile.PlyData.read(f'{mesh_path}/mesh_aligned_0.05.ply')
        shutil.copyfile(f'{mesh_path}/mesh_aligned_0.05.ply', f'{dir_neus}/mesh.ply')
        
        # sem mesh
        vtx_lables = torch.load(dir_scan_pth)['vtx_labels']
        vtx_lables_relabel = vtx_lables.astype(float) 
        vtx_lables_relabel[vtx_lables==-100] = -1
        
        vtx_labels = (vtx_lables_relabel+1).astype(np.uint8)
        vtx_color = colour_map_np[vtx_labels]
        vtx_x, vtx_y, vtx_z = plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']
        
        vertex_array = np.column_stack((vtx_x, vtx_y, vtx_z, \
                                    vtx_color[:,0], vtx_color[:,1], vtx_color[:,2], \
                                    vtx_labels))
        vertex = np.array([tuple(row) for row in vertex_array], \
                                dtype=[('x', 'float32'), ('y', 'float32'), ('z', 'float32'), \
                                       ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8'), \
                                        ('label', 'uint8')])                                                        
        
        vertex_element = plyfile.PlyElement.describe(vertex, 'vertex')

        plydata_sem = plyfile.PlyData([vertex_element, plydata['face']])
        plydata_sem.write(f'{dir_neus}/mesh_sem.ply')

    def calculate_normals(self, dir_normal, intrinsics_depth):
        for mode in ['train', 'test']:
            dir_normal_world_mode = f"{self.dir_neus}/normal/{mode}/{dir_normal}_world"
            dir_normal_cam_mode = f"{self.dir_neus}/normal/{mode}/{dir_normal}_cam"
            IOUtils.ensure_dir_existence(dir_normal_world_mode)
            IOUtils.ensure_dir_existence(dir_normal_cam_mode)

            poses_w2c = self.poses_w2c_train if mode == 'train' else self.poses_w2c_test
            file_depth_mode = glob(f"{self.dir_neus}/depth/{mode}/*.png")
            for idx in tqdm(range(len(file_depth_mode)), desc='Calculating normal'):
                file_name = os.path.basename(file_depth_mode[idx]).split('.')[0]
                
                # compute normal from depth maps
                depth = cv2.imread(file_depth_mode[idx], cv2.IMREAD_UNCHANGED) / 1000.0
                
                valid_mask = depth>0
                pts_i, normal_map_world_i = GeometryUtils.calculate_normalmap_from_depthmap(depth, intrinsics_depth, poses_w2c[idx])
                normal_wolrd_map_vis = ((normal_map_world_i+1) / 2).clip(0, 1)*255
                normal_wolrd_map_vis[~valid_mask] = np.array([0,0,0]).astype(np.uint8)
                # save normal_w
                np.savez(f"{dir_normal_world_mode}/{file_name}.npz", normal=normal_map_world_i)
                np.savez(f"{dir_normal_world_mode}/{file_name}_mask.npz", mask=valid_mask)
                cv2.imwrite(f"{dir_normal_world_mode}/{file_name}.png", normal_wolrd_map_vis[...,::-1])

                # get normal_cam
                normal_map_cam_i = GeometryUtils.get_world_normal(-normal_map_world_i.reshape(-1, 3), 
                                                                  GeometryUtils.get_poses_inverse(poses_w2c[idx]))
                normal_map_cam_i = normal_map_cam_i.reshape(normal_map_world_i.shape)
                normal_cam_map_vis = ((normal_map_cam_i+1) / 2).clip(0, 1)*255
                normal_cam_map_vis[~valid_mask] = np.array([0,0,0]).astype(np.uint8)
                
                # save normal_c
                np.savez(f"{dir_normal_cam_mode}/{file_name}.npz", normal=normal_map_cam_i)
                np.savez(f"{dir_normal_cam_mode}/{file_name}_mask.npz", mask=valid_mask)
                cv2.imwrite(f"{dir_normal_cam_mode}/{file_name}.png", normal_cam_map_vis[...,::-1])

    # transform camera poses and save
    def get_projection_matrix(self, intrin, poses, trans_n2w, mode):
        '''
        Args:
            poses: world to camera
        '''
        num_poses = poses.shape[0]
        
        projs = []
        poses_norm = []
        dir_pose_norm = self.dir_neus + f"/extrin_norm/{mode}"
        IOUtils.ensure_dir_existence(dir_pose_norm)
        for i in range(num_poses):
            # pose_norm_i = poses[i] @ trans_n2w

            # Method 2
            pose = poses[i] #w2c
            rot = pose[:3,:3]
            trans = pose[:3,3]

            cam_origin_world = - np.linalg.inv(rot) @ trans.reshape(3,1) #c2w
            cam_origin_world_homo = np.concatenate([cam_origin_world,[[1]]], axis=0)
            cam_origin_norm = np.linalg.inv(trans_n2w) @ cam_origin_world_homo #c2n
            trans_norm = -rot @ cam_origin_norm[:3] #n2c

            pose[:3,3] = np.squeeze(trans_norm) #n2c
            poses_norm.append(pose)
            proj_norm = intrin @ pose
            projs.append(proj_norm) #w2i
            
            np.savetxt(f'{dir_pose_norm}/{i:04d}.txt', pose, fmt='%f') # world to camera
            np.savetxt(f'{dir_pose_norm}/{i:04d}_inv.txt', GeometryUtils.get_pose_inv(pose) , fmt='%f') # inv: camera to world
        return np.array(projs), np.array(poses_norm)

    # generate cameras_sphere.npz and pcd_norm.ply
    def transform_scenes(self, intrinsics, poses_w2c, trans_n2w, point_cloud, mode='train'):
        projs, poses_norm = self.get_projection_matrix(intrinsics, poses_w2c, trans_n2w, mode)
    
        point_cloud_trans = point_cloud.transform(np.linalg.inv(trans_n2w))
        o3d.io.write_point_cloud(f'{self.dir_neus}/point_cloud_scan_norm.ply', point_cloud_trans)

        pts_cam_norm = GeometryUtils.get_camera_origins(poses_norm)
        GeometryUtils.save_points(f'{self.dir_neus}/cam_norm.ply', pts_cam_norm)
        
        pts_cam = (trans_n2w[None, :3,:3] @ pts_cam_norm[:, :, None]).squeeze()  + trans_n2w[None, :3, 3]
        GeometryUtils.save_points(f'{self.dir_neus}/cam_origin.ply', pts_cam)

        scale_mat = np.identity(4)
        num_cams = projs.shape[0]
        cams_neus = {}
        for i in range(num_cams):
            cams_neus[f"scale_mat_{i}"] = scale_mat
            cams_neus[f'world_mat_{i}'] = projs[i] #n2i
        
        np.savez(f'{self.dir_neus}/cameras_sphere_{mode}.npz', **cams_neus)
        
        # transform gt mesh
        path_gt_mesh = f'{self.dir_neus}/mesh.ply'
        if path_gt_mesh is None:
            return
        
        path_save = IOUtils.add_file_name_suffix(path_gt_mesh, "_trans")
        trans = np.linalg.inv(trans_n2w)
        GeometryUtils.transform_mesh(path_gt_mesh, trans, path_save)

    def generate_neus_data(self, intrinsics, 
                           radius_normalize_sphere=1.0,
                           compute_normal=False):
        self.intrinsics = intrinsics
        # Colmap: {dir_neus}/point_cloud_colmap.ply
        # Laser Scan: mesh_path/pc_aligned.ply
        self.path_cloud = f'{self.mesh_path}/pc_aligned.ply'

        if compute_normal:
            self.calculate_normals('normal_from_depth', intrinsics)

        shutil.copyfile(self.path_cloud, f'{self.dir_neus}/point_cloud_scan.ply')
        point_cloud = GeometryUtils.read_point_cloud(self.path_cloud)
        
        # transforms point cloud to normalized space
        path_trans_n2w = f'{self.dir_neus}/trans_n2w.txt'
        if not IOUtils.checkExistence(path_trans_n2w):
            logging.info(f"Generating transformation matrix...")
            trans_n2w = GeometryUtils.get_norm_matrix_from_point_cloud(point_cloud, radius_normalize_sphere=radius_normalize_sphere)
            np.savetxt(path_trans_n2w, trans_n2w, fmt = '%.04f')
        else:
            logging.info(f"Load transformation matrix: {path_trans_n2w}")
            trans_n2w = np.loadtxt(path_trans_n2w, dtype=np.float32)
        logging.info(f"Transformation matrix: {trans_n2w}")

        # transform camera poses and save
        self.transform_scenes(intrinsics, self.poses_w2c_train, trans_n2w, point_cloud, mode='train')
        self.transform_scenes(intrinsics, self.poses_w2c_test, trans_n2w, point_cloud, mode='test')


