import os
import sys
from turtle import width
from glob import glob
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import json

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def glob_data(data_dir):
    data_paths = []
    data_paths.extend(glob(data_dir))
    data_paths = sorted(data_paths)
    return data_paths

def render_scan(scene_name):
    data_dir=f'../Data/dataset/indoor'

    instance_dir = os.path.join(data_dir, scene_name)
    image_paths = glob_data(os.path.join('{0}'.format(instance_dir), 'image', "*.png"))
    n_images = len(image_paths)
    
    cam_file = '{0}/cameras_sphere.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    scale_mats_0 = np.load(cam_file)['scale_mat_0']

    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)

        intrinsics_all.append(intrinsics)
        pose_all.append(pose)
        
    H, W = 640,480
    # create tmp camera pose file for open3d
    camera_config = Path("../video_poses")
    camera_config.mkdir(exist_ok=True, parents=True)

    for image_id in range(n_images):
        
        c2w = pose_all[image_id]
        w2c = np.linalg.inv(c2w)

        K = intrinsics_all[0].copy()
        # K[:2, :] *= 2.
        K[0,2]=480/2-0.5
        K[1,2]=640/2-0.5
        tmp_json = json.load(open('../video_poses/c1.json'))
        tmp_json["extrinsic"] = w2c.T.reshape(-1).tolist()
        
        tmp_json["intrinsic"]["intrinsic_matrix"] = K[:3,:3].T.reshape(-1).tolist()
        tmp_json["intrinsic"]["height"] = H 
        tmp_json["intrinsic"]["width"] = W 
        json.dump(tmp_json, open('../video_poses/tmp%d.json'%(image_id), 'w'), indent=4)
    
    return f'{camera_config}'
    

def render(ply_file,out_path,camera_config):
    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        
        global index

        # if index >= 0:
        #     vis.capture_screen_image(os.path.join(out_path, "render_{}.jpg".format(index)), False)
        index = index + 1
        if index < 20:
            param = o3d.io.read_pinhole_camera_parameters(camera_config + "/tmp%d.json"%(index))
            ctr.convert_from_pinhole_camera_parameters(
                param) 
            print(index)           
        else:
            vis.register_animation_callback(None)
            return False

        vis.capture_screen_image(os.path.join(out_path, "render_{}.jpg".format(index)), True)
        return False
    

    ply = o3d.io.read_triangle_mesh(ply_file)
    ply.compute_vertex_normals()
    ply.paint_uniform_color([1, 1, 1])
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window("rendering", width=480, height=640)

    vis.add_geometry(ply)
    vis.get_render_option().load_from_json('../video_poses/render.json')

    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    ply_file='../exps/evaluation/semantic_3_refuse/scene0050_00.ply'
    scene_name='scene0050_00'
    out_path='../exps/rendering'
    camera_config=render_scan(scene_name)
    index = -1
    render(ply_file,out_path,camera_config)