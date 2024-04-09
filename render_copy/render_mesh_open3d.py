from pathlib import Path
import numpy as np
import os
from glob import glob
import json
import sys
import shutil

def read_file(file):
    data = open(file)

    lines = [[float(w) for w in line.strip().split()] for line in data]
    assert len(lines) == 4
    return np.array(lines).astype(np.float32)  

def glob_data(data_dir):
    data_paths = []
    data_paths.extend(glob(data_dir))
    data_paths = sorted(data_paths)
    return data_paths
 
def render_scan(scan, data_mode, mesh, out_path):
    
    dir_scan = os.path.join(data_dir, scan)
    
    image_paths = glob_data(os.path.join('{0}'.format(dir_scan), f'image/{data_mode}', "*.png"))
    n_images = len(image_paths)
    
    intrinsics_path = '{0}/intrinsic_color_crop1248_resize640.txt'.format(dir_scan)
    intrinsics = read_file(intrinsics_path)

    pose_paths = glob_data(f'{dir_scan}/pose/{data_mode}/*.txt')

    pose_all = []
    for pose_file in pose_paths:
        pose_all.append(read_file(pose_file))
        
    H, W = 480, 640

    # create tmp camera pose file for open3d
    camera_config = Path("render_copy/video_poses")
    camera_config.mkdir(exist_ok=True, parents=True)

    for image_id in range(n_images):
        
        c2w = pose_all[image_id]
        w2c = np.linalg.inv(c2w)

        K = intrinsics.copy()
     
        tmp_json = json.load(open('render_copy/c1.json'))
        tmp_json["extrinsic"] = w2c.T.reshape(-1).tolist()
        
        tmp_json["intrinsic"]["intrinsic_matrix"] = K[:3,:3].T.reshape(-1).tolist()
        tmp_json["intrinsic"]["height"] = H 
        tmp_json["intrinsic"]["width"] = W 
        json.dump(tmp_json, open('render_copy/video_poses/tmp%d.json'%(image_id), 'w'), indent=4)
    
    cmd = f"python ./render_copy/render_trajectory_open3d.py {mesh} \"{out_path}\" {camera_config} \{n_images}"
    os.system(cmd)

scan = sys.argv[1]
method_name = sys.argv[2]
data_mode = sys.argv[3]

data_dir = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/dataset/indoor'
# exps_dir = f'/home/du/Proj/3Dv_Reconstruction/Manhattan_sdf/exp/result/manhattan_sdf/{method_name}_{scan[-7:-3]}'
# mesh_path = f'{exps_dir}/50.ply'
# if os.path.exists(mesh_path):
#     print(f"{mesh_path}")
# else:
#     mesh_path = f'{exps_dir}/49.ply'
#     print(f"{mesh_path}")

exps_dir = f'/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/dataset/indoor/{scan}'
mesh_path = f'{exps_dir}/{scan}_vh_clean_2.ply'
    
out_path = f'{exps_dir}/rendering/colormesh'
Path(out_path).mkdir(exist_ok=True, parents=True)

render_scan(scan, data_mode, mesh_path, out_path)

