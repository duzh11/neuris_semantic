import os, sys
sys.path.append(os.getcwd())

method_name = sys.argv[1]
from confs.path import lis_name_scenes
target_idx= 0
end_idx= -1

for scene_name in lis_name_scenes:
    for data_mode in ['train', 'test']:
        render_cmd = f'python ./render/render_mesh_open3d.py {scene_name} {method_name} {data_mode}'
        os.system(render_cmd)

        vedio_cmd=f'python ./render/render_vedio.py {scene_name} {method_name} {data_mode} {target_idx} {end_idx}'
        os.system(vedio_cmd)
