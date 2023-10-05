import os

scan_lis = ['scene0616_00']
method_name_lis = ['semantic_40_test1', 'pred_40_test2_a', 'pred_40_test2_compa2']
target_idx= 0
end_idx= -1

for scan in scan_lis:
    for method_name in method_name_lis:
        render_cmd = f'python ./render/render_mesh_open3d.py {scan} {method_name}'
        os.system(render_cmd)

        vedio_cmd=f'python ./render/render_vedio.py {scan} {method_name} {target_idx} {end_idx}'
        os.system(vedio_cmd)
