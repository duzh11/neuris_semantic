import open3d as o3d
import os
mesh_root='../exps/evaluation'
mesh_dir1='neuris'
mesh_dir2='manhattan_newoffset_norefuse'
gt_dir='GT'
#mesh_name1='scene0580_00_clean_bbox_faces_mask.ply'
mesh_name1='scene0580_00_clean_bbox_faces_mask.ply'
mesh_name2='scene0580_00_clean_bbox_faces_mask.ply'
# mesh_name2='scene0580_00_clean_bbox_faces_mask.ply'
neuris_name='scene0580_00_vh_clean_2.ply'
manhattan_name='scene0580_00_manhattan_GT_new.ply'

neuris_file=os.path.join(mesh_root, gt_dir,neuris_name) #NeuRIS_GT
manhattan_file=os.path.join(mesh_root, gt_dir,manhattan_name) #manhattan_GT
mesh_file1=os.path.join(mesh_root, mesh_dir1, mesh_name1) #neuris
mesh_file2=os.path.join(mesh_root, mesh_dir2, mesh_name2) #manhattan

def visualize(mesh_file1, mesh_file2):
    mesh1 = o3d.io.read_triangle_mesh(mesh_file1)
    mesh2 = o3d.io.read_triangle_mesh(mesh_file2)
    # mesh.compute_vertex_normals()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry([mesh1,mesh2])
    vis.run()
    vis.destroy_window()

neuris_gt=o3d.io.read_triangle_mesh(neuris_file)
neuris_gt.compute_vertex_normals()
manhattan_gt=o3d.io.read_triangle_mesh(manhattan_file)
manhattan_gt.compute_vertex_normals()
mesh1 = o3d.io.read_triangle_mesh(mesh_file1)
mesh1.compute_vertex_normals()
mesh2 = o3d.io.read_triangle_mesh(mesh_file2)
mesh2.compute_vertex_normals()

neuris_gt.paint_uniform_color([0, 1, 0])
manhattan_gt.paint_uniform_color([1, 0, 0])
mesh1.paint_uniform_color([1, 0.706, 0])#黄色
mesh2.paint_uniform_color([0, 0.651, 0.929])#蓝色
o3d.visualization.draw_geometries([neuris_gt,mesh2])
