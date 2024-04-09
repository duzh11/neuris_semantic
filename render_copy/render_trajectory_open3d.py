import os
import sys
import open3d as o3d

ply_file = sys.argv[1]
# scan = 'scene0050_00'
# ply_file = f'/home/du/Proj/3Dv_Reconstruction/NeuRIS/exps/indoor/neus/{scan}/neuris_test4/meshes/{scan}.ply'
# out_path = 'render/rendering/'
# Path(out_path).mkdir(exist_ok=True, parents=True)

camera_config = "render_copy/video_poses"

ply = o3d.io.read_triangle_mesh(ply_file)
ply.compute_vertex_normals()
# ply.paint_uniform_color([1, 1, 1])
vis = o3d.visualization.VisualizerWithKeyCallback()

vis.create_window("rendering", width=640, height=480)

index = -1

def move_forward(vis):
    # This function is called within the o3d.visualization.Visualizer::run() loop
    # The run loop calls the function, then re-render
    # So the sequence in this function is to:
    # 1. Capture frame
    # 2. index++, check ending criteria
    # 3. Set camera
    # 4. (Re-render)
    ctr = vis.get_view_control()
    # view_control = o3d.visualization.ViewControl()

    global index

    if index >= 0:
        vis.capture_screen_image(os.path.join(sys.argv[2], "render_{}.jpg".format(index)), False)
        # vis.capture_screen_image(os.path.join(out_path, "render_{}.jpg".format(index)), False)
    index = index + 1
    if index < int(sys.argv[4]) :
        param = o3d.io.read_pinhole_camera_parameters(sys.argv[3] + "/tmp%d.json"%(index))     
        # param = o3d.io.read_pinhole_camera_parameters(camera_config+ "/tmp%d.json"%(index))

        ctr.convert_from_pinhole_camera_parameters(
            param, allow_arbitrary=True)
    else:
        vis.register_animation_callback(None)
        exit(-1)

    return False

vis.add_geometry(ply)
vis.get_render_option().load_from_json('./render_copy/render_semmesh.json')

vis.register_animation_callback(move_forward)
vis.run()
vis.destroy_window()