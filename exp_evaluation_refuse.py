import pyrender
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import trimesh
import cv2
import logging
from datetime import datetime
import evaluation.EvalScanNet as EvalScanNet
import utils.utils_io as IOUtils

class Renderer():
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()

def refuse(mesh, intrinsic_depth, rgb_all, c2w_all):
    #将颜色和几何点云混合？
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)

    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=3 * 0.01,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8
    )

    n_image=rgb_all.shape[0]

    for i in tqdm(range(n_image),desc='refusing'):
        h, w = rgb_all[i].shape[0], rgb_all[i].shape[1]

        intrinsic = np.eye(4)
        intrinsic[:3, :3] = intrinsic_depth
        pose = c2w_all[i]
        pose[:3, 3]=pose[:3, 3]
        rgb = rgb_all[i]
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        _, depth_pred = renderer(h, w, intrinsic, pose, mesh_opengl)
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    
    return volume.extract_triangle_mesh()

def evaluate_3D_mesh_byRefuse(scene_name,scene_dir,output_dir,path_mesh_pred,path_mesh_GT,eval_threshold=0.5,check_existence=True):
    image_dir=os.path.join(scene_dir,'image')
    image_list=os.listdir(image_dir)
    image_list.sort(key=lambda _:int(_.split('.')[0]))

    c2w_all=[]
    rgb_all=[]
    
    output_GT_dir=os.path.dirname(output_dir)
    output_GT_dir=os.path.join(output_GT_dir,'GT_refuse')
    os.makedirs(output_GT_dir,exist_ok=True)
    mesh_pred_dir=os.path.join(output_dir,f'{scene_name}.ply')
    mesh_GT_dir=os.path.join(output_GT_dir,f'{scene_name}_GT.ply')
    if check_existence and IOUtils.checkExistence(mesh_pred_dir) and IOUtils.checkExistence(mesh_GT_dir):
        logging.info(f'The pred mesh is already cleaned. [{mesh_pred_dir.split("/")[-1]}]')
        logging.info(f'The GT mesh is already cleaned. [{mesh_GT_dir.split("/")[-1]}]')
        metrices_eval = EvalScanNet.evaluate_geometry_neucon(mesh_pred_dir, mesh_GT_dir, 
                                            threshold=eval_threshold, down_sample=.02) 
        return metrices_eval


    for imgname in tqdm(image_list, desc='Loading data'):
        c2w = np.loadtxt(f'{scene_dir}/pose/{imgname[:-4]}.txt')
        c2w_all.append(c2w)

        rgb = cv2.imread(f'{image_dir}/{imgname[:-4]}.png')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = (rgb.astype(np.float32) / 255)

        rgb_all.append(rgb)
    
    intrinsic_dir=os.path.join(scene_dir,'intrinsic_depth.txt')
    intrinsic_depth=np.loadtxt(intrinsic_dir)

    if check_existence and IOUtils.checkExistence(mesh_pred_dir):
        logging.info(f'The pred mesh is already cleaned. [{mesh_pred_dir.split("/")[-1]}]')
    else:
        mesh_pred=trimesh.load(path_mesh_pred)
        mesh_pred_REFUSE = refuse(mesh_pred, intrinsic_depth[:3, :3], np.array(rgb_all), np.array(c2w_all))
        o3d.io.write_triangle_mesh(mesh_pred_dir, mesh_pred_REFUSE)

    if check_existence and IOUtils.checkExistence(mesh_GT_dir):
        logging.info(f'The GT mesh is already cleaned. [{mesh_GT_dir.split("/")[-1]}]')
    else:
        mesh_GT=trimesh.load(path_mesh_GT)
        mesh_GT_REFUSE = refuse(mesh_GT, intrinsic_depth[:3, :3], np.array(rgb_all), np.array(c2w_all))
        o3d.io.write_triangle_mesh(mesh_GT_dir, mesh_GT_REFUSE)

    metrices_eval = EvalScanNet.evaluate_geometry_neucon(mesh_pred_dir, mesh_GT_dir, 
                                            threshold=eval_threshold, down_sample=.02) 
    return metrices_eval

def main(exp_name,lis_name_scenes):
    dir_dataset = '../Data/dataset/indoor'
    name_baseline=f'{exp_name}_refuse'
    dir_results_baseline='../exps/evaluation/'
    
    eval_threshold = 0.05
    metrics_eval_all = []
    for scene_name in lis_name_scenes:
        logging.info(f'\n\nProcess: {scene_name}')
        dir_dataset = '../Data/dataset/indoor'
        scene_dir=os.path.join(dir_dataset,scene_name)
        output_dir=f'{dir_results_baseline}/{name_baseline}'
        os.makedirs(output_dir,exist_ok=True)
        logging.info(f'output_dir:{output_dir}')
        path_mesh_pred = f'{dir_results_baseline}/{exp_name}/{scene_name}.ply'
        path_mesh_GT=os.path.join(scene_dir,f'{scene_name}_vh_clean_2_clean.ply')
        metrics_eval = evaluate_3D_mesh_byRefuse(scene_name,scene_dir,output_dir,path_mesh_pred,path_mesh_GT,eval_threshold)
        metrics_eval_all.append(metrics_eval)
    
    metrics_eval_all = np.array(metrics_eval_all)
    str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path_log = f'{dir_results_baseline}/{name_baseline}/eval_{name_baseline}_3Dmesh_thres{eval_threshold}_{str_date}_markdown.txt'
    
    latex_header=f'{exp_name}\n  scene_name         Accu.      Comp.      Prec.     Recall     F-score \n'
    markdown_header=f'\n| scene_name   |    Method|    Accu.|    Comp.|    Prec.|   Recall|  F-score| \n'
    markdown_header=markdown_header+'| -------------| ---------| ------- | ------- | ------- | ------- | ------- |\n'
    EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                    header = markdown_header, 
                                                    exp_name=exp_name,
                                                    results = metrics_eval_all, 
                                                    names_item = lis_name_scenes, 
                                                    save_mean = True, 
                                                    mode = 'w')

if __name__=='__main__':
    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    exp_name = 'semantic_3_test4'
    lis_name_scenes=['scene0084_00','scene0616_00']
    # lis_name_scenes=['scene0009_01','scene0050_00','scene0084_00','scene0616_00']
    main(exp_name,lis_name_scenes)

    