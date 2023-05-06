import pyrender
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import trimesh
import cv2
import logging
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

def refuse_mesh(scene_name,
                 scene_dir,
                 output_dir,
                 path_mesh_pred,
                 path_mesh_GT,
                 check_existence=True):
    
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
        return mesh_pred_dir,mesh_GT_dir

    for imgname in tqdm(image_list, desc='Loading data'):
        c2w = np.loadtxt(f'{scene_dir}/pose/{imgname[:-4]}.txt')
        c2w_all.append(c2w)

        rgb = cv2.imread(f'{image_dir}/{imgname[:-4]}.png')
        reso=640/rgb.shape[0]
        
        if reso>2:
            rgb=rgb.astype(np.uint8)
            rgb=cv2.pyrUp(rgb)
            # rgb=cv2.resize(rgb, (480,640), interpolation=cv2.INTER_NEAREST)
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

    return mesh_pred_dir, mesh_GT_dir

def label3D(exp_name,
            scene_name,
             scene_dir,
             output_dir,
             path_mesh_pred,
             check_existence=True):
    
    image_dir=os.path.join(scene_dir,'image')
    image_list=os.listdir(image_dir)
    image_list.sort(key=lambda _:int(_.split('.')[0]))

    c2w_all=[]
    rgb_all=[]
    
    mesh_label_ply=os.path.join(output_dir,f'{scene_name}_label.ply')
    semantic_dir=os.path.join('../exps/indoor/neus/',scene_name,exp_name,'semantic_render_vis')

    if check_existence and IOUtils.checkExistence(mesh_label_ply):
        logging.info(f'The mesh is already labeled. [{mesh_label_ply.split("/")[-1]}]')

    for imgname in tqdm(image_list, desc='Loading data'):
        c2w = np.loadtxt(f'{scene_dir}/pose/{imgname[:-4]}.txt')
        c2w_all.append(c2w)

        rgb = cv2.imread(f'{semantic_dir}/'+'00160000_'+imgname[:-4]+'_reso2.png')
        reso=640/rgb.shape[1]
        
        if reso>1:
            rgb=rgb.astype(np.uint8)
            rgb=cv2.resize(rgb, (640,480), interpolation=cv2.INTER_NEAREST)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = (rgb.astype(np.float32) / 255)
        
        rgb_all.append(rgb)
    
    intrinsic_dir=os.path.join(scene_dir,'intrinsic_depth.txt')
    intrinsic_depth=np.loadtxt(intrinsic_dir)

    mesh_pred=trimesh.load(path_mesh_pred)
    mesh_pred_REFUSE = refuse(mesh_pred, intrinsic_depth[:3, :3], np.array(rgb_all), np.array(c2w_all))
    o3d.io.write_triangle_mesh(mesh_label_ply, mesh_pred_REFUSE)

    return mesh_label_ply
   


