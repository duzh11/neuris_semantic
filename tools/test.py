import pyrender
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import trimesh

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

def refuse(mesh, data_loader,scale=None,offset=None):
    #将颜色和几何点云混合？
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)

    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=0.04,
        sdf_trunc=3 * 0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8
    )

    for batch in tqdm(data_loader, desc='Refusing'):
        for b in range(batch['rgb'].shape[0]):
            h, w = batch['meta']['h'].item(), batch['meta']['w'].item()

            intrinsic = np.eye(4)
            intrinsic[:3, :3] = batch['intrinsic'][b].numpy()
            pose = batch['c2w'][b].numpy()
            pose[:3, 3]=pose[:3, 3]/scale+offset #需要变换pose
            rgb = batch['rgb'][b].view(h, w, 3).numpy()
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

def evaluate_3D_mesh_byRefuse(scene_dir,path_mesh_pred,path_mesh_GT):
    image_dir=os.path.join(scene_dir,'image')
    image_list=os.listdir(image_dir)
    image_list.sort(key=lambda _:int(_.split('.')[0]))

    c2w_all=[]
    rgb_all=[]
    
    for imgname in tqdm(image_list, desc='Loading data'):
        c2w = np.loadtxt(f'{scene_dir}/pose/{imgname[:-4]}.txt')
        c2w_all.append(c2w)

        rgb = cv2.imread(f'{scene_dir}/images/{imgname[:-4]}.png')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = (rgb.astype(np.float32) / 255).transpose(2, 0, 1)
        rgb = rgb.reshape(3, -1).transpose(1, 0)

        rgb_all.append(rgb)
    
    mesh_pred=trimesh.load(path_mesh_pred)
    mesh_GT=trimesh.load(path_mesh_GT)
        
    mesh_pred_REFUSE = refuse(mesh_pred, 

dir_dataset = '../Data/dataset/indoor'
name_baseline = 'neuris'
exp_name=name_baseline

dir_results_baseline='../exps/evaluation/'

metrics_eval_all = []

for scene_name in lis_name_scenes:
    logging.info(f'\n\nProcess: {scene_name}')
    dir_dataset = '../Data/dataset/indoor'
    scene_dir=os.path.join(dir_dataset,scene_name,name_baseline)
    path_mesh_pred = f'{dir_results_baseline}/{name_baseline}/{scene_name}.ply'
    path_mesh_GT=
    metrics_eval = evaluate_3D_mesh_byRefuse(scene_dir,path_mesh_pred,path_mesh_GT)

    