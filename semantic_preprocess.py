import torch
import cv2
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import utils.utils_colour as utils_colour
from glob import glob
from tqdm import tqdm

def compute_projection(H, W, intrinsic, pose_source, pose_target, depth_source, depth_target):
        ### 1. first project source point to world
        # acquiring camera coordinates
        x = np.linspace(0, W-1, W)
        y = np.linspace(0, H-1, H)
        x_mesh, y_mesh = np.meshgrid(x, y)
        u, v =torch.from_numpy(x_mesh).type(torch.cuda.FloatTensor).flatten(), torch.from_numpy(y_mesh).type(torch.cuda.FloatTensor).flatten()
        # reproject (u,v) to word
        depth_source = torch.from_numpy(depth_source).type(torch.cuda.FloatTensor)
        I = torch.zeros(4, depth_source.shape[0]).type(torch.cuda.FloatTensor)
        I[0, :] = u * depth_source
        I[1, :] = v * depth_source
        I[2, :] = depth_source
        I[3, :] = 1
        world_coordinates = torch.mm(torch.from_numpy(pose_source).type(torch.cuda.FloatTensor), torch.mm(
                torch.from_numpy(np.linalg.inv(intrinsic)).type(torch.cuda.FloatTensor), I))
        
        ### 2. Then project world point to target
        target_points = torch.mm(torch.from_numpy(intrinsic).type(torch.cuda.FloatTensor),
                torch.mm( torch.from_numpy(np.linalg.inv(pose_target)).type(torch.cuda.FloatTensor), world_coordinates))
        # normalized
        target_points = target_points.transpose(0, 1)[:, :3]
        target_points[:, 0] /= target_points[:, 2]
        target_points[:, 1] /= target_points[:, 2]
        target_points = (torch.round(target_points[:, :2])).long()
        
        projected_points = target_points.clone()
        # make sure the point is in the image
        valid_mask_0 = (target_points[:, 0] >= 0) & (target_points[:, 0] < W) & (
                                target_points[:, 1] >= 0) & (target_points[:, 1] < H)
        
        ### 3. Lastly backproject target to world and check
        # Note: depth_target is no match target_points
        target_points = target_points[valid_mask_0, :2]
        world_coordinates = world_coordinates[:, valid_mask_0]

        depth_target = torch.from_numpy(depth_target).type(torch.cuda.FloatTensor)
        depth_target = depth_target[target_points[:, 1]*W+target_points[:, 0]]
        
        I = torch.zeros(4, depth_target.shape[0]).type(torch.cuda.FloatTensor)
        I[0, :] = target_points[:, 0] * depth_target
        I[1, :] = target_points[:, 1] * depth_target
        I[2, :] = depth_target
        I[3, :] = 1
        backprojected_points = torch.mm(torch.from_numpy(pose_target).type(torch.cuda.FloatTensor), torch.mm(
                torch.from_numpy(np.linalg.inv(intrinsic)).type(torch.cuda.FloatTensor), I))
        
        valid_mask_1 = (torch.norm(world_coordinates - backprojected_points, dim=0) < 0.2)
        
        valid_mask = valid_mask_0.clone()
        #Note: consider depth is zero
        depth_mask0 = depth_target>0
        valid_mask[valid_mask_0] = valid_mask[valid_mask_0]&valid_mask_1&depth_mask0
        depth_mask1 = depth_source>0
        valid_mask = valid_mask & depth_mask1
        return projected_points, valid_mask


def compute_similarity(intrinsic_np, img_np, pose_np, depth_np, semantic_np, similarity_dir):
        '''
        计算mv_similarity
        '''
        N_img, H, W=semantic_np.shape
        os.makedirs(similarity_dir, exist_ok=True)
        for source_idx in tqdm(range(N_img), desc='computing mv_similarity'):
                semantic_source = torch.from_numpy(semantic_np[source_idx, :].flatten()).cuda()
                consistency = torch.zeros_like(semantic_source)
                count = torch.zeros_like(semantic_source)

                # 将source_img投影到target_img
                pose_source = pose_np[source_idx]
                depth_source = depth_np[source_idx].flatten()
                for target_idx in range(N_img):
                        pose_target = pose_np[target_idx]
                        depth_target = depth_np[target_idx].flatten()
                        semantic_target = torch.from_numpy(semantic_np[target_idx, :].flatten()).cuda()
                        projected_points, valid_mask= compute_projection(H, W, intrinsic_np, pose_source, pose_target, depth_source, depth_target)
                        # validating
                        if target_idx==2000:
                                plt.close('all')
                                img_source, img_target = img_np[source_idx], img_np[target_idx]
                                x_source, y_source =400, 120
                                (x_target, y_target) = projected_points[y_source*W+x_source-1,:]
                                print(f'correspondence: ({x_target}, {y_target})')
                                fig, ax = plt.subplots(1, 3)
                                ax[0].imshow(img_source[...,::-1])
                                ax[0].plot(x_source, y_source, 'ro', markersize=5)

                                ax[1].imshow(img_target[...,::-1])
                                ax[1].plot(x_target.cpu().numpy(), y_target.cpu().numpy(), 'ro', markersize=5)

                                mask_0 = valid_mask.cpu().numpy().reshape(H,W)
                                img_mask0 = img_source.copy()
                                img_mask0[~mask_0]=np.array([0,0,0])
                                ax[2].imshow(img_mask0[...,::-1])                              

                                plt.tight_layout()
                                plt.show()
                                print('validate')
                        
                        # 由source投影到target中对应的点的索引
                        projected_points_flat = (projected_points[:, 1] * W + projected_points[:, 0])
                        
                        projected_points_filter = projected_points_flat[valid_mask]
                        semantic_source_filter = semantic_source[valid_mask]
                        
                        # compute semantic consistency
                        semantic_projection = semantic_target[projected_points_filter]
                        similarity = (semantic_source_filter==semantic_projection)
                        consistency[valid_mask] += similarity
                        count[valid_mask] += valid_mask[valid_mask]
                
                # 认为出现次数出现一定次数以上为正确语义
                global_similarity = consistency/count
                # 只计算count>0的点
                count_mask = count>0
                global_similarity[~count_mask]=0
                
                semantic_similarity = global_similarity.reshape(H,W)
                np.savez(os.path.join(similarity_dir, f'{source_idx}.npz'), semantic_similarity.cpu().numpy())

# 可视化
def visualize_similarity(img_np, semantic_np, semantic_vis_np, eva, similarity_dir, similarity_vis_dir):
        '''
        可视化mv_similarity
        '''
        N_img, H, W=semantic_np.shape
        similarity_vis_dir = os.path.join(similarity_vis_dir, f'{eva}')
        os.makedirs(similarity_vis_dir, exist_ok=True)
        video_name = os.path.join(similarity_vis_dir, 'mv_similarity.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
        videowriter = cv2.VideoWriter(video_name, fourcc, 10, (W*4+30, H))
        
        for source_idx in tqdm(range(N_img), desc='visualizing'):
                img = img_np[source_idx]
                semantic_similarity = np.load(os.path.join(similarity_dir, f'{source_idx}.npz'))['arr_0']
                # semantic 原图
                semantic_vis = semantic_vis_np[source_idx]
                semantic_mask_vis = semantic_vis.copy()
                # 热力图
                colormap_func = matplotlib.cm.get_cmap("jet")
                semantic_similarity_vis = colormap_func(semantic_similarity)[:, :, :3]
                semantic_similarity_vis = (semantic_similarity_vis*255).astype('uint8')
                # eva = 0.5
                semantic_mask_np = semantic_similarity>eva
                semantic_mask_vis[~semantic_mask_np]=np.array([0,0,0])
                
                img_cat=(255 * np.ones((H, 10, 3))).astype('uint8')
                lis=[img, img_cat, semantic_vis, img_cat, semantic_similarity_vis, img_cat, semantic_mask_vis]
                vis = np.concatenate(lis, axis=1)
                cv2.imwrite(os.path.join(similarity_vis_dir, f'{source_idx}.png'), vis)
                videowriter.write(vis.astype(np.uint8))
        
        videowriter.release()

def repair_semantic(intrinsic_np, img_np, pose_np, depth_np, semantic_np, confidence, 
                    similarity_dir, semantic_repair_dir, semantic_repair_vis_dir):
        '''
        修复语义
        '''
        N_img, H, W=semantic_np.shape
        os.makedirs(semantic_repair_dir, exist_ok=True)
        os.makedirs(semantic_repair_vis_dir, exist_ok=True)
        
        for source_idx in tqdm(range(N_img), desc='repairing'):
                # 保存max_similarity以及对应的语义
                max_similarity = torch.from_numpy((np.load(os.path.join(similarity_dir, f'{source_idx}.npz'))['arr_0'])).flatten().cuda()
                semantic_source = torch.from_numpy(semantic_np[source_idx, :]).flatten().cuda()
                # 将source_img投影到target_img
                pose_source = pose_np[source_idx]
                depth_source = depth_np[source_idx].flatten()

                for target_idx in range(0, N_img):
                        semantic_target = torch.from_numpy(semantic_np[target_idx,:]).flatten().cuda()
                        semantic_similarity =  torch.from_numpy((np.load(os.path.join(similarity_dir, f'{target_idx}.npz'))['arr_0'])).flatten().cuda()
                        # project source to target
                        pose_target = pose_np[target_idx]
                        depth_target = depth_np[target_idx].flatten()
                        projected_points, valid_mask = compute_projection(H, W, intrinsic_np, pose_source, pose_target, depth_source, depth_target)
                        projected_points_flat = (projected_points[:, 1] * W + projected_points[:, 0])
                        projected_points_filter = projected_points_flat[valid_mask]
                        
                        # 在valid_mask中决定是否进行修复
                        similairty_projection = semantic_similarity[projected_points_filter]

                        mask0 = (similairty_projection>max_similarity[valid_mask])
                        mask1 = (similairty_projection>confidence)
                        mask = mask0&mask1

                        repair_mask = valid_mask.clone()
                        repair_mask[valid_mask] = valid_mask[valid_mask] & mask

                        # 修复
                        projected_points_repair = projected_points_flat[repair_mask]
                        semantic_source[repair_mask] = semantic_target[projected_points_repair]
                        max_similarity[repair_mask] = semantic_similarity[projected_points_repair]

                        # 找到max_simi对应的语义
                        
                        if target_idx==2000:
                                img_source, img_target = img_np[source_idx], img_np[target_idx]
                                x_source, y_source = 363, 365
                                (x_target, y_target) = projected_points[y_source*W+x_source-1,:]
                                print(f'correspondence: ({x_target}, {y_target})')

                                plt.close('all')
                                fig, ax = plt.subplots(1, 3)
                                ax[0].imshow(img_source[...,::-1])
                                ax[0].plot(x_source, y_source, 'ro', markersize=5)

                                ax[1].imshow(img_target[...,::-1])
                                ax[1].plot(x_target.cpu().numpy(), y_target.cpu().numpy(), 'ro', markersize=5)
                                
                                semantic_source_vis = colour_map_np[semantic_source.cpu().numpy()].reshape(H,W,3)
                                semantic_source_vis[~repair_mask.cpu().numpy().reshape(H,W)] = np.array([0,0,0])
                                ax[2].imshow(semantic_source_vis)

                                plt.tight_layout()
                                plt.show()
                                print('validate')
                
                cv2.imwrite(os.path.join(semantic_repair_dir, '%d.png'%source_idx), semantic_source.cpu().numpy().reshape(H,W))
                semantic_vis = colour_map_np[semantic_source.cpu().numpy()]
                cv2.imwrite(os.path.join(semantic_repair_vis_dir, '%d.png'%source_idx), semantic_vis[...,::-1].reshape(H,W,3))

                
if __name__=='__main__':
        Data_dir = '../Data/dataset/indoor'
        scene_list = ['scene0616_00']
        semantic_type_list = ['semantic_pred']
        confidence_list = [0.3, 0.5]
        eva_list=[0.3, 0.5]
        colour_map_np = utils_colour.nyu40_colour_code

        for scene_name in scene_list:
                print(f'---scene_name :{scene_name}----')
                for semantic_type in semantic_type_list:
                        print(f'---semantics :{semantic_type}----')
                        ### 1.read data
                        semantic_dir = os.path.join(Data_dir, scene_name, semantic_type)
                        semantic_vis_dir = os.path.join(Data_dir, scene_name, f'{semantic_type}_vis')
                        img_dir = os.path.join(Data_dir, scene_name, 'image')
                        depth_dir = os.path.join(Data_dir, scene_name, 'depth')
                        pose_dir = os.path.join(Data_dir, scene_name, 'pose')
                        intrinsic_dir = os.path.join(Data_dir, scene_name, 'intrinsic_color_crop1248_resize640.txt')

                        semantic_lis = glob(f'{semantic_dir}/*.png')
                        semantic_lis.sort(key=lambda x:int((x.split('/')[-1]).split('.')[0]))
                        semantic_vis_lis = glob(f'{semantic_vis_dir}/*.png')
                        semantic_vis_lis.sort(key=lambda x:int((x.split('/')[-1]).split('.')[0]))
                        img_lis = sorted(glob(f'{img_dir}/*.png'))
                        depth_lis = sorted(glob(f'{depth_dir}/*.png'))
                        pose_lis = sorted(glob(f'{pose_dir}/*.txt'))

                        semantic_np = np.stack([cv2.imread(seg_name)[:,:,0] for seg_name in semantic_lis])
                        semantic_vis_np = np.stack([cv2.imread(seg_vis_name) for seg_vis_name in semantic_vis_lis])
                        img_np = np.stack([cv2.imread(img_name) for img_name in img_lis])
                        depth_np = np.stack([cv2.imread(depth_name, cv2.IMREAD_UNCHANGED)/1000 for depth_name in depth_lis])
                        pose_np = np.stack([np.loadtxt(pose_name) for pose_name in pose_lis])
                        intrinsic_np = np.loadtxt(intrinsic_dir)

                        ### 2.compute mv_similarity
                        similarity_dir = os.path.join(Data_dir, scene_name, 'mv_similarity', semantic_type)
                        compute_similarity(intrinsic_np, img_np, pose_np, depth_np, semantic_np, similarity_dir)

                        # visualize mv_similarity
                        similarity_vis_dir = os.path.join(Data_dir, scene_name, 'mv_similarity_vis', semantic_type)
                        for eva in eva_list:
                                visualize_similarity(img_np, semantic_np, semantic_vis_np, eva, similarity_dir, similarity_vis_dir)
                        
                        ### 3. repair semantic
                        for confidence in confidence_list:
                                semantic_repair_type=semantic_type+f'_repair_{confidence}'

                                semantic_repair_dir = os.path.join(Data_dir, scene_name, semantic_repair_type)
                                semantic_repair_vis_dir = os.path.join(Data_dir, scene_name, f'{semantic_repair_type}_vis')
                                repair_semantic(intrinsic_np, img_np, pose_np, depth_np, semantic_np, confidence, 
                                                similarity_dir, semantic_repair_dir, semantic_repair_vis_dir)

                                ### 4. repeat computing and visualize mv_similarity
                                semantic_repair_lis = glob(os.path.join(semantic_repair_dir, '*.png'))
                                semantic_repair_lis.sort(key=lambda x:int((x.split('/')[-1]).split('.')[0]))
                                semantic_repair_np = np.stack([cv2.imread(semantic_repair)[:,:,0] for semantic_repair in semantic_repair_lis])
                                
                                semantic_repair_vis_lis = glob(f'{semantic_repair_vis_dir}/*.png')
                                semantic_repair_vis_lis.sort(key=lambda x:int((x.split('/')[-1]).split('.')[0]))
                                semantic_repair_vis_np = np.stack([cv2.imread(seg_vis_name) for seg_vis_name in semantic_repair_vis_lis])

                                repair_similarity_dir = os.path.join(Data_dir, scene_name, 'mv_similarity', semantic_repair_type)
                                compute_similarity(intrinsic_np, img_np, pose_np, depth_np, semantic_repair_np, repair_similarity_dir)
                                repair_similarity_vis_dir = os.path.join(Data_dir, scene_name, 'mv_similarity_vis', semantic_repair_type)
                                for eva in eva_list:
                                        visualize_similarity(img_np, semantic_repair_np, semantic_repair_vis_np, eva, 
                                                             repair_similarity_dir, repair_similarity_vis_dir)