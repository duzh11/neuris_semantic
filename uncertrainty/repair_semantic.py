import os
import cv2
import torch
import numpy as np

from tqdm import tqdm

def repair_semantic(name_lis, intrinsic_np, img_np, pose_np, depth_np, semantic_np, confidence, 
                    similarity_dir, semantic_repair_dir, semantic_repair_vis_dir):
        '''
        修复语义
        '''
        N_img, H, W=semantic_np.shape
        os.makedirs(semantic_repair_dir, exist_ok=True)
        os.makedirs(semantic_repair_vis_dir, exist_ok=True)
        
        for source_idx in tqdm(range(N_img), desc='repairing'):
                # 保存max_similarity以及对应的语义
                max_similarity = torch.from_numpy((np.load(os.path.join(similarity_dir, name_lis[source_idx]+'.npz'))['arr_0'])).flatten().cuda()
                semantic_source = torch.from_numpy(semantic_np[source_idx, :]).flatten().cuda()
                # 将source_img投影到target_img
                pose_source = pose_np[source_idx]
                depth_source = depth_np[source_idx].flatten()

                for target_idx in range(0, N_img):
                        semantic_target = torch.from_numpy(semantic_np[target_idx,:]).flatten().cuda()
                        semantic_similarity =  torch.from_numpy((np.load(os.path.join(similarity_dir, name_lis[target_idx]+'.npz'))['arr_0'])).flatten().cuda()
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

                        # validating
                        # if target_idx==2000:
                        #         img_source, img_target = img_np[source_idx], img_np[target_idx]
                        #         x_source, y_source = 363, 365
                        #         (x_target, y_target) = projected_points[y_source*W+x_source-1,:]
                        #         print(f'correspondence: ({x_target}, {y_target})')

                        #         plt.close('all')
                        #         fig, ax = plt.subplots(1, 3)
                        #         ax[0].imshow(img_source[...,::-1])
                        #         ax[0].plot(x_source, y_source, 'ro', markersize=5)

                        #         ax[1].imshow(img_target[...,::-1])
                        #         ax[1].plot(x_target.cpu().numpy(), y_target.cpu().numpy(), 'ro', markersize=5)
                                
                        #         semantic_source_vis = colour_map_np[semantic_source.cpu().numpy()].reshape(H,W,3)
                        #         semantic_source_vis[~repair_mask.cpu().numpy().reshape(H,W)] = np.array([0,0,0])
                        #         ax[2].imshow(semantic_source_vis)

                        #         plt.tight_layout()
                        #         plt.show()
                        #         print('validate')
                
                semantic_source = semantic_source.cpu().numpy().astype(np.uint16)
                cv2.imwrite(os.path.join(semantic_repair_dir, name_lis[source_idx]+'.png'), semantic_source.reshape(H,W))
                semantic_vis = colour_map_np[semantic_source]
                cv2.imwrite(os.path.join(semantic_repair_vis_dir, name_lis[source_idx]+'.png'), semantic_vis[...,::-1].reshape(H,W,3))