import os, sys
sys.path.append(os.getcwd())

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
import utils.utils_nyu as NyuUtils
colour_map_np = NyuUtils.nyu40_colour_code

def vis_viewAL(count_prob,
               valid_mask,
               semantic, 
               count_vis, 
               mv_con, 
               entropy, 
               kldiv,
               viewAL,
               exp=False):
        H, W = semantic.shape
        img_cat=(255 * np.ones((H, 10, 3))).astype('uint8')
        
        semantic_vis = colour_map_np[semantic.astype(int)]
        count_prob_vis = np.repeat(count_prob[:, :, np.newaxis], 3, axis=2)
        count_prob_vis = (count_prob_vis*255).astype('uint8')

        colormap_func = matplotlib.cm.get_cmap("jet")

        mv_uncertainty_vis = colormap_func(mv_con)[:, :, :3]
        mv_uncertainty_vis = (mv_uncertainty_vis*255).astype('uint8')
        mv_uncertainty_vis[~valid_mask,...]=[0, 0, 0]

        semantic_mvuncer_vis = semantic_vis.copy()
        semantic_mvuncer_vis[~(mv_con==1)]=[0, 0, 0]
        semantic_mvuncer_vis[~valid_mask,...]=[0, 0, 0]

        entropy_vis = colormap_func(entropy)[:, :, :3]
        entropy_vis = (entropy_vis[...,::-1]*255).astype('uint8')
        entropy_vis[~valid_mask,...]=[0, 0, 0]

        kldiv_vis = colormap_func(kldiv)[:, :, :3]
        kldiv_vis = (kldiv_vis[...,::-1]*255).astype('uint8')
        kldiv_vis[~valid_mask,...]=[0, 0, 0]

        viewAL_vis = colormap_func(viewAL)[:, :, :3]
        viewAL_vis = (viewAL_vis[...,::-1]*255).astype('uint8')
        viewAL_vis[~valid_mask,...]=[0, 0, 0]

        if exp:
             entropy_vis = entropy_vis[...,::-1]
             kldiv_vis = kldiv_vis[...,::-1]
             viewAL_vis = viewAL_vis[...,::-1]

        lis=[count_vis, img_cat, semantic_vis[...,::-1], img_cat, count_prob_vis, img_cat, \
             semantic_mvuncer_vis[...,::-1], img_cat, mv_uncertainty_vis, img_cat, entropy_vis, img_cat, kldiv_vis, img_cat, viewAL_vis]
        lis = np.concatenate(lis, axis=1)
        return lis

def compute_projection(H, W, intrinsic, pose_source, pose_target, depth_source, depth_target, confidence=0.2):
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
        
        valid_mask_1 = (torch.norm(world_coordinates - backprojected_points, dim=0) < confidence)
        
        valid_mask = valid_mask_0.clone()
        #Note: consider depth is zero
        depth_mask0 = depth_target>0
        valid_mask[valid_mask_0] = valid_mask[valid_mask_0]&valid_mask_1&depth_mask0
        depth_mask1 = depth_source>0
        valid_mask = valid_mask & depth_mask1
        return projected_points, valid_mask

def entropy_function(prob):
        prob_2_uncertainty = lambda x: torch.sum(-torch.log2(x+1e-12)*x, dim=-1)#todo log2?
        entropy = prob_2_uncertainty(prob)
        return entropy

def kldiv_function(P_info, Q_info):
        prob_2_kldiv = lambda P, Q: torch.sum(P*(torch.log2(P+1e-12) - torch.log2(Q+1e-12)), dim=-1)
        # this function assume input is already in log-probabilities 
        kldiv = prob_2_kldiv(P_info, Q_info)
        return kldiv

def viewAL_uncertainty(name_lis, 
                       intrinsic_np, 
                       img_np, 
                       pose_np, 
                       depth_np, 
                       semantic_np, 
                       uncertainty_dir,
                       logits_lis = None,
                       validate_flag=False):
        '''
        计算viewAL uncertainty
        '''
        N_img, H, W=semantic_np.shape
        uncertainty_vis_dir = uncertainty_dir+'_vis'
        os.makedirs(uncertainty_vis_dir, exist_ok=True)

        for source_idx in tqdm(range(N_img), desc='computing viewAL'):
                semantic_source = torch.from_numpy(semantic_np[source_idx, :].flatten()).cuda()
                logits_source_np = np.load(logits_lis[source_idx])['arr_0']
                logits_source_torch = torch.from_numpy(logits_source_np).cuda()
                logits_source = logits_source_torch.reshape(-1, logits_source_torch.shape[2])
                prob_source = F.softmax(logits_source, dim=-1)

                consistency = torch.zeros_like(semantic_source)
                count = torch.zeros_like(semantic_source)                
                kldiv = torch.zeros_like(semantic_source)
                prob = prob_source.clone()

                ## 将source_img投影到target_img
                pose_source = pose_np[source_idx]
                depth_source = depth_np[source_idx].flatten()

                ## select neighbour pixels
                SELECT_NEIGHBOURS = True
                if SELECT_NEIGHBOURS:
                        source_neighbors = [source_idx-3, source_idx-2, source_idx-1, source_idx+1, source_idx+2, source_idx+3]
                        if source_idx > N_img-4:
                                source_neighbors = [source_idx-3, source_idx-2, source_idx-1, source_idx+1-N_img, source_idx+2-N_img, source_idx+3-N_img]
                else:
                        source_neighbors = range(N_img)

                for target_idx in source_neighbors:
                        if target_idx==source_idx:
                                continue
                        pose_target = pose_np[target_idx]
                        depth_target = depth_np[target_idx].flatten()

                        projected_points, valid_mask= compute_projection(H, W, intrinsic_np, pose_source, pose_target, depth_source, depth_target)
                        ## validating
                        if validate_flag:
                                if (source_idx==0) & (target_idx==3):
                                        plt.close('all')
                                        img_source, img_target = img_np[source_idx], img_np[target_idx]
                                        x_source, y_source =280, 112
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
                        
                        ## 由source投影到target中对应的点的索引
                        projected_points_flat = (projected_points[:, 1] * W + projected_points[:, 0])
                        
                        projected_points_filter = projected_points_flat[valid_mask]
                        semantic_source_filter = semantic_source[valid_mask]
                        
                        ## compute viewAL
                        semantic_target = torch.from_numpy(semantic_np[target_idx, :].flatten()).cuda()
                        logits_target_np = np.load(logits_lis[target_idx])['arr_0']
                        logits_target_torch = torch.from_numpy(logits_target_np).cuda()
                        logits_target = logits_target_torch.reshape(-1, logits_target_torch.shape[2])
                        prob_target = F.softmax(logits_target, dim=-1)

                        semantic_projection = semantic_target[projected_points_filter]
                        prob_projection = prob_target[projected_points_filter,:]
                        prob_source_valid = prob_source[valid_mask]
                        kldiv_target = kldiv_function(prob_source_valid, prob_projection)

                        similarity = (semantic_source_filter==semantic_projection)
                        consistency[valid_mask] += similarity
                        count[valid_mask] += valid_mask[valid_mask]
                        kldiv[valid_mask] += kldiv_target
                        prob[valid_mask,:] += prob_projection
                
                # 考虑存在多个匹配的点
                valid_mask = (count>2)
                count_vis = img_np[source_idx]
                count_vis[~valid_mask.reshape(H,W).cpu().numpy(),:] = [0, 0, 0]
                cv2.imwrite(os.path.join(uncertainty_dir, 'valid_'+name_lis[source_idx]+'.png'), count_vis)
                np.savez(os.path.join(uncertainty_dir, 'valid_'+name_lis[source_idx]+'.npz'), valid_mask.reshape(H,W).cpu().numpy())
                # 多目一致性不确定性
                mv_con = consistency/(count+1e-12)
                mv_con = (mv_con).reshape(H,W).cpu().numpy()
                np.savez(os.path.join(uncertainty_dir, 'mvcon_'+name_lis[source_idx]+'.npz'), mv_con)
                # mv_entropy
                prob = prob/(count.view(-1,1)+1)
                entropy = entropy_function(prob)
                entropy_score = torch.exp(-entropy).reshape(H,W).cpu().numpy()
                np.savez(os.path.join(uncertainty_dir, 'entropy_'+name_lis[source_idx]+'.npz'), entropy.reshape(H,W).cpu().numpy())
                # kl_div
                kldiv = kldiv/(count+1e-12)
                kldiv_score = torch.exp(-kldiv).reshape(H,W).cpu().numpy()
                np.savez(os.path.join(uncertainty_dir, 'kldiv_'+name_lis[source_idx]+'.npz'), kldiv.reshape(H,W).cpu().numpy())
                # entropy*kl_div
                viewAL = entropy*kldiv
                viewAL_score = torch.exp(-viewAL).reshape(H,W).cpu().numpy()
                np.savez(os.path.join(uncertainty_dir, 'viewAL_'+name_lis[source_idx]+'.npz'), viewAL.reshape(H,W).cpu().numpy())
                # vis
                lis = vis_viewAL(count.reshape(H,W).cpu().numpy()/N_img,
                                 valid_mask.reshape(H,W).cpu().numpy(),
                                 semantic_np[source_idx], 
                                 count_vis, 
                                 mv_con, 
                                 entropy.reshape(H,W).cpu().numpy(), 
                                 kldiv.reshape(H,W).cpu().numpy(),
                                 viewAL.reshape(H,W).cpu().numpy())
                cv2.imwrite(os.path.join(uncertainty_vis_dir, 'uncer_'+name_lis[source_idx]+'.png'), lis)

                lis = vis_viewAL(count.reshape(H,W).cpu().numpy()/6,
                                 valid_mask.reshape(H,W).cpu().numpy(),
                                 semantic_np[source_idx], 
                                 count_vis, 
                                 mv_con, 
                                 entropy_score, 
                                 kldiv_score,
                                 viewAL_score,
                                 exp=True)
                cv2.imwrite(os.path.join(uncertainty_vis_dir, 'score_'+name_lis[source_idx]+'.png'), lis)
                
