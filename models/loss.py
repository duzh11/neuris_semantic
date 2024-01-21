import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import logging
from itertools import combinations

import utils.utils_training as TrainingUtils

MIN_PIXELS_PLANE = 20

def get_normal_consistency_loss(normals, mask_curr_plane, error_mode = 'angle_error'):
    '''Return normal loss of pixels on the same plane

    Return:
        normal_consistency_loss: float, on each pixels
        num_pixels_curr_plane: int
        normal_mean_curr, 3*1
    '''
    num_pixels_curr_plane = mask_curr_plane.sum()
    if num_pixels_curr_plane < MIN_PIXELS_PLANE:
        return 0.0,  num_pixels_curr_plane, torch.zeros(3)

    normals_fine_curr_plane = normals * mask_curr_plane
    normal_mean_curr = normals_fine_curr_plane.sum(dim=0) / num_pixels_curr_plane

    if error_mode == 'angle_error':
        inner = (normals * normal_mean_curr).sum(dim=-1,keepdim=True)
        norm_all =  torch.linalg.norm(normals, dim=-1, ord=2,keepdim=True)
        norm_mean_curr = torch.linalg.norm(normal_mean_curr, dim=-1, ord=2,keepdim=True)
        angles = torch.arccos(inner/((norm_all*norm_mean_curr) + 1e-6)) #.clip(-np.pi, np.pi)
        angles = angles*mask_curr_plane
        normal_consistency_loss = F.l1_loss(angles, torch.zeros_like(angles), reduction='sum')
    
    return normal_consistency_loss, num_pixels_curr_plane, normal_mean_curr

def get_plane_offset_loss(pts, ave_normal, mask_curr_plane, mask_subplanes):
    '''
    Args:
        pts: pts in world coordinates
        normals: normals of pts in world coordinates
        mask_plane: mask of pts which belong to the same plane
    '''
    mask_subplanes_curr = copy.deepcopy(mask_subplanes)
    mask_subplanes_curr[mask_curr_plane == False] = 0 # only keep subplanes of current plane
    
    loss_offset_subplanes = []
    num_subplanes = int(mask_subplanes_curr.max().item())
    if num_subplanes < 1:
        return 0, 0
    
    num_pixels_valid_subplanes = 0
    loss_offset_subplanes = torch.zeros(num_subplanes)
    for i in range(num_subplanes):
        curr_subplane = (mask_subplanes_curr == (i+1))
        num_pixels = curr_subplane.sum()
        if num_pixels < MIN_PIXELS_PLANE:
            continue
        
        offsets = (pts*ave_normal).sum(dim=-1,keepdim=True)
        ave_offset = ((offsets * curr_subplane).sum() / num_pixels) #.detach()  # detach?

        diff_offsset = (offsets-ave_offset)*curr_subplane
        loss_tmp = F.mse_loss(diff_offsset, torch.zeros_like(diff_offsset), reduction='sum') #/ num_pixels

        loss_offset_subplanes[i] = loss_tmp
        num_pixels_valid_subplanes += num_pixels
    
    return loss_offset_subplanes.sum(), num_pixels_valid_subplanes

def get_manhattan_normal_loss(normal_planes):
    '''The major planes should be vertical to each other
    '''
    normal_planes = torch.stack(normal_planes, dim=0)
    num_planes = len(normal_planes)
    assert num_planes < 4
    if num_planes < 2:
        return 0

    all_perms = np.array( list(combinations(np.arange(num_planes),2)) ).transpose().astype(int) # 2*N
    normal1, normal2 = normal_planes[all_perms[0]], normal_planes[all_perms[1]]
    inner = (normal1 * normal2).sum(-1)
    manhattan_normal_loss = F.l1_loss(inner, torch.zeros_like(inner), reduction='mean')
    return manhattan_normal_loss

class NeuSLoss(nn.Module):
    def __init__(self, semantic_class=None,conf=None):
        super().__init__()
        self.iter_step = 0
        self.iter_end = -1

        self.color_weight = conf['color_weight']

        self.igr_weight = conf['igr_weight']
        self.smooth_weight = conf['smooth_weight']
        self.mask_weight = conf['mask_weight']

        self.depth_weight = conf['depth_weight']
        self.normal_weight = conf['normal_weight']
        
        self.warm_up_start = conf['warm_up_start']
        self.warm_up_end = conf['warm_up_end']

        self.normal_consistency_weight = conf['normal_consistency_weight']
        self.plane_offset_weight = conf['plane_offset_weight']
        self.manhattan_constrain_weight = conf['manhattan_constrain_weight']
        self.plane_loss_milestone = conf['plane_loss_milestone']

        self.plane_depth_weight = conf['plane_depth_weight']
        if self.plane_depth_weight>0:
            self.planedepth_loss_milestone = conf['planedepth_loss_milestone']
            self.plane_depth_mode = conf['plane_depth_mode']
            logging.info('planedepth start: {}, planedepth_mode: {}'.format(self.planedepth_loss_milestone, 
                                                                            self.plane_depth_mode))

        self.semantic_weight = conf['semantic_weight']
        self.semantic_class=semantic_class
        self.joint_weight = conf['joint_weight']
        self.sv_con_weight = conf['sv_con_weight']
        self.sem_con_weight = conf['sem_con_weight']
        if self.semantic_weight>0:
            self.ce_mode = conf['ce_mode'] if 'ce_mode' in conf else 'ce_loss'
            logging.info('ce_mode: {}'.format(self.ce_mode))

        if (self.semantic_weight>0) and (self.joint_weight>0):
            self.joint_loss_milestone = conf['joint_loss_milestone']
            self.joint_mode = conf['joint_mode'] if 'joint_mode' in conf else 'true_se'
            logging.info('joint start: {}, joint_mode: {}'.format(self.joint_loss_milestone, 
                                                                  self.joint_mode))
        
        if (self.semantic_weight>0) and (self.sv_con_weight>0):
            self.svcon_loss_milestone = conf['svcon_loss_milestone']
            self.sv_con_mode = conf['sv_con_mode'] if 'sv_con_mode' in conf else 'num'
            self.sv_con_loss = conf['sv_con_loss'] if 'sv_con_loss' in conf else 'prob_mean'
            logging.info('svcon start: {}, sv_con_mode: {}, sv_con_loss: {}'.format(self.svcon_loss_milestone, 
                                                                                    self.sv_con_mode,
                                                                                    self.sv_con_loss) )
        
        if (self.semantic_weight>0) and (self.sem_con_weight>0):
            self.sem_con_loss = conf['sem_con_loss'] if 'sem_con_loss' in conf else 'prob_mean'
            logging.info('sem_con_loss: {}'.format(self.sem_con_loss))

    def get_warm_up_ratio(self):
        if self.warm_up_end == 0.0:
            return 1.0
        elif self.iter_step < self.warm_up_start:
            return 0.0
        else:
            return np.min([1.0, (self.iter_step - self.warm_up_start) / (self.warm_up_end - self.warm_up_start)])

    def forward(self, input_model, render_out, sdf_network_fine, patchmatch_out = None, theta=0):
        true_rgb = input_model['true_rgb']

        mask, rays_o, rays_d, near, far = input_model['mask'], input_model['rays_o'], input_model['rays_d'],  \
                                                    input_model['near'], input_model['far']
        mask_sum = mask.sum() + 1e-5
        batch_size = len(rays_o)

        color_fine = render_out['color_fine']
        semantic_fine = render_out['semantic_fine']
        variance = render_out['variance']
        cdf_fine = render_out['cdf_fine']
        gradient_error_fine = render_out['gradient_error_fine']
        weight_max = render_out['weight_max']
        weight_sum = render_out['weight_sum']
        weights = render_out['weights']
        depth = render_out['depth']

        planes_gt = None
        if 'planes_gt' in input_model:
            planes_gt = input_model['planes_gt']
        if 'subplanes_gt' in input_model:
            subplanes_gt = input_model['subplanes_gt']
        
        logs_summary = {}

        # patchmatch loss
        normals_target, mask_use_normals_target = None, None
        pts_target, mask_use_pts_target = None, None
        if patchmatch_out is not None:
            if patchmatch_out['patchmatch_mode'] == 'use_geocheck':
                mask_use_normals_target = (patchmatch_out['idx_scores_min'] > 0).float()
                normals_target = input_model['normals_gt']
            else:
                raise NotImplementedError
        else:
            if self.normal_weight>0:
                normals_target = input_model['normals_gt']
                mask_use_normals_target = torch.ones(batch_size, 1).bool()

        # Color loss
        color_fine_loss, background_loss, psnr = 0, 0, 0
        if True:
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
                
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            # Mask loss, optional
            background_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
            logs_summary.update({           
                'Loss/loss_color':  color_fine_loss.detach().cpu(),
                'Loss/loss_bg':     background_loss
            })
        
        # ce_loss
        semantic_fine_loss=0.
        if self.semantic_weight>0:
            true_semantic = input_model['true_semantic']
            semantic_fine_0 = semantic_fine.reshape(-1, self.semantic_class)
            true_semantic_0 = true_semantic.reshape(-1).long()
            
            # ce loss    
            if self.semantic_class==3:
                CrossEntropyLoss = nn.CrossEntropyLoss()
                crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label)
            else:
                CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-1)
                crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label-1)

            if self.ce_mode == 'ce_loss':
                semantic_fine_loss = crossentropy_loss(semantic_fine_0, true_semantic_0)
            
            elif self.ce_mode == 'ce_loss_weight':
                ce_weights = torch.ones(self.semantic_class) #todo 手动设置了weights
                ce_weights[0]=0.5
                ce_weights[1]=0.5
                if self.semantic_class==3:
                    CrossEntropyLoss = nn.CrossEntropyLoss(weight = ce_weights)
                    crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label)
                else:
                    CrossEntropyLoss = nn.CrossEntropyLoss(weight = ce_weights, ignore_index=-1)
                    crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label-1)
                semantic_fine_loss = crossentropy_loss(semantic_fine_0, true_semantic_0)

            elif self.ce_mode =='nll_loss':
                if self.semantic_class==3:
                    semantic_fine_loss = F.nll_loss(semantic_fine_0, true_semantic_0, ignore_index=-1)
                else:
                    semantic_fine_loss = F.nll_loss(semantic_fine_0, true_semantic_0-1, ignore_index=-1)
            
            elif self.ce_mode == 'ce_loss_score':
                if self.semantic_class==3:
                    CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
                    crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label)
                else:
                    CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
                    crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label-1)

                semantic_score = input_model['semantic_score']
                semantic_fine_loss_0 = crossentropy_loss(semantic_fine_0, true_semantic_0)
                semantic_fine_loss = torch.sum(semantic_score.squeeze()*semantic_fine_loss_0)/(semantic_score.sum() + 1e-6)

            logs_summary.update({           
                'Loss/loss_semantic':  semantic_fine_loss.detach().cpu(),
            })
        
        # sv_con loss
        sv_con_loss=0
        sem_con_loss=0
        if self.semantic_weight>0 and (self.sv_con_weight>0 or self.sem_con_weight>0) and self.iter_step > self.svcon_loss_milestone:
            grid=input_model['grid']
            #todo 选择render semantic还是true semantic
            semantic_score = F.softmax(semantic_fine, dim=-1)
            semantic = semantic_score.argmax(axis=-1) #0-39
            
            uncertainty = render_out['sem_uncertainty_fine'].squeeze()
            # uncertainty>0应该越小越好
            if self.sv_con_weight>0:
                if self.sv_con_mode == 'uncertainty'  or self.sv_con_mode == 'uncertainty_prob':
                    uncertainty_score = 1-uncertainty
                    uncertainty_score = uncertainty_score.clip(0, 1)
                elif self.sv_con_mode == 'uncertainty_exp' or self.sv_con_mode == 'uncertainty_exp_prob':
                    uncertainty_score = torch.exp(-uncertainty)
                else:
                    uncertainty_score = uncertainty
            
            # 每个grids内部语义应该一致
            grid_list=torch.unique(grid)
            for grid_idx in grid_list:
                if grid_idx==0:
                    continue #忽略void类别的grid
                grid_mask = (grid==grid_idx)
                
                semantic_grid = semantic[grid_mask.squeeze()]
                semantic_score_grid = semantic_score[grid_mask.squeeze()]   
                
                # sem_con loss
                if self.sem_con_weight>0:
                    semantic_score_mean = torch.mean(semantic_score_grid, axis=0)
                    if self.sem_con_loss == 'prob':
                        l1_loss = nn.L1Loss(reduction='sum')
                        for semantic_score_idx in semantic_score_grid:
                            sem_con_loss += l1_loss(semantic_score_mean, semantic_score_idx)
                    elif self.sem_con_loss == 'kldiv':
                        prob_2_kldiv = lambda P, Q: torch.sum(P*(torch.log(P+1e-12) - torch.log(Q+1e-12)), dim=-1)
                        for semantic_score_idx in semantic_score_grid:
                            sem_con_loss += prob_2_kldiv(semantic_score_mean, semantic_score_idx)

                ### sv-con loss
                if self.sv_con_weight>0:
                    uncertainty_score_grid = uncertainty_score[grid_mask.squeeze()] 
                    # 投票选出最大概率语义
                    # 1.通过数量来投票
                    if self.sv_con_mode == 'num':
                        mode_value, mode_count = torch.mode(semantic_grid)
                        semantic_maxprob = mode_value.item()

                    # 2.通过累加概率分布来投票
                    if self.sv_con_mode == 'prob':
                        semantic_score_grid_sum = semantic_score_grid.sum(axis=0)
                        semantic_maxprob = semantic_score_grid_sum.argmax(axis=-1)

                    if self.sv_con_mode == 'uncertainty_prob' or self.sv_con_mode == 'uncertainty_exp_prob':
                        # 加入uncertrainty作为权重 加权平均
                        semantic_score_grid_0 = semantic_score_grid*uncertainty_score_grid.view(-1, 1)
                        semantic_score_grid_sum = semantic_score_grid_0.sum(axis=0)
                        semantic_maxprob = semantic_score_grid_sum.argmax(axis=-1)
                        # todo 是否考虑将1-prob作为一个权重，概率低说明需要多进行优化
                    
                    # 3.通过不确定性进行投票
                    if self.sv_con_mode == 'uncertainty' or self.sv_con_mode == 'uncertainty_exp':
                        semantic_list = torch.unique(semantic_grid)
                        maxscore = -0.1
                        for semantic_idx in semantic_list:
                            semantic_idx_mask = (semantic_grid==semantic_idx)
                            semantic_idx_score = uncertainty_score_grid[semantic_idx_mask].sum()
                            if semantic_idx_score > maxscore:
                                semantic_maxprob = semantic_idx
                                maxscore = semantic_idx_score
                    
                    # sv-con loss
                    prob=semantic_score_grid[:,semantic_maxprob]
                    if self.sv_con_loss == 'prob_mean':
                        sv_con_loss += 1-prob.mean()
                    elif self.sv_con_loss == 'prob':
                        sv_con_loss += len(prob)-prob.sum()
                    elif self.sv_con_loss == 'ce_loss_mean':
                        sv_con_loss += -torch.log(prob).mean()
                    elif self.sv_con_loss == 'ce_loss':
                        sv_con_loss += -torch.log(prob).sum()
                    elif self.sv_con_loss == 'ce_loss_weight':
                        if semantic_maxprob==0 or semantic_maxprob==1:
                            sv_con_loss += -0.5*torch.log(prob).sum() #todo 手动设置了weights
                        else:
                            sv_con_loss += -torch.log(prob).sum()
            
            # 叠加sv-con loss
            if self.sv_con_weight>0:
                if self.sv_con_loss.endswith('mean'):
                    sv_con_loss=sv_con_loss/len(grid_list)
                else:
                    sv_con_loss=sv_con_loss/((grid>0).sum())
            
                logs_summary.update({           
                        'Loss/loss_svcon':  sv_con_loss.detach().cpu(),
                    })
            # 叠加sem-con loss
            if self.sem_con_weight>0:
                if self.sem_con_loss.endswith('mean'):
                    sem_con_loss=sem_con_loss/len(grid_list)
                else:
                    sem_con_loss=sem_con_loss/((grid>0).sum())
            
                logs_summary.update({           
                        'Loss/loss_semcon':  sem_con_loss.detach().cpu(),
                    })    
        # joint loss
        joint_loss = 0.
        if self.semantic_weight>0 and self.joint_weight>0 and self.iter_step > self.joint_loss_milestone:
            semantic_class=semantic_fine.shape[1]
            WALL_SEMANTIC_ID=1
            FLOOR_SEMANTIC_ID=2

            semantic_score = F.softmax(semantic_fine, dim=-1)
            surface_normals_normalized = F.normalize(render_out['normal'], dim=-1).clamp(-1., 1.)
            surface_normals_normalized=surface_normals_normalized.unsqueeze(0)
            if semantic_class==3:
                _, wall_score, floor_score = semantic_score[...,:3].split(dim=-1, split_size=1)
            else:
                wall_score, floor_score = semantic_score[...,:2].split(dim=-1, split_size=1)
            
            # 选择输入语义
            wall_mask1= (true_semantic==WALL_SEMANTIC_ID)
            floor_mask1= (true_semantic==FLOOR_SEMANTIC_ID)
            # #选择neuris render出来的语义
            render_semantic=semantic_fine.argmax(axis=1)
            wall_mask2= (render_semantic==0)
            floor_mask2= (render_semantic==1)
            
            # #考虑多个mask叠加
            if self.joint_mode=='true_se':
                wall_mask = wall_mask1
                floor_mask = floor_mask1
            elif self.joint_mode=='render_se':
                wall_mask = wall_mask2
                floor_mask = floor_mask2
            elif self.joint_mode=='both_se':           
                wall_mask = wall_mask1.squeeze() | wall_mask2
                floor_mask = floor_mask1.squeeze() | floor_mask2

            if floor_mask.sum() > 0:
                floor_mask=(floor_mask.unsqueeze(0)).squeeze(-1) #changed
                floor_normals = surface_normals_normalized[floor_mask]
                floor_loss = (1 - floor_normals[..., 2]) # Eq.8
                floor_mask=floor_mask.squeeze() #changed
                joint_floor_loss = (floor_score[floor_mask][..., 0] * floor_loss).mean() # Eq.13
                joint_loss += joint_floor_loss
            
            if wall_mask.sum() > 0:
                wall_mask=(wall_mask.unsqueeze(0)).squeeze(-1)#changed
                wall_normals = surface_normals_normalized[wall_mask]
                wall_loss_vertical = wall_normals[..., 2].abs() #设置了一个vertical的wall
                cos = wall_normals[..., 0] * torch.cos(theta) + wall_normals[..., 1] * torch.sin(theta)
                wall_loss_horizontal = torch.min(cos.abs(), torch.min((1 - cos).abs(), (1 + cos).abs())) # Eq.9
                wall_loss = wall_loss_vertical + wall_loss_horizontal
                wall_mask=wall_mask.squeeze()#changed
                joint_wall_loss = (wall_score[wall_mask][..., 0] * wall_loss).mean() # Eq.13
                joint_loss += joint_wall_loss
            
            if floor_mask.sum() > 0 or wall_mask.sum() > 0:
                logs_summary.update({'Loss/joint_loss': joint_loss})

        # Eikonal loss
        gradient_error_loss = 0
        if self.igr_weight > 0:
            gradient_error_loss = gradient_error_fine
            logs_summary.update({           
                'Loss/loss_eik':    gradient_error_loss.detach().cpu(),
            })            
        
        # try to use "Towards Better Gradient Consistency for Neural Signed Distance Functions via Level Set Alignment"
        # gradient_consis_loss = render_out['consis_error']
        # logs_summary.update({           
        #         'Loss/loss_con_gra':    gradient_consis_loss.detach().cpu(),
        #     }) 

        # Smooth loss, optional
        surf_reg_loss = 0.0
        if self.smooth_weight > 0:
            depth = render_out['depth'].detach()
            pts = rays_o + depth * rays_d
            n_pts = pts + torch.randn_like(pts) * 1e-3  # WARN: Hard coding here
            surf_normal = sdf_network_fine.gradient(torch.cat([pts, n_pts], dim=0)).squeeze()
            surf_normal = surf_normal / torch.linalg.norm(surf_normal, dim=-1, ord=2, keepdim=True)

            surf_reg_loss_pts = (torch.linalg.norm(surf_normal[:batch_size, :] - surf_normal[batch_size:, :], ord=2, dim=-1, keepdim=True))
            # surf_reg_loss = (surf_reg_loss_pts*pixel_weight).mean()
            surf_reg_loss = surf_reg_loss_pts.mean()

        # normal loss
        normals_fine_loss, mask_keep_gt_normal = 0.0, torch.ones(batch_size)
        if self.normal_weight > 0 and normals_target is not None:
            normals_gt = normals_target # input_model['normals_gt'] #
            normals_fine = render_out['normal']
            
            normal_certain_weight = torch.ones(batch_size, 1).bool()
            if 'normal_certain_weight' in input_model:
                normal_certain_weight = input_model['normal_certain_weight']

            thres_clip_angle = -1 #
            normal_certain_weight = normal_certain_weight*mask_use_normals_target
            angular_error, mask_keep_gt_normal = TrainingUtils.get_angular_error(normals_fine, normals_gt, normal_certain_weight, thres_clip_angle)

            normals_fine_loss = angular_error
            logs_summary.update({
                'Loss/loss_normal_gt': normals_fine_loss
            })

        # depth loss, optional
        depths_fine_loss = 0.0
        if self.depth_weight > 0 and (pts_target is not None):
            pts = rays_o + depth * rays_d
            pts_error = (pts_target - pts) * mask_use_pts_target
            pts_error = torch.linalg.norm(pts_error, dim=-1, keepdims=True)

            depths_fine_loss = F.l1_loss(pts_error, torch.zeros_like(pts_error), reduction='sum') / (mask_use_pts_target.sum()+1e-6)
            logs_summary.update({
                'Loss/loss_depth': depths_fine_loss,
                'Log/num_depth_target_use': mask_use_pts_target.sum().detach().cpu()
            })

        plane_loss_all = 0
        if self.normal_consistency_weight > 0 and self.iter_step > self.plane_loss_milestone:
            num_planes = int(planes_gt.max().item())

            depth = render_out['depth']   # detach?
            pts = rays_o + depth * rays_d
            normals_fine = render_out['normal']

            # (1) normal consistency loss
            num_pixels_on_planes = 0
            num_pixels_on_subplanes = 0

            dominant_normal_planes = []
            normal_consistency_loss = torch.zeros(num_planes)
            loss_plane_offset = torch.zeros(num_planes)
            for i in range(int(num_planes)):
                idx_plane = i + 1
                mask_curr_plane = planes_gt.eq(idx_plane)
                if mask_curr_plane.float().max() < 1.0:
                    # this plane is not existent
                    continue
                consistency_loss_tmp, num_pixels_tmp, normal_mean_curr = get_normal_consistency_loss(normals_fine, mask_curr_plane)
                normal_consistency_loss[i] = consistency_loss_tmp
                num_pixels_on_planes += num_pixels_tmp

                # for Manhattan loss
                if i < 3:
                    # only use the 3 dominant planes
                    dominant_normal_planes.append(normal_mean_curr)
                
                # (2) plane-to-origin offset loss
                if self.plane_offset_weight > 0:
                    # normal_mean_curr_no_grad =  normal_mean_curr.detach()
                    plane_offset_loss_curr, num_pixels_subplanes_valid_curr = get_plane_offset_loss(pts, normal_mean_curr, mask_curr_plane, subplanes_gt)
                    loss_plane_offset[i] = plane_offset_loss_curr
                    num_pixels_on_subplanes += num_pixels_subplanes_valid_curr
            
            assert num_pixels_on_planes >= MIN_PIXELS_PLANE                   
            normal_consistency_loss = normal_consistency_loss.sum() / (num_pixels_on_planes+1e-6)
            loss_plane_offset = loss_plane_offset.sum() / (num_pixels_on_subplanes+1e-6)
            
            # (3) normal manhattan loss
            loss_normal_manhattan = 0
            if self.manhattan_constrain_weight > 0:
                loss_normal_manhattan = get_manhattan_normal_loss(dominant_normal_planes)

            plane_loss_all = normal_consistency_loss * self.normal_consistency_weight  + \
                                    loss_normal_manhattan * self.manhattan_constrain_weight + \
                                    loss_plane_offset * self.plane_offset_weight
        
        # planedepth_loss
        planedepth_loss = 0 
        if self.plane_depth_weight>0 and self.iter_step > self.planedepth_loss_milestone:
            depthplanes_gt = input_model['depthplanes_gt']
            num_depthplanes = int(depthplanes_gt.max().item())

            depth = render_out['depth']   # detach?
            pts = rays_o + depth * rays_d
            normals_fine = render_out['normal']

            for idx_depthplane in range(1, int(num_depthplanes)+1):
                mask_curr_depthplane = depthplanes_gt.eq(idx_depthplane)
                if mask_curr_depthplane.float().max() < 1.0:
                    # this plane is not existent
                    continue

                pts_curr_depthplane = pts * mask_curr_depthplane
                # 抽取mask，平均normal
                num_pixels_curr_depthplane = mask_curr_depthplane.sum()
                normals_fine_curr_depthplane = normals_fine * mask_curr_depthplane
                normal_mean_curr_depthplane = normals_fine_curr_depthplane.sum(dim=0) / num_pixels_curr_depthplane
                # 开始计算planedepth loss
                
                ## 计算每个点处的offset
                # offset = (pts_curr_depthplane*normals_fine_curr_depthplane).sum(dim=-1,keepdim=True) # pts \cdot normals
                if self.plane_depth_mode=='diff_offset':
                    offset = (pts*normal_mean_curr_depthplane).sum(dim=-1,keepdim=True) # pts \cdot normals_mean

                    offset_mean = (offset*mask_curr_depthplane).sum() / num_pixels_curr_depthplane
                    diff_offset = (offset - offset_mean) * mask_curr_depthplane

                    planedepth_idx_loss = F.mse_loss(diff_offset, torch.zeros_like(diff_offset), reduction='sum')
                
                planedepth_loss += planedepth_idx_loss
            planedepth_loss = planedepth_loss/((depthplanes_gt>0).sum())
            
            logs_summary.update({
                'Loss/loss_planedepth': planedepth_loss
            })
            
        loss = color_fine_loss * self.color_weight +\
                gradient_error_loss * self.igr_weight +\
                surf_reg_loss * self.smooth_weight +\
                semantic_fine_loss * self.semantic_weight +\
                joint_loss * self.joint_weight +\
                sv_con_loss * self.sv_con_weight +\
                sem_con_loss * self.sem_con_weight +\
                plane_loss_all +\
                planedepth_loss * self.plane_depth_weight +\
                background_loss * self.mask_weight +\
                normals_fine_loss * self.normal_weight * self.get_warm_up_ratio()  + \
                depths_fine_loss * self.depth_weight  #+ \
    

        logs_summary.update({
            'Loss/loss': loss.detach().cpu(),
            'Loss/loss_smooth': surf_reg_loss,
            'Loss/variance':    variance.mean().detach(),
            'Log/psnr':         psnr,
            'Log/ratio_warmup_loss':  self.get_warm_up_ratio()
        })
        
        if self.semantic_weight>0 and self.joint_weight>0:
            logs_summary.update({
                'Log/theta': theta.detach().cpu(),
                'Log/loss_joint': joint_loss
                })
        return loss, logs_summary, mask_keep_gt_normal