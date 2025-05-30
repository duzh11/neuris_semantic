import torch
import torch.nn.functional as F
import numpy as np
import skimage
import mcubes
import trimesh
import logging
from tqdm import tqdm

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    
    # neuris use mcbues，but I need normals
    # vertices, triangles = mcubes.marching_cubes(u, threshold)
    vertices, faces, normals, values = skimage.measure.marching_cubes(u, level=threshold, gradient_direction='ascent')
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    
    return vertices, faces, normals, u


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network_fine,
                 variance_network_fine,
                 color_network_fine,
                 semantic_network_fine,
                 semantic_class,
                 n_samples,
                 n_importance,
                 n_outside,
                 perturb,
                 alpha_type='div',
                 semantic_mode=None,
                 stop_ce_grad=False):
        self.nerf = nerf
        self.sdf_network_fine = sdf_network_fine
        self.variance_network_fine = variance_network_fine
        self.color_network_fine = color_network_fine
        self.semantic_network_fine = semantic_network_fine
        self.semantic_class=semantic_class
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.perturb = perturb
        self.alpha_type = alpha_type
        self.radius = 1.0

        if semantic_network_fine:
            self.semantic_mode = semantic_mode
            logging.info(f'semantic mode: {self.semantic_mode}')
            self.stop_ce_grad = stop_ce_grad
            logging.info(f'stop_ce_grad: {self.stop_ce_grad}')
        else:
            self.semantic_mode = None
            logging.info(f'semantic mode: {self.semantic_mode}')
            self.stop_ce_grad = False
            logging.info(f'stop_ce_grad: {self.stop_ce_grad}')
        

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        batch_size, n_samples = z_vals.shape

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)

        mid_z_vals = z_vals + dists * 0.5
        mid_dists = mid_z_vals[..., 1:] - mid_z_vals[..., :-1]
        mid_dists = torch.cat([mid_dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        if self.n_outside > 0:
            dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
            pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]  # n_rays, n_samples
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }


    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_variance):
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        dot_val = None
        if self.alpha_type == 'uniform':
            dot_val = torch.ones([batch_size, n_samples - 1]) * -1.0
        else:
            dot_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
            prev_dot_val = torch.cat([torch.zeros([batch_size, 1]), dot_val[:, :-1]], dim=-1)
            dot_val = torch.stack([prev_dot_val, dot_val], dim=-1)
            dot_val, _ = torch.min(dot_val, dim=-1, keepdim=False)
            dot_val = dot_val.clip(-10.0, 0.0) * inside_sphere
        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - dot_val * dist * 0.5
        next_esti_sdf = mid_sdf + dot_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_variance)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_variance)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        new_sdf = self.sdf_network_fine.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        sdf = torch.cat([sdf, new_sdf], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
        xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
        index = index.reshape(-1)
        sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)
        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    variance_network,
                    color_network,
                    semantic_network,
                    semantic_class=None,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    alpha_inter_ratio=0.0,
                    validate_flag=False):
        logs_summary = {}

        batch_size, n_samples = z_vals.shape
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)

        mid_z_vals = z_vals + dists * 0.5
        mid_dists = mid_z_vals[..., 1:] - mid_z_vals[..., :-1]

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        # sdf-network
        sdf_nn_output = sdf_network(pts) #[sdf, feature]
        sdf = sdf_nn_output[:, :1] #batch_size*n_samples, 1
        feature_vector = sdf_nn_output[:, 1:] #batch_size*n_samples, 256
        # color-network
        gradients = sdf_network.gradient(pts).squeeze() #法向量
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3) #batch_szie, n_samples, 3
        
        #semantic-network
        sampled_semantic=torch.zeros([batch_size, n_samples, semantic_class])
        sampled_label = torch.zeros([batch_size, n_samples])
        if semantic_network:
            # stop semantic gradients to geometry feature
            if self.stop_ce_grad:
                feature_vector_new=feature_vector.detach()
                sampled_semantic = semantic_network(pts, feature_vector_new).reshape(batch_size, n_samples, semantic_class) 
            else:
                sampled_semantic = semantic_network(pts, feature_vector).reshape(batch_size, n_samples, semantic_class) 
            sampled_label = sampled_semantic.argmax(axis=-1)
                
        # variance-network
        inv_variance = variance_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_variance = inv_variance.expand(batch_size * n_samples, 1) #batch_szie*n_samples, 1
        
        true_dot_val = (dirs * gradients).sum(-1, keepdim=True) # normal*view_dir
        # |cos(theta)|
        # TODO alpha_inter_ratio有什么用？
        iter_cos = -(F.relu(-true_dot_val * 0.5 + 0.5) * (1.0 - alpha_inter_ratio) + F.relu(-true_dot_val) * alpha_inter_ratio) # always non-positive
        # sdf +/-|cos(theta)|*dists/2
        true_estimate_sdf_half_next = sdf + iter_cos.clip(-10.0, 10.0) * dists.reshape(-1, 1) * 0.5
        true_estimate_sdf_half_prev = sdf - iter_cos.clip(-10.0, 10.0) * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(true_estimate_sdf_half_prev * inv_variance)
        next_cdf = torch.sigmoid(true_estimate_sdf_half_next * inv_variance)

        p = prev_cdf - next_cdf
        c = prev_cdf
        # 参考neuris中公式(13)
        if self.alpha_type == 'div':
            alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        elif self.alpha_type == 'uniform':
            uniform_estimate_sdf_half_next = sdf - dists.reshape(-1, 1) * 0.5
            uniform_estimate_sdf_half_prev = sdf + dists.reshape(-1, 1) * 0.5
            uniform_prev_cdf = torch.sigmoid(uniform_estimate_sdf_half_prev * inv_variance)
            uniform_next_cdf = torch.sigmoid(uniform_estimate_sdf_half_next * inv_variance)
            uniform_alpha = F.relu(
                (uniform_prev_cdf - uniform_next_cdf + 1e-5) / (uniform_prev_cdf + 1e-5)).reshape(
                batch_size, n_samples).clip(0.0, 1.0)
            alpha = uniform_alpha
        else:
            assert False

        pts_radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_radius < 1.0*self.radius).float().detach()
        relax_inside_sphere = (pts_radius < 1.2*self.radius).float().detach()

        if background_alpha is not None:   # render with background
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] + background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)
        
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]  # n_rays, n_samples
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights_sum)
        
        # softmax策略
        # softmax_a: 对logits进行softmax，在积分后log，然后送入ce_loss
        # softmax_b: 对logits进行softmax，在积分后log，然后送入nll_loss
        # softmax_c: 对logits进行softmax，在积分前log，然后送入nll_loss
        if semantic_network:
            if self.semantic_mode=='softmax_a' or self.semantic_mode=='softmax_b':
                sampled_semantic_0 = F.softmax(sampled_semantic, dim=-1)
            elif self.semantic_mode=='softmax_c':
                sampled_semantic_0 = F.log_softmax(sampled_semantic, dim=-1)
            else:
                sampled_semantic_0 = sampled_semantic

        # stop semantic grad to weights
        if semantic_network:
            if self.stop_ce_grad:
                weights_new=weights.detach()
                semantic = (sampled_semantic_0 * weights_new[:, :, None]).sum(dim=1)
            else:
                semantic = (sampled_semantic_0 * weights[:, :, None]).sum(dim=1)     
        else:
            semantic = (sampled_semantic * weights[:, :, None]).sum(dim=1)    
        
        # softmax策略
        if semantic_network:
            if self.semantic_mode=='softmax_a' or self.semantic_mode=='softmax_b':
                semantic = semantic / (semantic.sum(-1).unsqueeze(-1) + 1e-8)
                semantic = torch.log(semantic + 1e-8)

        logits_2_uncertainty = lambda x: torch.sum(-F.log_softmax(x, dim=-1)*F.softmax(x, dim=-1), dim=-1, keepdim=True)
        sem_uncertainty = logits_2_uncertainty(semantic)
        
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        # try to use "Towards Better Gradient Consistency for Neural Signed Distance Functions via Level Set Alignment"
        # gradient_norm = F.normalize(gradients, dim=-1)
        # pts_moved = pts + gradient_norm * sdf

        # # sdf_moved = sdf_network(pts_moved)[:, :1] # 期望它落在零值面上
        # gradient_moved = sdf_network.gradient(pts_moved).squeeze()
        # gradient_moved_norm = F.normalize(gradient_moved, dim=-1)
        # consis_constraint = 1 - F.cosine_similarity(gradient_moved_norm, gradient_norm, dim=-1)
        # weight_moved = torch.exp(-10 * torch.abs(sdf)).reshape(-1,consis_constraint.shape[-1]) 
        # consis_constraint = consis_constraint * weight_moved      
        # consis_constraint = consis_constraint.reshape(batch_size, n_samples)
        # consis_gradients = (relax_inside_sphere * consis_constraint).sum() / (relax_inside_sphere.sum() + 1e-5)

        variance = (1.0 /inv_variance).mean(dim=-1, keepdim=True)
        assert (torch.isinf(variance).any() == False)
        assert (torch.isnan(variance).any() == False)

        depth = (mid_z_vals * weights[:, :n_samples]).sum(dim=1, keepdim=True)
        depth_varaince = ((mid_z_vals - depth) ** 2 * weights[:, :n_samples]).sum(dim=-1, keepdim=True)

        normal = (gradients.reshape(batch_size, n_samples, 3) * weights[:, :n_samples].reshape(batch_size, n_samples, 1)).sum(dim=1)

        # visualize embedding weights
        if sdf_network.weigth_emb_c2f != None:
            # print(sdf_network.weigth_emb_c2f)
            for id_w in range(len(sdf_network.weigth_emb_c2f)):
                logs_summary[f'weigth_emb_c2f/level{id_w+1}'] = sdf_network.weigth_emb_c2f[id_w].detach()
        
        with torch.no_grad():
            if inv_variance[0, 0] > 800:
                # logging.info(f"Use default inv-variance to calculate peak value")
                depth_peak = depth.clone()
                normal_peak = normal.clone()
                color_peak = color.clone()
                semantic_peak = semantic.clone()
                sem_uncertainty_peak = logits_2_uncertainty(semantic_peak)
                point_peak = rays_o + rays_d*depth
            else:
                # Reset a large inv-variance to get better peak value
                inv_variance2 = torch.tensor([800])
                inv_variance2 = inv_variance2.expand(batch_size * n_samples, 1)
                prev_cdf2 = torch.sigmoid(true_estimate_sdf_half_prev * inv_variance2)
                next_cdf2 = torch.sigmoid(true_estimate_sdf_half_next * inv_variance2)

                p2 = prev_cdf2 - next_cdf2
                c2 = prev_cdf2
                alpha2 = ((p2 + 1e-5) / (c2 + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
                weights2 = alpha2 * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha2 + 1e-7], -1), -1)[:, :-1]  # n_rays, n_samples
        
                depth_peak = (mid_z_vals * weights2[:, :n_samples]).sum(dim=1, keepdim=True)
                normal_peak = (gradients.reshape(batch_size, n_samples, 3) * weights2[:, :n_samples].reshape(batch_size, n_samples, 1)).sum(dim=1)
                color_peak = (sampled_color[:, :n_samples] * weights2[:, :n_samples, None]).sum(dim=1)
                semantic_peak = (sampled_semantic[:, :n_samples] * weights2[:, :n_samples, None]).sum(dim=1)
                sem_uncertainty_peak = logits_2_uncertainty(semantic_peak)
                point_peak = rays_o + rays_d*depth_peak

        render_out = {
            'variance': variance,
            'variance_inv_pts': inv_variance.reshape(batch_size, n_samples),
            'depth': depth,
            'depth_variance': depth_varaince,
            'normal': normal,
            'color_fine': color,
            'semantic_fine': semantic,
            'sampled_prob': F.softmax(sampled_semantic, dim=-1),
            'sampled_label': sampled_label,
            'sem_uncertainty_fine': sem_uncertainty,
            'cdf_fine': c.reshape(batch_size, n_samples),
            'sdf': sdf,
            'dists': dists,
            'mid_z_vals': mid_z_vals,
            'relax_inside_sphere': relax_inside_sphere,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            'gradient_error_fine': gradient_error,
            'weights': weights,
            'weight_sum': weights.sum(dim=-1, keepdim=True),
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'inside_sphere': inside_sphere,
            'depth_peak': depth_peak,
            'normal_peak': normal_peak,
            'color_peak': color_peak,
            'semantic_peak': semantic_peak,
            'sem_uncertainty_peak': sem_uncertainty_peak,
            'point_peak': point_peak
        }

        # try to use "Towards Better Gradient Consistency for Neural Signed Distance Functions via Level Set Alignment"
        # render_out.update({'consis_error': consis_gradients})

        if validate_flag:
            render_out.update(
                {'mid_z_vals': mid_z_vals.detach(),
                 'alpha': alpha.detach(),
                 'weights': weights.detach(),
                 'sdf': (sdf.detach()).reshape(batch_size, n_samples),
                 'sampled_semantic': sampled_semantic.detach()}
            )
        return render_out, logs_summary

    def render(self, 
               rays_o, 
               rays_d, 
               near, 
               far, 
               perturb_overwrite=-1, 
               background_rgb=None, 
               alpha_inter_ratio=0.0, 
               validate_flag=False):
        batch_size = len(rays_o)
        sphere_diameter = torch.abs(far-near).mean()
        sample_dist = sphere_diameter / self.n_samples
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            # get intervals between samples
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network_fine.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                n_steps = 4
                for i in range(n_steps):
                    new_z_vals = self.up_sample(rays_o, rays_d, z_vals, sdf, self.n_importance // n_steps, 64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf)

            n_samples = self.n_samples + self.n_importance

        # Background
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render
        ret_fine, logs_summary = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network_fine,
                                    self.variance_network_fine,
                                    self.color_network_fine,
                                    self.semantic_network_fine,
                                    semantic_class=self.semantic_class,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    alpha_inter_ratio=alpha_inter_ratio,
                                    validate_flag=validate_flag)
        return ret_fine, logs_summary

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        ret = extract_geometry(bound_min, bound_max, resolution, threshold, lambda pts: -self.sdf_network_fine.sdf(pts))
        return ret
    
    def extract_surface_semantic(self, vertices, chunk=512):
        vertices_labels = []
        B = vertices.shape[0]
        for i in tqdm(range(0, B, chunk), desc='surface rendering'):
            vertices_bs = vertices[i:i+chunk, :]
            geometry_feature = self.sdf_network_fine.geometry_feature(vertices_bs)
            semantic_bs = self.semantic_network_fine.semantic(vertices_bs, geometry_feature.detach())
            vertices_labels.append(semantic_bs.argmax(axis=1).cpu().numpy()+1)
        vertices_labels = np.concatenate(vertices_labels, axis=0)
        return vertices_labels
    
    def extract_volume_semantic(self, 
                                vertices, 
                                normal=None, 
                                volume_near = 0.05,
                                volume_far = 1.0,
                                chunk=512):
        vertices_labels = []
        B = vertices.shape[0]
        for i in tqdm(range(0, B, chunk), desc='volume rendering of virtual view'):
            vertices_bs = vertices[i:i+chunk, :]
            
            if normal==None:
                gradients = self.sdf_network_fine.gradient(vertices_bs)
                normal_bs = F.normalize(gradients.detach(), dim=2).squeeze()
            else:
                normal_bs = normal[i:i+chunk, :]

            # 考虑normal的方向问题
            consider_normal_dir=True
            if consider_normal_dir:
                sign = ( torch.sign( torch.sum(normal_bs*vertices_bs, dim=1) ) ).unsqueeze(1)
            else:
                sign = -1
            rays_d = normal_bs*sign
            rays_o = vertices_bs - volume_near*rays_d
            near, far = torch.zeros(len(rays_o), 1), volume_far * torch.ones(len(rays_o), 1)
            render_out, _ = self.render(rays_o, 
                                        rays_d, 
                                        near, 
                                        far, 
                                        alpha_inter_ratio=1.0)
            label = render_out['semantic_fine']
            vertices_labels.append(label.argmax(axis=1).cpu().numpy()+1)   
        vertices_labels = np.concatenate(vertices_labels, axis=0)    
        return vertices_labels
    
class NeRFRenderer:
    def __init__(self,
                 nerf_coarse,
                 nerf_fine,
                 nerf_outside,
                 n_samples,
                 n_importance,
                 n_outside,
                 perturb):
        self.nerf_coarse = nerf_coarse
        self.nerf_fine = nerf_fine
        self.nerf_outside = nerf_outside
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.perturb = perturb

    def render_core_coarse(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        batch_size, n_samples = z_vals.shape

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)

        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        if self.n_outside > 0:
            dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
            pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - torch.exp(-F.relu(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]  # n_rays, n_samples
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    nerf,
                    fine=False,
                    background_rgb=None,
                    background_alpha=None,
                    background_sampled_color=None):

        batch_size, n_samples = z_vals.shape
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)

        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3

        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        alpha = 1.0 - torch.exp(-F.relu(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)

        inside_sphere = None
        if background_alpha is not None:  # render without mask
            pts_radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
            inside_sphere = (torch.linalg.norm(pts, ord=2, dim=-1).reshape(batch_size, n_samples) < 1.0).float()
            # alpha = alpha * inside_sphere
            alpha = torch.cat([alpha, background_alpha], dim=-1)
            sampled_color = torch.cat([sampled_color, background_sampled_color], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]  # n_rays, n_samples
        color = (weights[:, :, None] * sampled_color.reshape(batch_size, n_samples + self.n_outside, 3)).sum(dim=1)

        return {
            'color': color,
            'weights': weights,
            'depth': (mid_z_vals * weights[:, :n_samples]).sum(dim=-1, keepdim=True)
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None):

        sample_dist = ((far - near) / self.n_samples).mean().item()
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]
        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals_outside.shape)
                z_vals_outside = lower + (upper - lower) * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        color_coarse = None

        background_alpha = None
        background_sampled_color = None
        background_color = torch.zeros([1, 3])

        ret_coarse = {
            'color': None,
            'weights': None,
        }

        # NeRF++
        if self.n_outside > 0:
            # z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            # z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            z_vals_feed = z_vals_outside
            ret_outside = self.render_core_coarse(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf_outside,
                                                  background_rgb=None)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        if self.n_importance > 0:
            ret_coarse = self.render_core(rays_o,
                                          rays_d,
                                          z_vals,
                                          sample_dist,
                                          self.nerf_coarse,
                                          fine=False,
                                          background_rgb=background_rgb,
                                          background_alpha=background_alpha,
                                          background_sampled_color=background_sampled_color)

            weights = ret_coarse['weights']
            # importance sampling
            z_samples = sample_pdf(z_vals, weights[..., :-1 + self.n_samples],
                                   self.n_importance, det=True).detach()
            z_vals = torch.cat([z_vals, z_samples], dim=-1)
            z_vals, _ = torch.sort(z_vals, dim=-1)
            z_vals = z_vals.detach()

            n_samples = self.n_samples + self.n_importance

        # ----------------------------------- fine --------------------------------------------
        # render again
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.nerf_fine,
                                    fine=True,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color)

        return {
            'color_coarse': ret_coarse['color'],
            'color_fine': ret_fine['color'],
            'weight_sum': ret_fine['weights'].sum(dim=-1, keepdim=True),
            'depth': ret_fine['depth']
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=25):
        ret = extract_geometry(bound_min, bound_max, resolution, threshold, lambda pts: self.nerf_fine(pts, torch.zeros_like(pts))[0])
        return ret