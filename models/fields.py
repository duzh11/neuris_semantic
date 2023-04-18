import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder, positional_encoding_c2f

class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 activation='softplus',
                 reverse_geoinit = False,
                 use_emb_c2f = False,
                 emb_c2f_start = 0.1,
                 emb_c2f_end = 0.5):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        self.multires = multires
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in, normalize=False)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch # input_dims*2*multires+input_dims
        logging.info(f'SDF input dimension: {dims[0]}') 

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        self.use_emb_c2f = use_emb_c2f
        if self.use_emb_c2f:
            self.emb_c2f_start = emb_c2f_start
            self.emb_c2f_end = emb_c2f_end
            logging.info(f"Use coarse-to-fine embedding (Level: {self.multires}): [{self.emb_c2f_start}, {self.emb_c2f_end}]")

        self.alpha_ratio = 0.0

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if reverse_geoinit:
                        logging.info(f"Geometry init: Indoor scene (reverse geometric init).")
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                    else:
                        logging.info(f"Geometry init: DTU scene (not reverse geometric init).")
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

        self.weigth_emb_c2f = None
        self.iter_step = 0
        self.end_iter = 3e5

    def forward(self, inputs):
        inputs = inputs * self.scale

        if self.use_emb_c2f and self.multires > 0:
            inputs, weigth_emb_c2f = positional_encoding_c2f(inputs, self.multires, emb_c2f=[self.emb_c2f_start, self.emb_c2f_end], alpha_ratio = (self.iter_step / self.end_iter))
            self.weigth_emb_c2f = weigth_emb_c2f
        elif self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        else:
            NotImplementedError

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class FixVarianceNetwork(nn.Module):
    def __init__(self, base):
        super(FixVarianceNetwork, self).__init__()
        self.base = base
        self.iter_step = 0

    def set_iter_step(self, iter_step):
        self.iter_step = iter_step

    def forward(self, x):
        return torch.ones([len(x), 1]) * np.exp(-self.iter_step / self.base)

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val=1.0, use_fixed_variance = False):
        super(SingleVarianceNetwork, self).__init__()
        if use_fixed_variance:
            logging.info(f'Use fixed variance: {init_val}')
            self.variance = torch.tensor([init_val])
        else:
            # Adds a parameter to the module.
            self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            d_feature,
            mode,
            d_in,
            d_out,
            d_hidden,
            n_layers,
            weight_norm=True,
            multires_view=0,
            squeeze_out=True,
    ):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
        logging.info(f'color input dimension: {dims[0]}') 

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


# Code from nerf-pytorch
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, d_in=3, d_in_view=3, multires=0, multires_view=0, output_ch=4, skips=[4], use_viewdirs=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in, normalize=False)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view, normalize=False)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            # 为什么alpha要加1
            return alpha + 1.0, rgb
        else:
            assert False

class DenseLayer(nn.Linear):
    def __init__(self, input_dim: int, out_dim: int, *args, activation=None, **kwargs):
        super().__init__(input_dim, out_dim, *args, **kwargs)
        self.MLP=nn.Linear(input_dim, out_dim)
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        out = self.MLP(x)
        out = self.activation(out)
        return out

class SemanticNetwork(nn.Module):
    def __init__(self,
                d_in=259,
                d_out=None,
                d_hidden=256,
                n_layers=4,
                skips=[],
                multires=-1,
                scale=1,
                activation='relu',
                weight_norm=True,
                semantic_mode=''):
        super().__init__()
                
        self.n_layers = n_layers
        self.skips = skips
        self.scale = scale
        self.d_hidden=d_hidden
        self.embed_fn = None        
        self.multires = multires
        embed_fn, input_ch = get_embedder(multires, input_dims=3, normalize=False)
        self.embed_fn = embed_fn
        d_in = d_in-3+input_ch
        logging.info(f'Semantic input dimension: {d_in}') 
        logging.info(f'Semantic output dimension: {d_out}') 
        
        fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(self.n_layers + 1):
            # decicde out_dim
            if l == self.n_layers:
                out_dim = d_out
            else:
                out_dim = self.d_hidden

            # decide in_dim
            if l == 0:
                in_dim = d_in
            elif l in self.skips:
                in_dim = d_in + self.d_hidden
            else:
                in_dim = self.d_hidden
            
            if l != self.n_layers:
                layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                if semantic_mode=='sigmoid':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
                    logging.info(f'Semantic_mode: {semantic_mode}')
                elif semantic_mode=='softmax':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Softmax(dim=-1))
                    logging.info(f'Semantic_mode: {semantic_mode}')
                else:
                    layer = nn.Linear(in_dim, out_dim)
                    logging.info(f'Semantic_mode: {semantic_mode}')
            
            if weight_norm:
                layer = nn.utils.weight_norm(layer)
            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)
    
    def forward(
        self, 
        x: torch.Tensor, 
        geometry_feature: torch.Tensor):
        # calculate semantic field
        x = self.embed_fn(x)
        semantic_input = torch.cat([x, geometry_feature], dim=-1)
        
        h = semantic_input
        for i in range(self.n_layers+1):
            if i in self.skips:
                h = torch.cat([h, semantic_input], dim=-1)
            h = self.layers[i](h)
        return h

