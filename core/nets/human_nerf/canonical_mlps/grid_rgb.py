import torch
import torch.nn as nn

from core.utils.network_util import initseq
from core.nets.human_nerf.gridencoder import GridEncoder

class CanonicalMLP(nn.Module):
    def __init__(self, mlp_depth=8, mlp_width=256, 
                 input_ch=3, skips=None, bound=1, geo_feat_dim=63,
                 **_):
        super(CanonicalMLP, self).__init__()

        if skips is None:
            skips = [4]

        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.input_ch = input_ch
        self.bound = bound

        self.encoder = GridEncoder(input_dim=3, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048*bound, gridtype='hash', align_corners=False)
        #self.dir_encoder = SHEncoder(input_dim=3, degree=4)
        self.neural_point_dim = 32 + 32#self.encoder.output_dim

        pts_block_mlps = [nn.Linear(32, self.mlp_width), nn.ReLU(inplace=True)]


        layers_to_cat_input = []
        for i in range(mlp_depth-1):
            if i in skips:
                layers_to_cat_input.append(len(pts_block_mlps))
                pts_block_mlps += [nn.Linear(mlp_width + input_ch, mlp_width), 
                                   nn.ReLU(inplace=True)]
            else:
                pts_block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU(inplace=True)]

        self.pts_linears = nn.ModuleList(pts_block_mlps)
        initseq(self.pts_linears)

        self.geo_linear = nn.Sequential(nn.Linear(mlp_width, 64 + 1))
        initseq(self.geo_linear)

        ################ color
        pts_block_mlps = [nn.Linear(64, mlp_width), nn.ReLU(inplace=True)]

        layers_to_cat_input = []
        for i in range(mlp_depth-1):
            if i in skips:
                layers_to_cat_input.append(len(pts_block_mlps))
                pts_block_mlps += [nn.Linear(mlp_width + input_ch, mlp_width), 
                                   nn.ReLU(inplace=True)]
            else:
                pts_block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU(inplace=True)]

        self.rgb_linears = nn.ModuleList(pts_block_mlps)
        initseq(self.rgb_linears)

        self.output_linear = nn.Sequential(nn.Linear(mlp_width, 4)) # rgb + uncertainty
        initseq(self.output_linear)
    
    def forward(self, xyz, **_):
        h = self.encoder(xyz, bound=None)
        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

        h = self.geo_linear(h)
        sigma = h[..., [0]]
        h = h[..., 1:]
        for i, _ in enumerate(self.rgb_linears):
            h = self.rgb_linears[i](h)

        h = self.output_linear(h)

        return torch.cat((h[...,:3], sigma, h[...,3:]), dim=-1)
