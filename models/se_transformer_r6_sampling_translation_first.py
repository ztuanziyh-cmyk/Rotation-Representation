import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import kornia
from torch.autograd import Function
import e3nn.o3 as o3
from e3nn.math import soft_one_hot_linspace
from .layers.self_attention_r6 import SelfAttentionModule

print(torch.__version__)  
print(torch.version.cuda)  
from models.layers.self_attention_r6 import r6_to_matrix

def set_deterministic():
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)


###############################################################################
# SphericalSteerablePatchConv3D
###############################################################################
class SphericalSteerablePatchConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: tuple = (4,4,4),
        max_radius: float = 1.0,
        num_radial_basis: int = 4,
        lmax: int = 2    
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.patch_size   = patch_size
        self.max_radius   = max_radius
        self.num_radial_basis = num_radial_basis

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=lmax)
        self.sh_dim = self.irreps_sh.dim

        self.weight_numel = in_channels * out_channels * self.sh_dim

        hidden_dim = 16
        self.mlp = nn.Sequential(
            nn.Linear(num_radial_basis, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.weight_numel)
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        B, C, Pd, Ph, Pw = patch.shape
        device = patch.device

        z_vals = torch.linspace(-1, 1, Pd, device=device)
        y_vals = torch.linspace(-1, 1, Ph, device=device)
        x_vals = torch.linspace(-1, 1, Pw, device=device)
        zz, yy, xx = torch.meshgrid(z_vals, y_vals, x_vals, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        n_vox = coords.shape[0]

        sh = o3.spherical_harmonics(
            self.irreps_sh,
            coords,
            normalize=True,
            normalization='component'
        )

        dist = coords.norm(dim=1)
        r_embed = soft_one_hot_linspace(
            dist,
            start=0.0, end=self.max_radius,
            number=self.num_radial_basis,
            basis='smooth_finite', cutoff=True
        ) * (self.num_radial_basis**0.5)

        w_raw = self.mlp(r_embed).view(
            n_vox, self.in_channels, self.out_channels, self.sh_dim
        )

        patch_flat = patch.view(B, C, n_vox)

        out = torch.einsum('b i n, n i o l, n l -> b o',
                           patch_flat, w_raw, sh)
        return out


###############################################################################
# PatchLogitsSteerableNet3D 
###############################################################################
class PatchLogitsSteerableNet3D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hidden_channels=8,
        patch_size=(4,4,4),
        max_radius=1.0,
        sampling_ratio=1.0
    ):
        super().__init__()
        self.patch_conv1 = SphericalSteerablePatchConv3D(
            in_channels=in_channels,
            out_channels=hidden_channels,
            patch_size=patch_size,
            max_radius=max_radius
        )
        
        self.hidden_channels = hidden_channels
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU()
        )
        self.fc_final = nn.Linear(hidden_channels, 1)

        self.sampling_ratio = sampling_ratio
        self.in_channels = in_channels
        self.patch_size = patch_size

    def forward(self, x_patches: torch.Tensor) -> torch.Tensor:
        device = x_patches.device
        N, C, Pd, Ph, Pw = x_patches.shape
        if self.sampling_ratio >= 1.0:
            feat = self.patch_conv1(x_patches)  
            feat = self.mlp(feat)
            logits = self.fc_final(feat).squeeze(-1)  
            return logits
        else:
            chunk_size = max(1, int(N * self.sampling_ratio))
            logits_out = []
            start = 0
            while start < N:
                end = min(start+chunk_size, N)
                subset = x_patches[start:end]  
                subset_feat = self.patch_conv1(subset)
                subset_feat = self.mlp(subset_feat)
                subset_logit = self.fc_final(subset_feat).squeeze(-1)
                logits_out.append(subset_logit)
                start = end
            logits = torch.cat(logits_out, dim=0)  
            return logits

###############################################################################
# LPSDown3D / LPSUp3D
###############################################################################
class LPSDown3D(nn.Module):
    def __init__(
        self,
        patch_size=(4,4,4),
        in_channels=1,
        hidden_channels=8,
        temperature=1.0,
        use_gumbel=True,
        max_radius=1.0,
        sampling_ratio=1.0
    ):
        super().__init__()
        self.patch_size   = patch_size
        self.in_channels  = in_channels
        self.hidden_channels = hidden_channels
        self.temperature  = temperature
        self.use_gumbel   = use_gumbel

        self.logit_net = PatchLogitsSteerableNet3D(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            patch_size=patch_size,
            max_radius=max_radius,
            sampling_ratio=sampling_ratio
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        pd, ph, pw = self.patch_size
        gd, gh, gw = D//pd, H//ph, W//pw
        total_patches = gd*gh*gw

        x_resh = x.view(B,C, gd,pd, gh,ph, gw,pw)
        x_resh = x_resh.permute(0,1,2,4,6,3,5,7).contiguous()
        x_patches = x_resh.view(B*total_patches, C, pd, ph, pw)

        logits_all = self.logit_net(x_patches)
        logits_all = logits_all.view(B, total_patches)

        if self.training and self.use_gumbel:
            noise = -torch.log(-torch.log(torch.rand_like(logits_all)))
            y = (logits_all + noise) / self.temperature
            patch_probs = torch.softmax(y, dim=1)
        else:
            idx_sel = torch.argmax(logits_all, dim=1)  # shape (B,)
            patch_probs = torch.zeros_like(logits_all)
            patch_probs.scatter_(1, idx_sel.unsqueeze(1), 1.0)

        idx_sel = torch.argmax(patch_probs, dim=1)
        pz = idx_sel // (gh*gw)
        remain = idx_sel % (gh*gw)
        py = remain // gw
        px = remain % gw

        lz, ly, lx = pd-1, ph-1, pw-1
        x_pad = F.pad(x, (0,lx,0,ly,0,lz), mode='circular')

        device = x.device
        theta = torch.zeros(B,3,4, device=device)
        theta[:,0,0] = 1
        theta[:,1,1] = 1
        theta[:,2,2] = 1
        theta[:,0,3] = px * 2/(W+lx)
        theta[:,1,3] = py * 2/(H+ly)
        theta[:,2,3] = pz * 2/(D+lz)

        grid = F.affine_grid(theta, x_pad.size(), align_corners=False)
        x_shifted = F.grid_sample(x_pad, grid, mode='nearest', align_corners=False)
        x_shifted = x_shifted[:,:,:D,:H,:W]
        return x_shifted, patch_probs

class LPSUp3D(nn.Module):
    def __init__(self, patch_size=(4,4,4)):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x_shifted, patch_probs):
        B, C, D, H, W = x_shifted.shape
        pd, ph, pw = self.patch_size
        gd, gh, gw = D//pd, H//ph, W//pw

        idx_sel = torch.argmax(patch_probs, dim=1)
        pz = idx_sel // (gh*gw)
        remain = idx_sel % (gh*gw)
        py = remain // gw
        px = remain % gw

        lz, ly, lx = pd-1, ph-1, pw-1
        x_pad = F.pad(x_shifted, (0,lx,0,ly,0,lz), mode='circular')

        device = x_shifted.device
        theta = torch.zeros(B,3,4, device=device)
        theta[:,0,0] = 1
        theta[:,1,1] = 1
        theta[:,2,2] = 1

        theta[:,0,3] = - px * 2/(W+lx)
        theta[:,1,3] = - py * 2/(H+ly)
        theta[:,2,3] = - pz * 2/(D+lz)

        grid = F.affine_grid(theta, x_pad.size(), align_corners=False)
        x_unshifted = F.grid_sample(x_pad, grid, mode='nearest', align_corners=False)
        x_unshifted = x_unshifted[:,:,:D,:H,:W]
        return x_unshifted

###############################################################################
# LPS3DNetE3NNNoScatter
###############################################################################
class LPS3DNetE3NNNoScatter(nn.Module):
    def __init__(
        self,
        patch_size=(4,4,4),
        in_channels=1,
        hidden_dim=16,
        max_radius=1.0,
        sampling_ratio=1.0
    ):
        super().__init__()
        self.lps_down = LPSDown3D(
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_channels=8,
            temperature=1.0,
            use_gumbel=True,
            max_radius=max_radius,
            sampling_ratio=sampling_ratio
        )
        self.conv = nn.Conv3d(in_channels, hidden_dim, 3, padding=1, padding_mode='circular')
        self.act  = nn.GELU()
        self.lps_up = LPSUp3D(patch_size=patch_size)

    def forward(self, x):
        x_shifted, patch_probs = self.lps_down(x)
        y = self.conv(x_shifted)
        y = self.act(y)
        out = self.lps_up(y, patch_probs)
        return out


###############################################################################
# Polyphase Patch Embedding
###############################################################################
class PolyPatchEmbed3D(nn.Module):
    def __init__(self, patch_size, in_chans, out_chans, norm_layer=None):
        super().__init__()
        self.poly_order_module = LPSDown3D(patch_size=patch_size)
        self.proj = nn.Conv3d(
            in_chans, out_chans, kernel_size=patch_size, stride=patch_size, padding=0, padding_mode="circular"
        )
        self.norm = norm_layer(out_chans) if norm_layer else nn.Identity()

    def forward(self, x):
        x_shifted, _ = self.poly_order_module(x)  
        x = self.proj(x_shifted)                 
        x = self.norm(x)
        return x

class PosConv3D(nn.Module):
    def __init__(self, in_chans, out_chans, patch_size=(4, 4, 4), use_polyphase=True):
        super(PosConv3D, self).__init__()
        self.use_polyphase = use_polyphase
        if self.use_polyphase:
            self.poly_order_module = LPSDown3D(patch_size=patch_size)

        self.conv = nn.Conv3d(
            in_chans, out_chans, kernel_size=3, padding=1, padding_mode='circular', groups=in_chans
        )

    def forward(self, x):
        if self.use_polyphase:
            x_shifted, _ = self.poly_order_module(x)
            x = x_shifted
        x = self.conv(x)
        return x

###############################################################################
# FeatureComparator 
###############################################################################
class FeatureComparator(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim  = hidden_dim

        self.diff_conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU()
        )

        self.merge = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1, padding=0)

    def forward(self, target_feats, input_feats):
        diff = target_feats - input_feats
        out = self.diff_conv(diff)
        out = self.merge(out)
        return out

###############################################################################
# PolyphaseFeatureExtractor 
###############################################################################
class PolyphaseFeatureExtractor(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.phase_processor = CNNPolyphaseProcessor(
            in_channels=in_channels,
            out_channels=embed_dim,
            patch_size=patch_size,
            spherical_out=8,
            max_radius=1.0
        )

        self.post_cnn = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, 3, 
                      stride=1, padding=1, padding_mode='circular'),
            nn.ReLU()
        )

    def forward(self, x):
        feats = self.phase_processor(x)  

        feats = self.post_cnn(feats)     
        return feats


###############################################################################
# ShiftEquivariantPositionalEncoder 
###############################################################################
class ShiftEquivariantPositionalEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = nn.Conv3d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1, padding_mode='circular', groups=hidden_dim
        )
    def forward(self, x):
        return self.conv(x)

###############################################################################
# CNNPolyphaseProcessor 
###############################################################################
class CNNPolyphaseProcessor(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=(4,4,4), 
                 spherical_out=8, max_radius=1.0):
        super().__init__()
        self.patch_size = patch_size
        pd, ph, pw = patch_size

        self.poly_anchor = LPSDown3D(
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_channels=8,
            temperature=1.0,
            use_gumbel=True,
            max_radius=max_radius,
            sampling_ratio=1.0
        )

        self.num_phases = pd * ph * pw
        self.spherical_out = spherical_out

        self.phase_convs = nn.ModuleList([
            SphericalSteerablePatchConv3D(
                in_channels=in_channels,
                out_channels=spherical_out,
                patch_size=(3,3,3), 
                max_radius=max_radius,
                num_radial_basis=4
            )
            for _ in range(self.num_phases)
        ])

        self.fuse = nn.Sequential(
            nn.Conv3d(self.spherical_out * self.num_phases, out_channels, kernel_size=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        x_shifted, _ = self.poly_anchor(x)  

        phases = self.decompose_phases(x_shifted)  

        out_per_phase = []
        for idx, (p, sphconv) in enumerate(zip(phases, self.phase_convs)):
            B, C, Dp, Hp, Wp = p.shape 
            p_flat = p.reshape(B * Dp * Hp * Wp, C, 1, 1, 1)
            out_pf = sphconv(p_flat)  
            out_pf = out_pf.reshape(B, Dp, Hp, Wp, self.spherical_out).permute(0, 4, 1, 2, 3)  
            out_per_phase.append(out_pf)

        merged = torch.cat(out_per_phase, dim=1)  
        return self.fuse(merged)

    def decompose_phases(self, x):
        B, C, D, H, W = x.shape
        pd, ph, pw = self.patch_size
        num_phases = pd * ph * pw

        assert D % pd == 0 and H % ph == 0 and W % pw == 0, \
            f"Spatial dimensions must be divisible by patch size. Received D={D}, H={H}, W={W} with patch_size={self.patch_size}"

        x_reshaped = x.view(B, C, D//pd, pd, H//ph, ph, W//pw, pw)
        x_permuted = x_reshaped.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x_reshaped = x_permuted.view(B, C, num_phases, D//pd, H//ph, W//pw)
        phases = torch.split(x_reshaped, 1, dim=2)  
        phases = [phase.squeeze(2) for phase in phases]  

        assert len(phases) == num_phases, \
            f"Expected {num_phases} phases, but got {len(phases)}"

        return phases

###############################################################################
# Transformation Prediction Head
###############################################################################

class TransformationPredictionHead(nn.Module):
    # def __init__(self, in_channels, hidden_dim=32, out_dim=6):
    #def __init__(self, in_channels, hidden_dim=32, out_dim=12):
    def __init__(self, in_channels, hidden_dim=32, out_dim=9):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim  = hidden_dim
        self.out_dim     = out_dim

        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, (1,1,5), padding=(0,0,2), padding_mode='circular'),
            nn.GELU(),
            ResidualBlockNoNorm(hidden_dim),
            nn.Conv3d(hidden_dim, hidden_dim, (1,1,3), padding=(0,0,1), padding_mode='circular', groups=hidden_dim),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, (1,5,1), padding=(0,2,0), padding_mode='circular'),
            nn.GELU(),
            ResidualBlockNoNorm(hidden_dim),
            nn.Conv3d(hidden_dim, hidden_dim, (1,3,1), padding=(0,1,0), padding_mode='circular', groups=hidden_dim),
            nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, (5,1,1), padding=(2,0,0), padding_mode='circular'),
            nn.GELU(),
            ResidualBlockNoNorm(hidden_dim),
            nn.Conv3d(hidden_dim, hidden_dim, (3,1,1), padding=(1,0,0), padding_mode='circular', groups=hidden_dim),
            nn.GELU()
        )

    
        self.merge_conv = nn.Sequential(
            nn.Conv3d(hidden_dim*3, hidden_dim, kernel_size=1, padding=0, padding_mode='circular'),
            nn.GELU()
        )

        self.pool = nn.AdaptiveAvgPool3d(4)  
        self.fc_in = hidden_dim*4*4*4
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)  
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        merged = torch.cat([b1, b2, b3], dim=1)  
        merged = self.merge_conv(merged)         

        pooled = self.pool(merged)
        B_, C_, D_, H_, W_ = pooled.shape
        flat = pooled.view(B_, C_*D_*H_*W_)

        out = self.fc(flat)
        #rot_r9 = out[:, :9]
        rot_r6 = out[:, :6]
            
        #trans = out[:, 9:]
        trans = out[:, 6:]
        #return torch.cat([rot_r9, trans], dim=1)
        return torch.cat([rot_r6, trans], dim=1)

class ResidualBlockNoNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, padding_mode='circular')
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, padding_mode='circular')
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + res)

###############################################################################
# SE3 Equivariant Transformer Block
###############################################################################
class SE3EquivariantTransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads, ff_hidden_dim, feature_type='vector'):
        super().__init__()
        assert in_channels % 3 == 0, "in_channels must be multiple of 3"
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.feat_per_head = in_channels // num_heads

        self.ff_hidden_dim = ((ff_hidden_dim + 2) // 3) * 3

        self.self_attn = SelfAttentionModule(
            in_channels=in_channels,
            num_heads=num_heads,
            feature_type=feature_type
        )

        self.norm1 = nn.InstanceNorm3d(in_channels)

        self.mix = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1, padding_mode='circular', groups=3),
            nn.GELU(),
            nn.Conv3d(in_channels, in_channels, 1, groups=3)
        )

        self.ff = nn.Sequential(
            nn.Conv3d(in_channels, self.ff_hidden_dim, 1, groups=3),
            nn.GELU(),
            nn.Conv3d(self.ff_hidden_dim, self.ff_hidden_dim, 3, padding=1, padding_mode='circular', groups=3),
            nn.GELU(),
            nn.Conv3d(self.ff_hidden_dim, in_channels, 1, groups=3)
        )

        self.norm2 = nn.InstanceNorm3d(in_channels)

    def forward(self, x, pos_emb, mask=None):
        B, C, D, H, W = x.shape

        attn_out = self.self_attn(x, pos_emb, mask)
        x = x + attn_out
        x = self.norm1(x)

        mixed_out = self.mix(x)
        x = x + mixed_out

        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)

        return x
    
###############################################################################
# Transformer
###############################################################################
# class Transformer(nn.Module):
#     def __init__(self):
#         super().__init__()
#     # def forward(self, x, rot_angles, translations):
#     # def forward(self, x, r9_params, translations):
#     #     B, C, D, H, W = x.shape
#     #     device = x.device
#     #     # R = kornia.geometry.axis_angle_to_rotation_matrix(rot_angles)[:, :3, :3]  
#     #     R = r9_to_matrix(r9_params)
#     #     R_inv = R.transpose(1, 2)
#     #     translations = translations.float()
#     #     t_inv = -torch.bmm(R_inv, translations.unsqueeze(2)).squeeze(2)  

#     #     transform_matrix = torch.zeros(B, 3, 4, device=device)
#     #     transform_matrix[:, :, :3] = R_inv
#     #     transform_matrix[:, :, 3] = t_inv

#     #     grid = F.affine_grid(transform_matrix, x.size(), align_corners=False)
#     #     transformed = F.grid_sample(
#     #         x, grid, mode='nearest', padding_mode='border', align_corners=False
#     #     )
#     #     return transformed
#     def forward(self, x, r6_params, translations):
#         B, C, D, H, W = x.shape
#         device = x.device
#         # R = kornia.geometry.axis_angle_to_rotation_matrix(rot_angles)[:, :3, :3]  
#         R = r6_to_matrix(r6_params)
#         R_inv = R.transpose(1, 2)
#         translations = translations.float()
#         t_inv = -torch.bmm(R_inv, translations.unsqueeze(2)).squeeze(2)  

#         transform_matrix = torch.zeros(B, 3, 4, device=device)
#         transform_matrix[:, :, :3] = R_inv
#         transform_matrix[:, :, 3] = t_inv

#         grid = F.affine_grid(transform_matrix, x.size(), align_corners=False)
#         transformed = F.grid_sample(
#             x, grid, mode='nearest', padding_mode='border', align_corners=False
#         )
#         return transformed
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, r6_params, translations):
        B, C, D, H, W = x.shape
        device = x.device
        
        R = r6_to_matrix(r6_params)
        R = R.to(x.dtype)
        translations = translations.to(x.dtype)
        
        affine_trans = torch.zeros(B,3,4, device=device, dtype=x.dtype)
        affine_trans[:,0,0] = 1
        affine_trans[:,1,1] = 1
        affine_trans[:,2,2] = 1
        
        dx = (2.0 * translations[:,0]) / W
        dy = (2.0 * translations[:,1]) / H
        dz = (2.0 * translations[:,2]) / D
        affine_trans[:,0,3] = dx
        affine_trans[:,1,3] = dy
        affine_trans[:,2,3] = dz
        
        grid_trans = F.affine_grid(affine_trans, x.size(), align_corners=False)
        x_translated = F.grid_sample(x, grid_trans, mode='nearest',
                                   padding_mode='border', align_corners=False)
        
        center = torch.tensor([W/2, H/2, D/2], device=device, dtype=x.dtype).view(1,3)
        center = center.repeat(B,1)
        center_col = center.view(B,3,1)
        shift_vec = center - torch.bmm(R, center_col).squeeze(-1)
        
        affine_rot = torch.zeros(B,3,4, device=device, dtype=x.dtype)
        affine_rot[:,:3,:3] = R
        affine_rot[:,:,3] = shift_vec
        
        grid_rot = F.affine_grid(affine_rot, x.size(), align_corners=False)
        transformed = F.grid_sample(x_translated, grid_rot, mode='nearest',
                                  padding_mode='border', align_corners=False)
        
        return transformed
###############################################################################
# SETransformerR9Sampling Module
###############################################################################
class SETransformerR6Sampling(nn.Module):
    def __init__(
        self,
        in_channels,
        num_transformer_blocks,
        num_heads,
        ff_hidden_dim,
        hidden_dim,
        feature_type='vector',
        patch_size=(4, 4, 4)
    ):
        super().__init__()
        print(f"SETransformerR9Sampling initialized")
        self.hidden_dim = hidden_dim
        self.feature_type = feature_type
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.out_D = 32 // patch_size[0]
        self.out_H = 32 // patch_size[1]
        self.out_W = 32 // patch_size[2]

        self.feature_extractor = PolyphaseFeatureExtractor(
            in_channels=in_channels,
            embed_dim=hidden_dim,
            patch_size=patch_size,
        )

        self.feature_comparator = FeatureComparator(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        self.pos_combiner = nn.Sequential(
            nn.Conv3d(hidden_dim * 2, hidden_dim, 1),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU()
        )

        self.pos_encoder = ShiftEquivariantPositionalEncoder(
            hidden_dim=hidden_dim,
        )

        self.transformer_blocks = nn.ModuleList([
            SE3EquivariantTransformerBlock(
                in_channels=hidden_dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                feature_type=feature_type,
            )
            for _ in range(num_transformer_blocks)
        ])

        self.trans_head = TransformationPredictionHead(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim
        )

        self.transformer = Transformer()

    def forward(self, input, target, attn_mask=None):
        target_features = self.feature_extractor(target)
        input_features = self.feature_extractor(input)

        pos_emb_target = self.pos_encoder(target_features)
        pos_emb_input = self.pos_encoder(input_features)

        combined_features = self.feature_comparator(target_features, input_features)

        combined_pos_emb = self.pos_combiner(torch.cat([pos_emb_target, pos_emb_input], dim=1))

        aligned_input = input
        for i, block in enumerate(self.transformer_blocks):
            combined_features = block(
                combined_features, combined_pos_emb, attn_mask
            )

            transformation_pred = self.trans_head(combined_features)

            # aligned_input = self.transformer(
            #     aligned_input,
            #     transformation_pred[:, :3],  
            #     transformation_pred[:, 3:]   
            # )
            
            aligned_input = self.transformer(
                aligned_input,
                transformation_pred[:, :6],
                transformation_pred[:, 6:]
            )

        return transformation_pred, aligned_input
    

# =============================================================================
# Helper Functions
# =============================================================================
def axis_angle_to_matrix(ax, ay, az, device):
    a = torch.tensor([ax, ay, az], device=device)
    angle = torch.norm(a)
    if angle < 1e-14:
        return torch.eye(3, device=device)
    axis = a / angle
    rotvec = axis * angle
    # Below requires kornia or a similar library that defines axis_angle_to_rotation_matrix
    R = kornia.geometry.axis_angle_to_rotation_matrix(rotvec.unsqueeze(0))[:, :3, :3]  # => (1,3,3)
    return R

def print_diff(test_name, diff):
    print(f"[{test_name}] => diff = {diff:.6f} ({diff:.2e})")

def random_valid_r9(device):
    # This function returns a "small" axis-angle rotation (3 random angles * 0.1)
    # Then transforms that axis angle into a 3x3 => r9, so we get a random rotation close to identity
    R = torch.eye(3, device=device)
    small_angles = torch.rand(3, device=device) * 0.1
    R_perturb = axis_angle_to_matrix(small_angles[0], small_angles[1], small_angles[2], device)
    R = torch.matmul(R, R_perturb)
    r9 = R.reshape(-1)
    return r9

def rotate_volume(volume, affine_matrix):
    """
    volume: (B, C, D, H, W)
    affine_matrix: (B, 3, 4)
    Returns rotated volume by applying affine_matrix via grid_sample.
    """
    B, C, D, H, W = volume.shape
    grid = F.affine_grid(affine_matrix, volume.size(), align_corners=False)
    vol_rot = F.grid_sample(volume, grid, mode='nearest', padding_mode='border', align_corners=False)
    return vol_rot

def translate_volume(volume, offsets):
    """
    volume: (B, C, D, H, W)
    offsets: (B, 3) => (d_offset, h_offset, w_offset) in voxel units
    Build an affine that shifts by these offsets, then call rotate_volume.
    """
    device = volume.device
    B, C, D, H, W = volume.shape

    affine = torch.zeros(B, 3, 4, device=device)
    affine[:, 0, 0] = 1
    affine[:, 1, 1] = 1
    affine[:, 2, 2] = 1

    # Convert raw voxel offsets to normalized affine offsets in [-1,1].
    d_off = offsets[:, 0] * (2.0 / D)
    h_off = offsets[:, 1] * (2.0 / H)
    w_off = offsets[:, 2] * (2.0 / W)

    affine[:, 2, 3] = d_off
    affine[:, 1, 3] = h_off
    affine[:, 0, 3] = w_off

    return rotate_volume(volume, affine)

def rotate_around_z(volume, angle_radians):
    """
    volume: (B, C, D, H, W)
    angle_radians: rotation around z-axis
    Build a (B,3,4) transform, apply rotate_volume.
    """
    device = volume.device
    B, _, _, _, _ = volume.shape

    rot_mat = torch.zeros(B, 3, 4, device=device)
    cosA = math.cos(angle_radians)
    sinA = math.sin(angle_radians)
    R_2d = torch.tensor([
        [ cosA, -sinA, 0],
        [ sinA,  cosA, 0],
        [   0,     0, 1]
    ], device=device)

    for i in range(B):
        rot_mat[i, :3, :3] = R_2d

    return rotate_volume(volume, rot_mat)


# =============================================================================
# Tests
# =============================================================================
class LPSDown3DTests:
    @staticmethod
    def test_shift_equivariance(device):
        model = LPSDown3D(
            patch_size=(4,4,4),
            in_channels=1, hidden_channels=8,
            temperature=1.0, use_gumbel=True,
            max_radius=1.0,
            sampling_ratio=1.0
        ).to(device).eval()

        B, C, D, H, W = 1, 1, 32, 32, 32
        x = torch.zeros(B, C, D, H, W, device=device)
        x[:,:,15:18,15:18,15:18] = 1.0

        shift = (3, 5, 2)
        x_shifted = torch.roll(x, shifts=shift, dims=(2, 3, 4))

        out1, _ = model(x)
        out2, _ = model(x_shifted)

        out1_rolled = torch.roll(out1, shifts=shift, dims=(2, 3, 4))
        diff = (out2 - out1_rolled).abs().mean().item()
        print_diff("test_shift_equivariance_lpsdown3d", diff)
        return diff

    @staticmethod
    def test_rotation_equivariance(device):
        model = LPSDown3D(
            patch_size=(4,4,4),
            in_channels=1, hidden_channels=8,
            temperature=1.0, use_gumbel=True,
            max_radius=1.0,
            sampling_ratio=1.0
        ).to(device).eval()

        B, C, D, H, W = 1, 1, 32, 32, 32
        x = torch.zeros(B, C, D, H, W, device=device)
        x[:,:,15:17,15:17,15:17] = 1.0

        angle = math.pi / 2
        transform = torch.zeros(B, 3, 4, device=device)
        R = axis_angle_to_matrix(0, 0, angle, device)
        transform[:, :3, :3] = R
        grid = F.affine_grid(transform, x.size(), align_corners=False)
        x_rot = F.grid_sample(x, grid, mode='nearest', align_corners=False)

        out1, _ = model(x)
        out2, _ = model(x_rot)

        grid2 = F.affine_grid(transform, out1.size(), align_corners=False)
        out1_rot = F.grid_sample(out1, grid2, mode='nearest', align_corners=False)

        diff = (out2 - out1_rot).abs().mean().item()
        print_diff("test_rotation_equivariance_lpsdown3d", diff)
        return diff

    @staticmethod
    def test_down_up_identity(device):
        lps_down = LPSDown3D(
            patch_size=(4,4,4),
            in_channels=1,
            hidden_channels=8,
            temperature=1.0,
            use_gumbel=True,
            max_radius=1.0,
            sampling_ratio=1.0
        ).to(device)
        lps_up = LPSUp3D(patch_size=(4,4,4)).to(device)

        B, C, D, H, W = 2, 1, 32, 32, 32
        vol = torch.zeros(B, C, D, H, W, device=device)
        for b in range(B):
            z1 = 10 + b*2
            z2 = z1 + 3
            vol[b, :, z1:z2, 15:18, 15:18] = 1.0

        with torch.no_grad():
            x_shifted, patch_probs = lps_down(vol)
            x_unshifted = lps_up(x_shifted, patch_probs)

        diff = (vol - x_unshifted).abs().mean().item()
        print(f"[test_lps_down_up_identity] => mean diff = {diff:.6f} ({diff:.2e})")
        assert diff < 1e-2, f"Down->Up identity test failed with diff={diff}"

    
    @staticmethod
    def test_rotation_parameter_prediction_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for LPSDown3D ===")
        model = LPSDown3D(
            patch_size=(4,4,4),
            in_channels=1, hidden_channels=8,
            temperature=1.0, use_gumbel=True,
            max_radius=1.0,
            sampling_ratio=1.0
        ).to(device).eval()

        B, C, D, H, W = 1, 1, 32, 32, 32
        x = torch.zeros(B, C, D, H, W, device=device)
        x[:,:,15:18,15:18,15:18] = 1.0

        x_rot = rotate_around_z(x, angle_radians=math.pi/4)

        out, _ = model(x)
        out_rot, _ = model(x_rot)
        out_rot_prime = rotate_around_z(out, math.pi/4)
        diff = (out_rot - out_rot_prime).abs().mean().item()
        print(f"[explicit rotation] diff = {diff:.6f}")
        return diff

    @staticmethod
    def test_translation_parameter_prediction_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TESTfor LPSDown3D ===")
        model = LPSDown3D(
            patch_size=(4,4,4),
            in_channels=1,
            hidden_channels=8,
            temperature=1.0,
            use_gumbel=True,
            max_radius=1.0,
            sampling_ratio=1.0
        ).to(device).eval()

        B, C, D, H, W = 1,1,32,32,32
        x = torch.zeros(B,C,D,H,W, device=device)
        x[:,:,12:15,12:15,12:15] = 1.0

        offsets = torch.tensor([[2.0,3.0,4.0]], device=device)
        x_shift = translate_volume(x, offsets)

        out, _ = model(x)
        out_shift, _ = model(x_shift)
        out_shift_prime = translate_volume(out, offsets)
        
        diff = (out_shift - out_shift_prime).abs().mean().item()
        print(f"[explicit translation] diff = {diff:.6f}")
        return diff


class LPSUp3DTests:
    @staticmethod
    def test_rotation_parameter_prediction_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for LPSUp3D ===")
        down = LPSDown3D(
            patch_size=(4,4,4),
            in_channels=1, hidden_channels=8,
            temperature=1.0, use_gumbel=True,
        ).to(device).eval()
        up = LPSUp3D(patch_size=(4,4,4)).to(device).eval()

        B, C, D, H, W = 1,1,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,15:18,15:18,15:18] = 1.0

        vol_rot = rotate_around_z(vol, math.pi/6)
        out_down, probs = down(vol)
        out_down_rot, probs_rot = down(vol_rot)

        out_up = up(out_down, probs)
        out_up_rot = up(out_down_rot, probs_rot)
        out_up_prime = rotate_around_z(out_up, math.pi/6)

        diff = (out_up_rot - out_up_prime).abs().mean().item()
        print(f"[explicit rotation] diff = {diff:.6f}")
        return diff

    @staticmethod
    def test_translation_parameter_prediction_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TESTfor LPSUp3D ===")
        down = LPSDown3D(patch_size=(4,4,4), in_channels=1).to(device).eval()
        up   = LPSUp3D(patch_size=(4,4,4)).to(device).eval()

        B,C,D,H,W = 1,1,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,10:14,10:14,10:14] = 1.0

        offsets = torch.tensor([[2.0,3.0,1.0]], device=device)
        vol_shift = translate_volume(vol, offsets)

        out_down, probs = down(vol)
        out_down_shift, probs_shift = down(vol_shift)

        out_up = up(out_down, probs)
        out_up_shift = up(out_down_shift, probs_shift)

        out_up_shift_prime = translate_volume(out_up, offsets)
        diff = (out_up_shift - out_up_shift_prime).abs().mean().item()
        print(f"[explicit translation] diff = {diff:.6f}")
        return diff


class PolyPatchEmbed3DTests:
    @staticmethod
    def test_rotation_parameter_prediction_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for PolyPatchEmbed3D ===")
        model = PolyPatchEmbed3D(patch_size=(4,4,4), in_chans=1, out_chans=8).to(device).eval()

        B,C,D,H,W = 1,1,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,12:16,12:16,12:16] = 1.0

        angle = math.pi/6
        vol_rot = rotate_around_z(vol, angle)

        out = model(vol)
        out_rot = model(vol_rot)
        out_rot_prime = rotate_around_z(out, angle)

        diff = (out_rot - out_rot_prime).abs().mean().item()
        print(f"[explicit rotation] diff = {diff:.6f}")
        return diff

    @staticmethod
    def test_translation_parameter_prediction_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TESTfor PolyPatchEmbed3D ===")
        model = PolyPatchEmbed3D((4,4,4), 1, 8).to(device).eval()
        B,C,D,H,W = 1,1,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,10:13,11:14,12:15] = 1.0

        offsets = torch.tensor([[2.0,4.0,3.0]], device=device)
        vol_shift = translate_volume(vol, offsets)

        out = model(vol)
        out_shift = model(vol_shift)
        out_shift_prime = translate_volume(out, offsets)

        diff = (out_shift - out_shift_prime).abs().mean().item()
        print(f"[explicit translation] diff = {diff:.6f}")
        return diff


class PosConv3DTests:
    @staticmethod
    def test_rotation_parameter_prediction_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for PosConv3D ===")
        model = PosConv3D(in_chans=1, out_chans=4).to(device).eval()

        B,C,D,H,W = 1,1,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,15:18,15:18,15:18] = 1.0

        angle = math.pi/6
        vol_rot = rotate_around_z(vol, angle)

        out = model(vol)
        out_rot = model(vol_rot)
        out_rot_prime = rotate_around_z(out, angle)

        diff = (out_rot - out_rot_prime).abs().mean().item()
        print(f"[explicit rotation] diff = {diff:.6f}")
        return diff

    @staticmethod
    def test_translation_parameter_prediction_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TESTfor PosConv3D ===")
        model = PosConv3D(in_chans=1, out_chans=4).to(device).eval()

        B,C,D,H,W = 1,1,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,12:15,12:15,12:15] = 1.0

        offsets = torch.tensor([[4.0,2.0,3.0]], device=device)
        vol_shift = translate_volume(vol, offsets)

        out = model(vol)
        out_shift = model(vol_shift)
        out_shift_prime = translate_volume(out, offsets)

        diff = (out_shift - out_shift_prime).abs().mean().item()
        print(f"[explicit translation] diff = {diff:.6f}")
        return diff


class FeatureComparatorTests:
    @staticmethod
    def test_shift_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TEST for FeatureComparator ===")
        comp = FeatureComparator(in_channels=4, hidden_dim=8).to(device).eval()

        B, C, D, H, W = 1, 4, 32, 32, 32
        target = torch.zeros(B, C, D, H, W, device=device)
        inp = torch.zeros(B, C, D, H, W, device=device)
        target[:,:,15:18,15:18,15:18] = 1.0
        inp[:,:,16:19,16:19,16:19] = 2.0

        shift = (2, 3, 5)
        target_shifted = torch.roll(target, shift, dims=(2, 3, 4))
        inp_shifted = torch.roll(inp, shift, dims=(2, 3, 4))

        out1 = comp(target, inp)
        out2 = comp(target_shifted, inp_shifted)
        out1_rolled = torch.roll(out1, shift, dims=(2, 3, 4))

        diff = (out2 - out1_rolled).abs().mean().item()
        print_diff("test_shift_equivariance_feature_comparator", diff)

    @staticmethod
    def test_rotation_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for FeatureComparator ===")
        comp = FeatureComparator(in_channels=4, hidden_dim=8).to(device).eval()

        B, C, D, H, W = 1, 4, 32, 32, 32
        target = torch.zeros(B, C, D, H, W, device=device)
        inp = torch.zeros(B, C, D, H, W, device=device)
        target[:,:,15:18,15:18,15:18] = 1.0
        inp[:,:,16:19,16:19,16:19] = 2.0

        angle = math.pi / 4
        transform = torch.zeros(B, 3, 4, device=device)
        R = axis_angle_to_matrix(0, 0, angle, device)
        transform[:, :3, :3] = R
        grid = F.affine_grid(transform, target.size(), align_corners=False)
        t_rot = F.grid_sample(target, grid, mode='nearest', align_corners=False)
        i_rot = F.grid_sample(inp, grid, mode='nearest', align_corners=False)

        out1 = comp(target, inp)
        out2 = comp(t_rot, i_rot)
        grid2 = F.affine_grid(transform, out1.size(), align_corners=False)
        out1_rot = F.grid_sample(out1, grid2, mode='nearest', align_corners=False)

        diff = (out2 - out1_rot).abs().mean().item()
        print_diff("test_rotation_equivariance_feature_comparator", diff)

    
    @staticmethod
    def test_rotation_parameter_prediction_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for FeatureComparator ===")
        comp = FeatureComparator(in_channels=4, hidden_dim=8).to(device).eval()

        B,C,D,H,W = 1,4,32,32,32
        targ = torch.zeros(B,C,D,H,W, device=device)
        inp  = torch.zeros(B,C,D,H,W, device=device)
        targ[:,:,15:18,15:18,15:18] = 1.0
        inp[:,:,16:19,16:19,16:19]  = 2.0

        angle = math.pi/4
        targ_rot = rotate_around_z(targ, angle)
        inp_rot  = rotate_around_z(inp,  angle)

        out = comp(targ, inp)
        out_rot = comp(targ_rot, inp_rot)
        out_rot_prime = rotate_around_z(out, angle)

        diff = (out_rot - out_rot_prime).abs().mean().item()
        print(f"[explicit rotation] diff = {diff:.6f}")
        return diff

    @staticmethod
    def test_translation_parameter_prediction_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TESTfor FeatureComparator ===")        
        comp = FeatureComparator(in_channels=4, hidden_dim=8).to(device).eval()

        B,C,D,H,W = 1,4,32,32,32
        targ = torch.zeros(B,C,D,H,W, device=device)
        inp  = torch.zeros(B,C,D,H,W, device=device)
        targ[:,:,15:18,15:18,15:18] = 1.0
        inp[:,:,16:19,16:19,16:19]  = 2.0

        offsets = torch.tensor([[2.0,3.0,4.0]], device=device)
        targ_shift = translate_volume(targ, offsets)
        inp_shift  = translate_volume(inp, offsets)

        out = comp(targ, inp)
        out_shift = comp(targ_shift, inp_shift)
        out_shift_prime = translate_volume(out, offsets)

        diff = (out_shift - out_shift_prime).abs().mean().item()
        print(f"[explicit translation] diff = {diff:.6f}")
        return diff


class PolyphaseFeatureExtractorTests:
    @staticmethod
    def test_shift_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TEST for PolyphaseFeatureExtractor ===")
        extractor = PolyphaseFeatureExtractor(1, 8, (4,4,4)).to(device).eval()

        B, C, D, H, W = 1, 1, 32, 32, 32
        x = torch.zeros(B, C, D, H, W, device=device)
        x[:,:,15:18,15:18,15:18] = 1.0

        shift = (2, 5, 3)
        x_shifted = torch.roll(x, shift, dims=(2, 3, 4))

        out1 = extractor(x)
        out2 = extractor(x_shifted)
        out1_rolled = torch.roll(out1, shift, dims=(2, 3, 4))
        diff = (out2 - out1_rolled).abs().mean().item()
        print_diff("test_shift_equivariance_polyphase_feature_extractor", diff)

    @staticmethod
    def test_rotation_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for PolyphaseFeatureExtractor ===")
        extractor = PolyphaseFeatureExtractor(1, 8, (4,4,4)).to(device).eval()

        B, C, D, H, W = 1, 1, 32, 32, 32
        x = torch.zeros(B, C, D, H, W, device=device)
        x[:,:,15:18,15:18,15:18] = 1.0

        angle = math.pi / 4
        transform = torch.zeros(B, 3, 4, device=device)
        R = axis_angle_to_matrix(0, 0, angle, device)
        transform[:, :3, :3] = R
        grid = F.affine_grid(transform, x.size(), align_corners=False)
        x_rot = F.grid_sample(x, grid, mode='nearest', align_corners=False)

        out1 = extractor(x)
        out2 = extractor(x_rot)

        grid2 = F.affine_grid(transform, out1.size(), align_corners=False)
        out1_rot = F.grid_sample(out1, grid2, mode='nearest', align_corners=False)

        diff = (out2 - out1_rot).abs().mean().item()
        print_diff("test_rotation_equivariance_polyphase_feature_extractor", diff)
    
    @staticmethod
    def test_rotation_parameter_prediction_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for PolyphaseFeatureExtractor ===")
        model = PolyphaseFeatureExtractor(in_channels=1, embed_dim=8, patch_size=(4,4,4)).to(device).eval()

        B,C,D,H,W = 1,1,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,12:15,14:17,16:19] = 1.0

        angle = math.pi/3
        vol_rot = rotate_around_z(vol, angle)
        out = model(vol)
        out_rot = model(vol_rot)
        out_rot_prime = rotate_around_z(out, angle)

        diff = (out_rot - out_rot_prime).abs().mean().item()
        print(f"[explicit rotation] diff = {diff:.6f}")
        return diff

    @staticmethod
    def test_translation_parameter_prediction_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TESTfor PolyphaseFeatureExtractor ===")
        model = PolyphaseFeatureExtractor(in_channels=1, embed_dim=8, patch_size=(4,4,4)).to(device).eval()
        
        B,C,D,H,W = 1,1,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,13:17,16:20,19:23] = 1.0

        offsets = torch.tensor([[2.0,3.0,1.0]], device=device)
        vol_shift = translate_volume(vol, offsets)

        out = model(vol)
        out_shift = model(vol_shift)
        out_shift_prime = translate_volume(out, offsets)

        diff = (out_shift - out_shift_prime).abs().mean().item()
        print(f"[explicit translation] diff = {diff:.6f}")
        return diff


class ShiftEquivariantPositionalEncoderTests:
    @staticmethod
    def test_shift_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TEST for ShiftEquivariantPositionalEncoder ===")
        hidden_dim = 12
        enc = ShiftEquivariantPositionalEncoder(hidden_dim=hidden_dim).to(device).eval()

        B, C, D, H, W = 1, hidden_dim, 32, 32, 32  # C matches hidden_dim
        vol = torch.zeros(B, C, D, H, W, device=device)
        vol[:, :, 15:18, 15:18, 15:18] = 1.0

        shift = (3, 2, 5)
        vol_shifted = torch.roll(vol, shift, dims=(2, 3, 4))

        out1 = enc(vol)
        out2 = enc(vol_shifted)
        out1_rolled = torch.roll(out1, shift, dims=(2, 3, 4))

        diff = (out2 - out1_rolled).abs().mean().item()
        print(f"[test_shift_equivariance_shiftpos_encoder] => diff = {diff:.6f} ({diff:.2e})")
        assert diff < 1e-2, f"Shift equivariance test failed with diff={diff}"

    @staticmethod
    def test_rotation_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for ShiftEquivariantPositionalEncoder ===")
        hidden_dim = 12  
        enc = ShiftEquivariantPositionalEncoder(hidden_dim=hidden_dim).to(device).eval()

        B, C, D, H, W = 1, hidden_dim, 32, 32, 32  
        vol = torch.zeros(B, C, D, H, W, device=device)
        vol[:, :, 15:18, 15:18, 15:18] = 1.0

        angle = math.pi / 6  
        transform = torch.zeros(B, 3, 4, device=device)
        R = axis_angle_to_matrix(0, 0, angle, device)
        transform[:, :3, :3] = R
        grid = F.affine_grid(transform, vol.size(), align_corners=False)
        vol_rot = F.grid_sample(vol, grid, mode='nearest', align_corners=False)

        out1 = enc(vol)
        out2 = enc(vol_rot)

        grid2 = F.affine_grid(transform, out1.size(), align_corners=False)
        out1_rot = F.grid_sample(out1, grid2, mode='nearest', align_corners=False)

        diff = (out2 - out1_rot).abs().mean().item()
        print_diff("test_rotation_equivariance_shiftpos_encoder", diff)

    @staticmethod
    def test_rotation_parameter_prediction_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for ShiftEquivariantPositionalEncoder ===")
        enc = ShiftEquivariantPositionalEncoder(hidden_dim=12).to(device).eval()

        B,C,D,H,W = 1,12,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,15:18,15:18,15:18] = 1.0

        angle = math.pi/4
        vol_rot = rotate_around_z(vol, angle)

        out = enc(vol)
        out_rot = enc(vol_rot)
        out_rot_prime = rotate_around_z(out, angle)

        diff = (out_rot - out_rot_prime).abs().mean().item()
        print(f"[explicit rotation] diff = {diff:.6f}")
        return diff

    @staticmethod
    def test_translation_parameter_prediction_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TESTfor ShiftEquivariantPositionalEncoder ===")
        enc = ShiftEquivariantPositionalEncoder(hidden_dim=12).to(device).eval()
        
        B,C,D,H,W = 1,12,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,10:13,14:17,18:21] = 1.0

        offsets = torch.tensor([[3.0,2.0,1.0]], device=device)
        vol_shift = translate_volume(vol, offsets)

        out = enc(vol)
        out_shift = enc(vol_shift)
        out_shift_prime = translate_volume(out, offsets)

        diff = (out_shift - out_shift_prime).abs().mean().item()
        print(f"[explicit translation] diff = {diff:.6f}")
        return diff


class CNNPolyphaseProcessorTests:
    @staticmethod
    def test_shift_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TEST for CNNPolyphaseProcessor ===")
        proc = CNNPolyphaseProcessor(in_channels=1, out_channels=4, patch_size=(4,4,4)).to(device).eval()

        B, C, D, H, W = 1, 1, 32, 32, 32
        vol = torch.zeros(B, C, D, H, W, device=device)
        vol[:,:,15:18,15:18,15:18] = 1.0

        shift = (3, 5, 2)
        vol_shifted = torch.roll(vol, shift, dims=(2, 3, 4))

        out1 = proc(vol)
        out2 = proc(vol_shifted)
        out1_rolled = torch.roll(out1, shift, dims=(2, 3, 4))
        diff = (out2 - out1_rolled).abs().mean().item()
        print_diff("test_shift_equivariance_cnnpoly_processor", diff)

    @staticmethod
    def test_rotation_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for CNNPolyphaseProcessor ===")
        proc = CNNPolyphaseProcessor(in_channels=1, out_channels=4, patch_size=(4,4,4)).to(device).eval()

        B, C, D, H, W = 1, 1, 32, 32, 32
        vol = torch.zeros(B, C, D, H, W, device=device)
        vol[:,:,15:18,15:18,15:18] = 1.0

        angle = math.pi / 4
        transform = torch.zeros(B, 3, 4, device=device)
        R = axis_angle_to_matrix(0, 0, angle, device)
        transform[:, :3, :3] = R
        grid = F.affine_grid(transform, vol.size(), align_corners=False)
        vol_rot = F.grid_sample(vol, grid, mode='nearest', align_corners=False)

        out1 = proc(vol)
        out2 = proc(vol_rot)

        grid2 = F.affine_grid(transform, out1.size(), align_corners=False)
        out1_rot = F.grid_sample(out1, grid2, mode='nearest', align_corners=False)

        diff = (out2 - out1_rot).abs().mean().item()
        print_diff("test_rotation_equivariance_cnnpoly_processor", diff)

    
    @staticmethod
    def test_rotation_parameter_prediction_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for CNNPolyphaseProcessor ===")
        proc = CNNPolyphaseProcessor(in_channels=1, out_channels=4, patch_size=(4,4,4)).to(device).eval()

        B,C,D,H,W = 1,1,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,15:18,15:18,15:18] = 1.0

        angle = math.pi/4
        vol_rot = rotate_around_z(vol, angle)

        out = proc(vol)
        out_rot = proc(vol_rot)
        out_rot_prime = rotate_around_z(out, angle)

        diff = (out_rot - out_rot_prime).abs().mean().item()
        print(f"[explicit rotation] diff = {diff:.6f}")
        return diff

    @staticmethod
    def test_translation_parameter_prediction_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TESTfor CNNPolyphaseProcessor ===")
        proc = CNNPolyphaseProcessor(in_channels=1, out_channels=4, patch_size=(4,4,4)).to(device).eval()
        
        B,C,D,H,W = 1,1,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,12:16,14:18,10:14] = 1.0

        offsets = torch.tensor([[2.0,3.0,4.0]], device=device)
        vol_shift = translate_volume(vol, offsets)

        out = proc(vol)
        out_shift = proc(vol_shift)
        out_shift_prime = translate_volume(out, offsets)

        diff = (out_shift - out_shift_prime).abs().mean().item()
        print(f"[explicit translation] diff = {diff:.6f}")
        return diff


class TransformationPredictionHeadTests:
    @staticmethod
    def test_shift_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TEST for TransformationPredictionHead ===")
        in_channels = 4
        hidden_dim = 16
        head = TransformationPredictionHead(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_dim=12
        ).to(device).eval()

        B, C, D, H, W = 1, in_channels, 32, 32, 32
        vol = torch.zeros(B, C, D, H, W, device=device)
        vol[:, :, 10:14, 15:19, 20:24] = 1.0

        shift = (3, 5, 2)
        vol_shifted = torch.roll(vol, shifts=shift, dims=(2, 3, 4))

        out1 = head(vol)
        out2 = head(vol_shifted)
        
        r9_diff = (out1[:, :9] - out2[:, :9]).abs().mean().item()
        trans_diff = (out1[:, 9:] - out2[:, 9:]).abs().mean().item()
        
        print(f"R9 diff: {r9_diff:.6f}")
        print(f"Translation diff: {trans_diff:.6f}")
        print_diff("test_shift_equivariance_transformation_pred_head", (r9_diff + trans_diff)/2)

    @staticmethod
    def test_rotation_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for TransformationPredictionHead ===")
        in_channels = 4
        hidden_dim = 16
        head = TransformationPredictionHead(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_dim=12
        ).to(device).eval()

        B, C, D, H, W = 1, in_channels, 32, 32, 32
        vol = torch.zeros(B, C, D, H, W, device=device)
        vol[:, :, 12:16, 10:14, 17:21] = 1.0

        angle = math.pi / 4
        transform = torch.zeros(B, 3, 4, device=device)
        R = axis_angle_to_matrix(0, 0, angle, device)
        transform[:, :3, :3] = R
        grid = F.affine_grid(transform, vol.size(), align_corners=False)
        vol_rot = F.grid_sample(vol, grid, mode='nearest', align_corners=False)

        out1 = head(vol)     
        out2 = head(vol_rot) 

        r9_diff = (out1[:, :9] - out2[:, :9]).abs().mean().item()
        trans_diff = (out1[:, 9:] - out2[:, 9:]).abs().mean().item()
        
        print(f"R9 diff: {r9_diff:.6f}")
        print(f"Translation diff: {trans_diff:.6f}")
        print_diff("test_rotation_equivariance_transformation_pred_head", (r9_diff + trans_diff)/2)
    
    @staticmethod
    def test_rotation_parameter_prediction_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for TransformationPredictionHead ===")
        head = TransformationPredictionHead(in_channels=4, hidden_dim=16, out_dim=12).to(device).eval()
        
        B,C,D,H,W = 1,4,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,15:19,15:19,15:19] = 1.0

        angle = math.pi/4
        vol_rot = rotate_around_z(vol, angle)

        out = head(vol)
        out_rot = head(vol_rot)

        diff = (out - out_rot).abs().mean().item()
        print(f"[explicit rotation (transform preds)] diff = {diff:.6f}")
        return diff

    @staticmethod
    def test_translation_parameter_prediction_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TESTfor TransformationPredictionHead ===")
        head = TransformationPredictionHead(in_channels=4, hidden_dim=16, out_dim=12).to(device).eval()

        B,C,D,H,W = 1,4,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,10:14,15:19,20:24] = 1.0

        offsets = torch.tensor([[2.0,3.0,4.0]], device=device)
        vol_shift = translate_volume(vol, offsets)

        out = head(vol)
        out_shift = head(vol_shift)

        diff = (out - out_shift).abs().mean().item()
        print(f"[explicit translation (transform preds)] diff = {diff:.6f}")
        return diff


class SE3EquivariantTransformerBlockTests:
    @staticmethod
    def test_shift_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TEST for SE3EquivariantTransformerBlock ===")
        in_channels = 6 
        block = SE3EquivariantTransformerBlock(
            in_channels=in_channels,
            num_heads=2,
            ff_hidden_dim=12,
            feature_type='vector'
        ).to(device).eval()

        B, C, D, H, W = 1, in_channels, 32, 32, 32
        feats = torch.zeros(B, C, D, H, W, device=device)
        feats[:, :, 10:14, 15:19, 20:24] = 1.0

        pos_emb = torch.zeros_like(feats)
        pos_emb[:, :, 11:15, 14:18, 19:23] = 2.0

        shift = (3, 2, 5)
        feats_shifted = torch.roll(feats, shifts=shift, dims=(2, 3, 4))
        pos_emb_shifted = torch.roll(pos_emb, shifts=shift, dims=(2, 3, 4))

        out1 = block(feats, pos_emb)
        out2 = block(feats_shifted, pos_emb_shifted)

        out1_rolled = torch.roll(out1, shifts=shift, dims=(2, 3, 4))
        diff = (out2 - out1_rolled).abs().mean().item()
        print_diff("test_shift_equivariance_se3transformer_block", diff)

    @staticmethod
    def test_rotation_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for SE3EquivariantTransformerBlock ===")
        in_channels = 6  
        block = SE3EquivariantTransformerBlock(
            in_channels=in_channels,
            num_heads=2,
            ff_hidden_dim=12,
            feature_type='vector'
        ).to(device).eval()

        B, C, D, H, W = 1, in_channels, 32, 32, 32
        feats = torch.zeros(B, C, D, H, W, device=device)
        feats[:, :, 10:14, 15:19, 20:24] = 1.0
        pos_emb = torch.zeros_like(feats)
        pos_emb[:, :, 5:9, 10:14, 15:19] = 1.0

        transform = torch.zeros(B, 3, 4, device=device)
        r9 = random_valid_r9(device)
        R = r9_to_matrix(r9)
        transform[:, :3, :3] = R
        
        grid = F.affine_grid(transform, feats.size(), align_corners=False)
        feats_rot = F.grid_sample(feats, grid, mode='nearest', align_corners=False)
        pos_emb_rot = F.grid_sample(pos_emb, grid, mode='nearest', align_corners=False)

        out1 = block(feats, pos_emb)
        out2 = block(feats_rot, pos_emb_rot)

        grid2 = F.affine_grid(transform, out1.size(), align_corners=False)
        out1_rot = F.grid_sample(out1, grid2, mode='nearest', align_corners=False)

        diff = (out2 - out1_rot).abs().mean().item()
        print_diff("test_rotation_equivariance_se3transformer_block", diff)

    @staticmethod
    def test_rotation_parameter_prediction_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for SE3EquivariantTransformerBlock ===")
        block = SE3EquivariantTransformerBlock(
            in_channels=6, num_heads=2, ff_hidden_dim=12, feature_type='vector'
        ).to(device).eval()

        B,C,D,H,W = 1,6,32,32,32
        feats = torch.zeros(B,C,D,H,W, device=device)
        feats[:,:,10:14,15:19,15:19] = 1.0
        pos_emb = torch.zeros_like(feats)
        pos_emb[:,:,12:16,14:18,14:18] = 2.0

        feats_rot = rotate_around_z(feats, math.pi/3)
        pos_emb_rot = rotate_around_z(pos_emb, math.pi/3)

        out = block(feats, pos_emb)
        out_rot = block(feats_rot, pos_emb_rot)
        out_rot_prime = rotate_around_z(out, math.pi/3)

        diff = (out_rot - out_rot_prime).abs().mean().item()
        print(f"[explicit rotation] diff = {diff:.6f}")
        return diff

    @staticmethod
    def test_translation_parameter_prediction_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TESTfor SE3EquivariantTransformerBlock ===")
        block = SE3EquivariantTransformerBlock(
            in_channels=6, num_heads=2, ff_hidden_dim=12, feature_type='vector'
        ).to(device).eval()

        B,C,D,H,W = 1,6,32,32,32
        feats = torch.zeros(B,C,D,H,W, device=device)
        feats[:,:,10:14,15:19,15:19] = 1.0
        pos_emb = torch.zeros_like(feats)
        pos_emb[:,:,14:18,19:23,19:23] = 3.0

        offsets = torch.tensor([[3.0,2.0,4.0]], device=device)
        feats_shift = translate_volume(feats, offsets)
        pos_emb_shift = translate_volume(pos_emb, offsets)

        out = block(feats, pos_emb)
        out_shift = block(feats_shift, pos_emb_shift)
        out_shift_prime = translate_volume(out, offsets)

        diff = (out_shift - out_shift_prime).abs().mean().item()
        print(f"[explicit translation] diff = {diff:.6f}")
        return diff


class TransformerTests:
    @staticmethod
    def test_shift_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TEST for Transformer ===")
        transformer_module = Transformer().to(device).eval()

        B, C, D, H, W = 1, 4, 32, 32, 32
        vol = torch.zeros(B, C, D, H, W, device=device)
        vol[:, :, 12:16, 10:14, 18:22] = 1.0

        r9_params = random_valid_r9(device).unsqueeze(0)
        translations = torch.tensor([[0.1, -0.1, 0.2]], device=device, dtype=torch.float64)

        shift = (4, 3, 6)
        vol_shifted = torch.roll(vol, shift, dims=(2, 3, 4))

        out1 = transformer_module(vol, r9_params, translations)
        out2 = transformer_module(vol_shifted, r9_params, translations)
        
        out1_rolled = torch.roll(out1, shift, dims=(2, 3, 4))

        diff = (out2 - out1_rolled).abs().mean().item()
        print_diff("test_shift_equivariance_transformer", diff)

    @staticmethod
    def test_rotation_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for Transformer ===")
        transformer_module = Transformer().to(device).eval()

        B, C, D, H, W = 1, 4, 32, 32, 32
        vol = torch.zeros(B, C, D, H, W, device=device)
        vol[:, :, 12:16, 10:14, 18:22] = 1.0

        r9_params = random_valid_r9(device).unsqueeze(0)
        translations = torch.tensor([[0.1, -0.1, 0.2]], device=device, dtype=torch.float64)

        transform = torch.zeros(B, 3, 4, device=device)
        test_r9 = random_valid_r9(device)
        R = r9_to_matrix(test_r9)
        transform[:, :3, :3] = R
        
        grid = F.affine_grid(transform, vol.size(), align_corners=False)
        vol_rot = F.grid_sample(vol, grid, mode='nearest', align_corners=False)

        out1 = transformer_module(vol, r9_params, translations)
        out2 = transformer_module(vol_rot, r9_params, translations)
        
        grid2 = F.affine_grid(transform, out1.size(), align_corners=False)
        out1_rot = F.grid_sample(out1, grid2, mode='nearest', align_corners=False)

        diff = (out2 - out1_rot).abs().mean().item()
        print_diff("test_rotation_equivariance_transformer", diff)

    @staticmethod
    def test_rotation_parameter_prediction_equivariance(device):
        print("\n=== ROTATION EQUIVARIANCE TEST for Transformer ===")
        model = Transformer().to(device).eval()

        B,C,D,H,W = 1,4,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,10:14,15:19,20:24] = 1.0

        r9_base = torch.eye(3, device=device).reshape(9).unsqueeze(0)
        t = torch.zeros(B,3,device=device)

        angle = math.pi/6
        vol_rot = rotate_around_z(vol, angle)

        out = model(vol, r9_base, t)
        out_rot = model(vol_rot, r9_base, t)
        out_rot_prime = rotate_around_z(out, angle)

        diff = (out_rot - out_rot_prime).abs().mean().item()
        print(f"[explicit rotation] diff = {diff:.6f}")
        return diff

    @staticmethod
    def test_translation_parameter_prediction_equivariance(device):
        print("\n=== SHIFT EQUIVARIANCE TESTfor Transformer ===")
        model = Transformer().to(device).eval()
        B,C,D,H,W = 1,4,32,32,32
        vol = torch.zeros(B,C,D,H,W, device=device)
        vol[:,:,12:16,14:18,18:22] = 1.0

        r9_base = torch.eye(3, device=device).reshape(9).unsqueeze(0)
        t = torch.zeros(B,3,device=device)

        offsets = torch.tensor([[2.0,3.0,1.0]], device=device)
        vol_shift = translate_volume(vol, offsets)

        out = model(vol, r9_base, t)
        out_shift = model(vol_shift, r9_base, t)
        out_shift_prime = translate_volume(out, offsets)

        diff = (out_shift - out_shift_prime).abs().mean().item()
        print(f"[explicit translation] diff = {diff:.6f}")
        return diff

class SETransformerR9SamplingTests:
    @staticmethod
    def test_forward_pass(device):
        print("Testing SETransformerR9Sampling single forward pass...")

        batch_size = 1
        in_channels = 1
        spatial_size = 32

        input_tensor = torch.randn(batch_size, in_channels, spatial_size, spatial_size, spatial_size).to(device)
        target_tensor = torch.randn(batch_size, in_channels, spatial_size, spatial_size, spatial_size).to(device)

        model = SETransformerR9Sampling(
            in_channels=in_channels,
            num_transformer_blocks=4,
            num_heads=4,
            ff_hidden_dim=256,
            hidden_dim=60
        ).to(device)

        print("\nModel initialized with:")
        print(f"- Input channels: {in_channels}")
        print(f"- Transformer blocks: 4")
        print(f"- Number of heads: 4")
        print(f"- FF hidden dim: 256")
        print(f"- Hidden dim: 60")

        print("\nPerforming forward pass...")
        with torch.no_grad():
            trans_pred, aligned_input = model(input_tensor, target_tensor)

        print("\nOutput shapes:")
        print(f"Transformation prediction: {trans_pred.shape}")
        print(f"Aligned input: {aligned_input.shape}")

        # expected_trans_pred_shape = (batch_size, 6)
        expected_trans_pred_shape = (batch_size, 12)
        expected_aligned_shape = input_tensor.shape

        assert trans_pred.shape == expected_trans_pred_shape, \
            f"Transformation prediction shape mismatch. Expected {expected_trans_pred_shape}, got {trans_pred.shape}"
        assert aligned_input.shape == expected_aligned_shape, \
            f"Aligned input shape mismatch. Expected {expected_aligned_shape}, got {aligned_input.shape}"

        assert not torch.isnan(trans_pred).any(), "NaN values in transformation prediction"
        assert not torch.isnan(aligned_input).any(), "NaN values in aligned input"

        print("\nAll shape checks passed!")
        print("\nTransformation predictions:")
        print(f"Rotations (radians): {trans_pred[0, :9].cpu().numpy()}")
        print(f"Translations: {trans_pred[0, 9:].cpu().numpy()}")
        R = r9_to_matrix(trans_pred[0, :9])
        det = torch.linalg.det(R)
        print(f"Rotation matrix determinant: {det.item():.6f}") 
        print("\nSETransformerR9Sampling forward pass test completed successfully!")

class FullSETransformerR9SamplingPipelineTests:
    @staticmethod
    def test_pipeline(device):
        print("\n=== FULL SHIFT + ROTATION TEST for SETransformerR9Sampling Pipeline ===")
        model = SETransformerR9Sampling(
            in_channels=1,
            num_transformer_blocks=2,
            num_heads=2,
            ff_hidden_dim=24,
            hidden_dim=6,   
            feature_type='vector',
            patch_size=(4,4,4)
        ).to(device).eval()

        B, C, D, H, W = 1, 1, 32, 32, 32
        input_vol = torch.zeros(B, C, D, H, W, device=device)
        target_vol = torch.zeros(B, C, D, H, W, device=device)
        input_vol[:, :, 10:14, 15:19, 20:24] = 1.0
        target_vol[:, :, 12:16, 9:13, 22:26] = 1.0

        shift = (2, 5, 3)
        input_shifted = torch.roll(input_vol, shift, dims=(2, 3, 4))
        target_shifted = torch.roll(target_vol, shift, dims=(2, 3, 4))

        out1_pred, out1_aligned = model(input_vol, target_vol)
        out2_pred, out2_aligned = model(input_shifted, target_shifted)

        r9_diff = (out1_pred[:, :9] - out2_pred[:, :9]).abs().mean().item()
        trans_diff = (out1_pred[:, 9:] - out2_pred[:, 9:]).abs().mean().item()
        print(f"Shift test - R9 diff: {r9_diff:.6f}")
        print(f"Shift test - Translation diff: {trans_diff:.6f}")

        out1_aligned_rolled = torch.roll(out1_aligned, shift, dims=(2, 3, 4))
        diff_shift_eq = (out2_aligned - out1_aligned_rolled).abs().mean().item()
        print(f"[test_full_SETransformerR9Sampling_pipeline => SHIFT eq diff] = {diff_shift_eq:.6f} ({diff_shift_eq:.2e})")

        transform = torch.zeros(B, 3, 4, device=device)
        test_r9 = random_valid_r9(device)
        R = r9_to_matrix(test_r9)
        transform[:, :3, :3] = R
        
        grid = F.affine_grid(transform, input_vol.size(), align_corners=False)
        input_rot = F.grid_sample(input_vol, grid, mode='nearest', align_corners=False)
        target_rot = F.grid_sample(target_vol, grid, mode='nearest', align_corners=False)

        out3_pred, out3_aligned = model(input_rot, target_rot)

        r9_rot_diff = (out1_pred[:, :9] - out3_pred[:, :9]).abs().mean().item()
        trans_rot_diff = (out1_pred[:, 9:] - out3_pred[:, 9:]).abs().mean().item()
        print(f"Rotation test - R9 diff: {r9_rot_diff:.6f}")
        print(f"Rotation test - Translation diff: {trans_rot_diff:.6f}")

        grid2 = F.affine_grid(transform, out1_aligned.size(), align_corners=False)
        out1_aligned_rot = F.grid_sample(out1_aligned, grid2, mode='nearest', align_corners=False)
        diff_rot_eq = (out3_aligned - out1_aligned_rot).abs().mean().item()
        print(f"[test_full_SETransformerR9Sampling_pipeline => ROT eq diff] = {diff_rot_eq:.6f} ({diff_rot_eq:.2e})")


class EdgeCaseTests:
   @staticmethod
   def test_zero_volume_input(device):
       print("\n=== Testing zero volume input ===")
       model = SETransformerR9Sampling(
           in_channels=1,
           num_transformer_blocks=2,
           num_heads=2,
           ff_hidden_dim=24,
           hidden_dim=6,
           feature_type='vector',
           patch_size=(4,4,4)
       ).to(device).eval()
       
       B, C, D, H, W = 1, 1, 32, 32, 32
       zero_vol = torch.zeros(B, C, D, H, W, device=device)
       
       with torch.no_grad():
           trans_pred, aligned = model(zero_vol, zero_vol)
       
       assert not torch.isnan(trans_pred).any(), "NaN in transformation prediction"
       assert not torch.isnan(aligned).any(), "NaN in aligned output"
       print("Zero volume input test passed!")

   @staticmethod 
   def test_extreme_rotations(device):
       print("\n=== Testing extreme rotation angles ===")
       model = SETransformerR9Sampling(
           in_channels=1,
           num_transformer_blocks=2,
           num_heads=2,
           ff_hidden_dim=24,
           hidden_dim=6,
           feature_type='vector',
           patch_size=(4,4,4)
       ).to(device).eval()

       B, C, D, H, W = 1, 1, 32, 32, 32
       input_vol = torch.zeros(B, C, D, H, W, device=device)
       input_vol[:,:,15:18,15:18,15:18] = 1.0

       angles = [
           math.pi,
           2*math.pi - 0.017,
           0.001,
           math.pi/2,
           3*math.pi/2
       ]

       for angle in angles:
           input_rot = rotate_around_z(input_vol, angle)
           target_vol = torch.roll(input_vol, shifts=(2,3,1), dims=(2,3,4))
           
           with torch.no_grad():
               trans_pred, aligned = model(input_rot, target_vol)
           
           assert not torch.isnan(trans_pred).any(), f"NaN in transformation prediction at angle {angle}"
           assert not torch.isnan(aligned).any(), f"NaN in aligned output at angle {angle}"
           print(f"Extreme rotation test passed for angle: {angle:.3f} radians")


class NumericalStabilityTests:
   @staticmethod
   def test_input_scale_stability(device):
       print("\n=== Testing input scale stability ===")
       model = SETransformerR9Sampling(
           in_channels=1,
           num_transformer_blocks=2,
           num_heads=2,
           ff_hidden_dim=24,
           hidden_dim=6,
           feature_type='vector',
           patch_size=(4,4,4)
       ).to(device).eval()
       
       B, C, D, H, W = 1, 1, 32, 32, 32
       base_vol = torch.zeros(B, C, D, H, W, device=device)
       base_vol[:,:,15:18,15:18,15:18] = 1.0

       scales = [1e-6, 1e-3, 1.0, 1e3, 1e6]
       
       for scale in scales:
           input_vol = base_vol * scale
           target_vol = torch.roll(input_vol, shifts=(2,3,1), dims=(2,3,4))
           
           with torch.no_grad():
               trans_pred, aligned = model(input_vol, target_vol)
           
           assert not torch.isnan(trans_pred).any(), f"NaN in transformation prediction at scale {scale}"
           assert not torch.isnan(aligned).any(), f"NaN in aligned output at scale {scale}"
           print(f"Scale stability test passed for scale: {scale:.0e}")

   @staticmethod
   def test_gradient_stability(device):
       print("\n=== Testing gradient stability ===")
       model = SETransformerR9Sampling(
           in_channels=1,
           num_transformer_blocks=2,
           num_heads=2,
           ff_hidden_dim=24,
           hidden_dim=6,
           feature_type='vector',
           patch_size=(4,4,4)
       ).to(device).train()
       
       B, C, D, H, W = 1, 1, 32, 32, 32
       input_vol = torch.zeros(B, C, D, H, W, device=device)
       input_vol[:,:,15:18,15:18,15:18] = 1.0
       target_vol = torch.roll(input_vol, shifts=(2,3,1), dims=(2,3,4))
       
       grad_norms = []
       for i in range(10):
           trans_pred, aligned = model(input_vol, target_vol)
           loss = F.mse_loss(aligned, target_vol)
           loss.backward()
           
           total_norm = 0
           for p in model.parameters():
               if p.grad is not None:
                   total_norm += p.grad.data.norm(2).item() ** 2
           total_norm = total_norm ** 0.5
           grad_norms.append(total_norm)
           
           print(f"Iteration {i+1}, Gradient norm: {total_norm:.6f}")
           
           if i > 0:
               grad_diff = abs(grad_norms[i] - grad_norms[i-1])
               assert grad_diff < 1.0, f"Gradient changed too much: {grad_diff}"
           
           model.zero_grad()


class DistributionTests:
   @staticmethod
   def test_feature_distribution(device):
       print("\n=== Testing feature distribution ===")
       model = PolyphaseFeatureExtractor(
           in_channels=1,
           embed_dim=8,
           patch_size=(4,4,4)
       ).to(device).eval()
       
       B, C, D, H, W = 1, 1, 32, 32, 32
       input_vol = torch.randn(B, C, D, H, W, device=device)
       
       with torch.no_grad():
           features = model(input_vol)
       
       mean = features.mean().item()
       std = features.std().item()
       min_val = features.min().item()
       max_val = features.max().item()
       
       print(f"Feature statistics:")
       print(f"Mean: {mean:.6f}")
       print(f"Std: {std:.6f}")
       print(f"Min: {min_val:.6f}")
       print(f"Max: {max_val:.6f}")
       
       assert abs(mean) < 1.0, f"Mean too large: {mean}"
       assert 0.01 < std < 10.0, f"Std out of range: {std}"
       assert max_val - min_val < 50.0, f"Feature range too large: {max_val - min_val}"


class BatchConsistencyTests:
   @staticmethod
   def test_batch_invariance(device):
       print("\n=== Testing batch processing consistency ===")
       model = SETransformerR9Sampling(
           in_channels=1,
           num_transformer_blocks=2,
           num_heads=2,
           ff_hidden_dim=24,
           hidden_dim=6,
           feature_type='vector',
           patch_size=(4,4,4)
       ).to(device).eval()
       
       C, D, H, W = 1, 32, 32, 32
       single_input = torch.zeros(1, C, D, H, W, device=device)
       single_input[:,:,15:18,15:18,15:18] = 1.0
       single_target = torch.roll(single_input, shifts=(2,3,1), dims=(2,3,4))
       
       batch_size = 4
       batch_input = single_input.repeat(batch_size, 1, 1, 1, 1)
       batch_target = single_target.repeat(batch_size, 1, 1, 1, 1)
       
       with torch.no_grad():
           single_trans, single_aligned = model(single_input, single_target)
           batch_trans, batch_aligned = model(batch_input, batch_target)
       
       for i in range(batch_size):
           trans_diff = (single_trans - batch_trans[i]).abs().max().item()
           aligned_diff = (single_aligned - batch_aligned[i]).abs().max().item()
           
           print(f"Batch sample {i+1}:")
           print(f"Transform prediction diff: {trans_diff:.6f}")
           print(f"Aligned output diff: {aligned_diff:.6f}")
           
           assert trans_diff < 1e-5, f"Transform predictions inconsistent: {trans_diff}"
           assert aligned_diff < 1e-5, f"Aligned outputs inconsistent: {aligned_diff}"


# =============================================================================
# Runner combining all tests
# =============================================================================
def run_all_tests():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running tests on device: {device}")
    
    # 1) SETransformerR9Sampling forward test
    print("\n=== Running SETransformerR9Sampling Tests (Forward Pass Only) ===")
    SETransformerR9SamplingTests.test_forward_pass(device)

    # 2) LPSDown3D
    print("\n=== Running LPSDown3D Tests ===")
    LPSDown3DTests.test_shift_equivariance(device)
    LPSDown3DTests.test_rotation_equivariance(device)
    LPSDown3DTests.test_down_up_identity(device)
    LPSDown3DTests.test_rotation_parameter_prediction_equivariance(device)
    LPSDown3DTests.test_translation_parameter_prediction_equivariance(device)

    # 3) LPSUp3D
    print("\n=== Running LPSUp3D Tests (Explicit checks) ===")
    LPSUp3DTests.test_rotation_parameter_prediction_equivariance(device)
    LPSUp3DTests.test_translation_parameter_prediction_equivariance(device)

    # 4) FeatureComparator
    print("\n=== Running FeatureComparator Tests ===")
    FeatureComparatorTests.test_shift_equivariance(device)
    FeatureComparatorTests.test_rotation_equivariance(device)
    FeatureComparatorTests.test_rotation_parameter_prediction_equivariance(device)
    FeatureComparatorTests.test_translation_parameter_prediction_equivariance(device)

    # 5) PolyphaseFeatureExtractor
    print("\n=== Running PolyphaseFeatureExtractor Tests ===")
    PolyphaseFeatureExtractorTests.test_shift_equivariance(device)
    PolyphaseFeatureExtractorTests.test_rotation_equivariance(device)
    PolyphaseFeatureExtractorTests.test_rotation_parameter_prediction_equivariance(device)
    PolyphaseFeatureExtractorTests.test_translation_parameter_prediction_equivariance(device)

    # 6) ShiftEquivariantPositionalEncoder
    print("\n=== Running ShiftEquivariantPositionalEncoder Tests ===")
    ShiftEquivariantPositionalEncoderTests.test_shift_equivariance(device)
    ShiftEquivariantPositionalEncoderTests.test_rotation_equivariance(device)
    ShiftEquivariantPositionalEncoderTests.test_rotation_parameter_prediction_equivariance(device)
    ShiftEquivariantPositionalEncoderTests.test_translation_parameter_prediction_equivariance(device)

    # 7) CNNPolyphaseProcessor
    print("\n=== Running CNNPolyphaseProcessor Tests ===")
    CNNPolyphaseProcessorTests.test_shift_equivariance(device)
    CNNPolyphaseProcessorTests.test_rotation_equivariance(device)
    CNNPolyphaseProcessorTests.test_rotation_parameter_prediction_equivariance(device)
    CNNPolyphaseProcessorTests.test_translation_parameter_prediction_equivariance(device)

    # 8) TransformationPredictionHead
    print("\n=== Running TransformationPredictionHead Tests ===")
    TransformationPredictionHeadTests.test_shift_equivariance(device)
    TransformationPredictionHeadTests.test_rotation_equivariance(device)
    TransformationPredictionHeadTests.test_rotation_parameter_prediction_equivariance(device)
    TransformationPredictionHeadTests.test_translation_parameter_prediction_equivariance(device)

    # 9) SE3EquivariantTransformerBlock
    print("\n=== Running SE3EquivariantTransformerBlock Tests ===")
    SE3EquivariantTransformerBlockTests.test_shift_equivariance(device)
    SE3EquivariantTransformerBlockTests.test_rotation_equivariance(device)
    SE3EquivariantTransformerBlockTests.test_rotation_parameter_prediction_equivariance(device)
    SE3EquivariantTransformerBlockTests.test_translation_parameter_prediction_equivariance(device)

    # 10) Transformer
    print("\n=== Running Transformer Tests ===")
    TransformerTests.test_shift_equivariance(device)
    TransformerTests.test_rotation_equivariance(device)
    TransformerTests.test_rotation_parameter_prediction_equivariance(device)
    TransformerTests.test_translation_parameter_prediction_equivariance(device)

    # 11) Full SETransformerR9Sampling Pipeline
    print("\n=== Running Full SETransformerR9Sampling Pipeline Tests ===")
    FullSETransformerR9SamplingPipelineTests.test_pipeline(device)

    # 12) Some Additionanl Tests
    print("\n=== Running Edge Case Tests ===")
    EdgeCaseTests.test_zero_volume_input(device)
    EdgeCaseTests.test_extreme_rotations(device)
   
    print("\n=== Running Numerical Stability Tests ===")
    NumericalStabilityTests.test_input_scale_stability(device)
    NumericalStabilityTests.test_gradient_stability(device)
   
    print("\n=== Running Distribution Tests ===")
    DistributionTests.test_feature_distribution(device)
   
    print("\n=== Running Batch Consistency Tests ===")
    BatchConsistencyTests.test_batch_invariance(device)
    
    print("\nAll tests completed successfully.")

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    set_deterministic()
    torch.set_default_dtype(torch.float64)
    run_all_tests()
