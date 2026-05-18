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
from .layers.self_attention_euler import SelfAttentionModule

print(torch.__version__)  
print(torch.version.cuda)  


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
    def __init__(self, in_channels, hidden_dim=32, out_dim=6):
    # def __init__(self, in_channels, hidden_dim=32, out_dim=12):
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
        
        # rot_r9 = out[:, :9]
        # trans = out[:, 9:]
        # return torch.cat([rot_r9, trans], dim=1)
        
        rot = torch.tanh(out[:, :3]) * math.pi
        trans = torch.tanh(out[:, 3:])
        return torch.cat([rot, trans], dim=1)

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
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, rot_angles, translations):
    # def forward(self, x, r9_params, translations):
        B, C, D, H, W = x.shape
        device = x.device
        R = kornia.geometry.axis_angle_to_rotation_matrix(rot_angles)[:, :3, :3]  
        # R = r9_to_matrix(r9_params)
        R_inv = R.transpose(1, 2)  
        t_inv = -torch.bmm(R_inv, translations.unsqueeze(2)).squeeze(2)  

        transform_matrix = torch.zeros(B, 3, 4, device=device)
        transform_matrix[:, :, :3] = R_inv
        transform_matrix[:, :, 3] = t_inv

        grid = F.affine_grid(transform_matrix, x.size(), align_corners=False)
        transformed = F.grid_sample(
            x, grid, mode='nearest', padding_mode='border', align_corners=False
        )
        return transformed


###############################################################################
# SETransformerEulerSampling Module
###############################################################################
class SETransformerEulerSampling(nn.Module):
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
        print("Euler sampling revised")
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
        
        transform_predictions = []
        aligned_volumes = []
        
        aligned_input = input
        for i, block in enumerate(self.transformer_blocks):
            combined_features = block(combined_features, combined_pos_emb, attn_mask)
            
            transformation_pred = self.trans_head(combined_features)
            transform_predictions.append(transformation_pred)
            
            aligned_input = self.transformer(
                aligned_input,
                transformation_pred[:, :3],  
                transformation_pred[:, 3:]  
            )
            aligned_volumes.append(aligned_input)
        
        return transform_predictions, aligned_input, aligned_volumes