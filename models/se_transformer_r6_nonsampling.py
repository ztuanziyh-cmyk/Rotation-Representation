import torch
import torch.nn as nn
import kornia
import math
import torch.nn.functional as F
from torch.autograd import Function
from .layers.self_attention_r6 import SelfAttentionModule

def r6_to_matrix(r6):
    if r6.size(-1) != 6:
        raise ValueError(f"r6_to_matrix expects last dimension=6, got {r6.size(-1)}")
    with torch.autocast(device_type='cuda', enabled=False):
        r6 = r6.float()
        eps = 1e-7
    
        v1 = torch.nn.functional.normalize(r6[..., :3], dim=-1, eps=eps)
        v2 = r6[..., 3:6]
        v2_proj = torch.sum(v2 * v1, dim=-1, keepdim=True) * v1
        v2 = torch.nn.functional.normalize(v2 - v2_proj, dim=-1, eps=eps)
        v3 = torch.cross(v1, v2, dim=-1)
        R = torch.stack([v1, v2, v3], dim=-1)

    return R


# =============================
# Polyphase Anchoring Wrapper
# =============================
class PolyOrderModule3D(nn.Module):
    def __init__(self, patch_size, norm=2, invariance=False) :
        super().__init__()
        self.patch_size = patch_size
        self.norm = norm
        self.invariance = invariance

    def forward(self, x):
        return PolyOrder3D.apply(x, self.patch_size, self.norm, self.invariance)

# =============================
# Polyphase Anchoring Function
# =============================

class PolyOrder3D(Function):
    @staticmethod
    def forward(ctx, x, patch_size, norm=2, invariance=False):
        device = x.device
        B, C, D, H, W = x.shape  # Input dimensions: Batch, Channels, Depth, Height, Width
        pd, ph, pw = patch_size  # Patch sizes for Depth, Height, Width
        ##print(f"Input shape: {x.shape}, Patch size: {patch_size}")
        # Calculate the grid size (number of patches along each dimension)
        gd, gh, gw = D // pd, H // ph, W // pw  # Grid dimensions, number of patches along each axis
        grid_size = (gd, gh, gw)

        # Reshape the input tensor into patches representing polyphase components
        tmp = x.view(B, C, gd, pd, gh, ph, gw, pw)
        ##print(f"Reshaped tensor for polyphase components: {tmp.shape}")
        # Rearrange dimensions to group patches together: (B, C, gd, gh, gw, pd, ph, pw)
        tmp = tmp.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()

        # Flatten the patches into a single dimension: (B, C, total_patches, patch_volume)
        tmp = tmp.view(B, C, gd * gh * gw, pd * ph * pw)

        # Calculate the vector norm over patches to find their magnitudes
        # This will compute the L2 norm of each polyphase component
        norm_vals = torch.linalg.vector_norm(tmp, dim=(2, 3))
        # del tmp  # Delete temporary tensor to free up memory
        ##print("Norm values shape:", norm_vals.shape)

        # Find the index of the patch with the largest norm for each batch element
        idx = torch.argmax(norm_vals, dim=1).int()
        ##print("Max norm indices:", idx)
        # Calculate the shifts in depth, height, and width based on the index
        pz = (idx // (gh * gw)).int()  # Depth shift (z-axis)
        remainder = idx % (gh * gw)
        py = (remainder // gw).int()   # Height shift (y-axis)
        px = (remainder % gw).int()    # Width shift (x-axis)
        ##print(f"Shift values - px: {px}, py: {py}, pz: {pz}")
        # Calculate padding required along z, y, x to handle circular shifts
        lz, ly, lx = pd - 1, ph - 1, pw - 1

        # Apply circular padding to the input tensor along all three dimensions
        x_padded = F.pad(x, (0, lx, 0, ly, 0, lz), mode="circular").float()
        ##print("Padded input shape:", x_padded.shape)
        # Create 3D affine transformation matrix for each batch
        theta = torch.zeros((B, 3, 4), requires_grad=False).to(device).float()

        # Set scaling factors to 1 (no scaling along any axis)
        theta[:, 0, 0] = 1  # Scale along the x-axis (width)
        theta[:, 1, 1] = 1  # Scale along the y-axis (height)
        theta[:, 2, 2] = 1  # Scale along the z-axis (depth)

        # Set translation terms based on calculated px, py, pz
        # The translations need to be normalized to the range [-1, 1] for affine_grid
        theta[:, 0, 3] = px * 2 / (W + lx)  # Translation along the x-axis
        theta[:, 1, 3] = py * 2 / (H + ly)  # Translation along the y-axis
        theta[:, 2, 3] = pz * 2 / (D + lz)  # Translation along the z-axis
        ##print("Theta matrix for affine transformation:", theta)
        # Save tensors for backward pass
        ctx.save_for_backward(x, theta, torch.tensor(patch_size), torch.tensor(norm))

        # Create affine grid for sampling, which defines the new coordinates
        grid = F.affine_grid(theta, x_padded.size(), align_corners=False)
        ##print("Affine grid shape:", grid.shape)
        # Sample the input tensor using grid_sample to apply the 3D shift
        x_transformed = F.grid_sample(x_padded, grid, mode="nearest", align_corners=False)
        ##print("Transformed x shape after grid_sample:", x_transformed.shape)
        # Since we applied padding earlier, we now crop the tensor back to its original size
        x_transformed = x_transformed[:, :, :D, :H, :W]  # Crop to original depth, height, and width

        return x_transformed

    @staticmethod
    def backward(ctx, grad_output):
        # Recover the saved tensors from the forward pass
        x, theta, patch_size, norm = ctx.saved_tensors
        B, C, D, H, W = x.shape

        # Create an affine grid for the backward pass
        grid = F.affine_grid(theta, (B, C, D, H, W), align_corners=False)

        # Use grid_sample to apply the affine transformation on the gradient
        grad_input = F.grid_sample(grad_output, grid, mode="nearest", align_corners=False, padding_mode="zeros")

        # Return the gradient for x, and None for the other inputs as they don't require gradients
        return grad_input, None, None, None, None  # Only return gradients for 'x', others are None

# =============================
# Polyphase Patch Embedding
# =============================
class PolyPatchEmbed3D(nn.Module):
    def __init__(self, patch_size, in_chans, out_chans, norm_layer=None):
        super().__init__()
        self.poly_order_module = PolyOrderModule3D(patch_size=patch_size)
        self.proj = nn.Conv3d(
            in_chans, out_chans, kernel_size=patch_size, stride=patch_size, padding=0, padding_mode="circular"
        )
        self.norm = norm_layer(out_chans) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.poly_order_module(x)
        x = self.proj(x)
        x = self.norm(x)
        return x


class PosConv3D(nn.Module):
    def __init__(self, in_chans, out_chans, patch_size=(4, 4, 4), use_polyphase=True):
        super(PosConv3D, self).__init__()
        self.use_polyphase = use_polyphase
        if self.use_polyphase:
            self.poly_order_module = PolyOrderModule3D(patch_size=patch_size)

        self.conv = nn.Conv3d(
            in_chans, out_chans, kernel_size=3, padding=1, padding_mode='circular', groups=in_chans
        )

    def forward(self, x):
        if self.use_polyphase:
            x = self.poly_order_module(x)
        x = self.conv(x)
        return x


# =============================
# Feature Comparator
# =============================
class FeatureComparator(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        
        self.target_processor = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, 1),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU()
        )
        
        self.input_processor = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, 1),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU()
        )
        
        self.compare_net = nn.Sequential(
            nn.Conv3d(hidden_dim * 2 + in_channels * 2, hidden_dim, 1),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, padding_mode='circular'),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU()
        )
        
        self.attention = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Conv3d(hidden_dim, in_channels, 1)
        
    def forward(self, target_features, input_features):
        diff = target_features - input_features
        prod = target_features * input_features
        
        target_processed = self.target_processor(target_features)
        input_processed = self.input_processor(input_features)
        
        combined = torch.cat([target_processed, input_processed, diff, prod], dim=1)
        
        features = self.compare_net(combined)
        
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        output = self.output_proj(attended_features)
        
        return output


# ===========================
# Enhanced Cross-Scale Feature Extractor
# ===========================
class CrossScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return x * self.attention(x)


class PolyphaseFeatureExtractor(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim 
        pd, ph, pw = patch_size
        
        self.poly_anchor = PolyOrderModule3D(patch_size=patch_size)
        
        self.cnn_extract = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim//2, 3, 
                     stride=1, padding=1, padding_mode='circular'),
            norm_layer(embed_dim//2) if norm_layer else nn.Identity(),
            nn.GELU(),
            nn.Conv3d(embed_dim//2, embed_dim, 3, 
                     stride=1, padding=1, padding_mode='circular'),
            norm_layer(embed_dim) if norm_layer else nn.Identity(),
            nn.GELU()
        )
        
        self.scales = nn.ModuleList([
            nn.Identity(),
            nn.Sequential(
                nn.Conv3d(embed_dim, embed_dim, 3, 
                         stride=2, padding=1, padding_mode='circular'),
                norm_layer(embed_dim) if norm_layer else nn.Identity(),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv3d(embed_dim, embed_dim, 3, 
                         stride=4, padding=1, padding_mode='circular'),
                norm_layer(embed_dim) if norm_layer else nn.Identity(),
                nn.GELU()
            )
        ])
        
        self.cross_attention = nn.ModuleList([
            CrossScaleBlock(embed_dim) for _ in range(3)
        ])
        
        self.scale_refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(embed_dim, embed_dim, 3, 
                         padding=1, padding_mode='circular'),
                norm_layer(embed_dim) if norm_layer else nn.Identity(),
                nn.GELU()
            ) for _ in range(3)
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv3d(embed_dim * 3, embed_dim * 2, 1),
            norm_layer(embed_dim * 2) if norm_layer else nn.Identity(),
            nn.GELU(),
            nn.Conv3d(embed_dim * 2, embed_dim, 1),
            norm_layer(embed_dim) if norm_layer else nn.Identity(),
            nn.GELU()
        )
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        expected_D, expected_H, expected_W = D//self.patch_size[0], H//self.patch_size[1], W//self.patch_size[2]
        
        x = self.poly_anchor(x)
        x = self.cnn_extract(x)
        
        base_size = (expected_D, expected_H, expected_W)
        
        multi_scale_features = []
        
        for scale in self.scales:
            scale_feat = scale(x)
            
            if scale_feat.shape[2:] != base_size:
                scale_feat = F.interpolate(
                    scale_feat, 
                    size=base_size,
                    mode='trilinear',
                    align_corners=False
                )
            multi_scale_features.append(scale_feat)
        
        attended_features = []
        for feat, attn, refine in zip(multi_scale_features, 
                                     self.cross_attention,
                                     self.scale_refine):
            attended = attn(feat)
            refined = refine(attended)
            attended_features.append(refined)
        
        output = torch.cat(attended_features, dim=1)
        output = self.fusion(output)
        
        assert output.shape == (B, self.embed_dim, *base_size), \
            f"Shape mismatch: got {output.shape}, expected {(B, self.embed_dim, *base_size)}"
        
        return output

# =============================
# ShiftEquivariantPositionalEncoder
# =============================
class ShiftEquivariantPositionalEncoder(nn.Module):
    def __init__(self, hidden_dim, patch_size=(4, 4, 4)):
        super().__init__()
        self.patch_size = patch_size
        
        
        mid_dim = hidden_dim // 2
        
        self.pos_embed = nn.Sequential(
            nn.Conv3d(3, mid_dim, 3, padding=1, padding_mode='circular'),
            nn.InstanceNorm3d(mid_dim),
            nn.GELU(),
            nn.Conv3d(mid_dim, hidden_dim, 1),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU()
        )
        
        self.spatial_mixer = nn.Conv3d(hidden_dim, hidden_dim, 1)
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        device = x.device
        
        z = torch.linspace(-1, 1, D, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        
        zz, yy, xx = torch.meshgrid(z, y, x_coords, indexing='ij')
        coords = torch.stack((xx, yy, zz), dim=0)
        
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1, -1)
        pos_embed = self.pos_embed(coords)
        pos_embed = self.spatial_mixer(pos_embed)
        
        return pos_embed

# =============================
# Transformation Prediction Head
# =============================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, padding_mode='circular')
        self.norm1 = nn.InstanceNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, padding_mode='circular')
        self.norm2 = nn.InstanceNorm3d(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)

class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels // 4, 1)
        self.norm1 = nn.InstanceNorm3d(channels // 4)
        self.conv2 = nn.Conv3d(channels // 4, 1, 1)
        self.activation = nn.GELU()

    def forward(self, x):
        feat = self.activation(self.norm1(self.conv1(x)))
        att = torch.sigmoid(self.conv2(feat))
        return x * att

class TransformationPredictionHead(nn.Module):
    # def __init__(self, in_channels, hidden_dim, out_dim=6):
    # def __init__(self, in_channels, hidden_dim=32, out_dim=12):
    def __init__(self, in_channels, hidden_dim=32, out_dim=9):
        super().__init__()
        
        self.spatial_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, hidden_dim, (1, 1, 5), padding=(0, 0, 2), padding_mode='circular'),
                nn.InstanceNorm3d(hidden_dim),
                nn.GELU(),
                ResidualBlock(hidden_dim),
                nn.Conv3d(hidden_dim, hidden_dim, (1, 1, 3), padding=(0, 0, 1), padding_mode='circular', groups=hidden_dim),
                nn.InstanceNorm3d(hidden_dim),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv3d(in_channels, hidden_dim, (1, 5, 1), padding=(0, 2, 0), padding_mode='circular'),
                nn.InstanceNorm3d(hidden_dim),
                nn.GELU(),
                ResidualBlock(hidden_dim),
                nn.Conv3d(hidden_dim, hidden_dim, (1, 3, 1), padding=(0, 1, 0), padding_mode='circular', groups=hidden_dim),
                nn.InstanceNorm3d(hidden_dim),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv3d(in_channels, hidden_dim, (5, 1, 1), padding=(2, 0, 0), padding_mode='circular'),
                nn.InstanceNorm3d(hidden_dim),
                nn.GELU(),
                ResidualBlock(hidden_dim),
                nn.Conv3d(hidden_dim, hidden_dim, (3, 1, 1), padding=(1, 0, 0), padding_mode='circular', groups=hidden_dim),
                nn.InstanceNorm3d(hidden_dim),
                nn.GELU()
            )
        ])

        self.dim_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(hidden_dim, hidden_dim // 2, 1),
                nn.InstanceNorm3d(hidden_dim // 2),
                nn.GELU(),
                ResidualBlock(hidden_dim // 2),
                SpatialAttention(hidden_dim // 2)
            ) for _ in range(3)
        ])

        self.dim_pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(4),
                ResidualBlock(hidden_dim // 2)
            ) for _ in range(3)
        ])

        pool_size = 4 * 4 * 4 
        combined_features = 3 * (hidden_dim // 2) * pool_size
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(combined_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )

        # self.rotation_predictor = nn.Linear(hidden_dim // 2, 3)
        # self.rotation_predictor = nn.Linear(hidden_dim // 2, 9)
        self.rotation_predictor = nn.Linear(hidden_dim // 2, 6)
        self.translation_predictor = nn.Linear(hidden_dim // 2, 3)

        self._init_weights()

    def forward(self, x):
        dim_features = []
        for i, extractor in enumerate(self.spatial_extractors):
            features = extractor(x)
            dim_features.append(features)

        processed_features = []
        for i, (feat, processor, pool) in enumerate(zip(dim_features, self.dim_processors, self.dim_pools)):
            processed = processor(feat)
            
            pooled = pool(processed)
            processed_features.append(pooled)

        flattened_features = []
        for i, feat in enumerate(processed_features):
            flat = feat.flatten(1)
            flattened_features.append(flat)

        combined = torch.cat(flattened_features, dim=1)

        shared_features = self.shared_mlp(combined)

        # rot = torch.tanh(self.rotation_predictor(shared_features)) * math.pi
        # trans = torch.tanh(self.translation_predictor(shared_features))
        
        # predictions = torch.cat([rot, trans], dim=1)
        
        # rot_r9 = self.rotation_predictor(shared_features)
        # rot_matrix = r9_to_matrix(rot_r9)
        # rot_r9_valid = rot_matrix.reshape(*rot_matrix.shape[:-2], 9)  

        # rot_r6 = self.rotation_predictor(shared_features)
        # rot_matrix = r6_to_matrix(rot_r6)
        # rot_r6_valid = rot_matrix.reshape(*rot_matrix.shape[:-2], 6)
        rot_r6_valid = self.rotation_predictor(shared_features)
        trans = torch.tanh(self.translation_predictor(shared_features))
        
        predictions = torch.cat([rot_r9_valid, trans], dim=1)
        return predictions

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# =============================
# SE3 Equivariant Transformer Block
# =============================
class SE3EquivariantTransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads, ff_hidden_dim, feature_type='vector'):
        super().__init__()
        self.self_attn = SelfAttentionModule(
            in_channels=in_channels,
            num_heads=num_heads,
            feature_type=feature_type
        )
        self.norm1 = nn.InstanceNorm3d(in_channels)
        self.ffn = nn.Sequential(
            nn.Conv3d(in_channels, ff_hidden_dim, 1),
            nn.GELU(),
            nn.Conv3d(ff_hidden_dim, in_channels, 1)
        )
        self.norm2 = nn.InstanceNorm3d(in_channels)

    def forward(self, x, pos_emb, mask=None):
        # x: Combined features
        # pos_emb: Combined positional embeddings
        attn_out = self.self_attn(x, pos_emb, mask)
        x = x + attn_out
        x = self.norm1(x)

        ff_out = self.ffn(x)
        x = x + ff_out
        x = self.norm2(x)

        return x



# =============================
# Transformer
# =============================
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

    # def forward(self, x, rot_angles, translations):
    # def forward(self, x, r9_params, translations):
    def forward(self, x, r6_params, translations):
        B, C, D, H, W = x.shape
        device = x.device

        # R = kornia.geometry.axis_angle_to_rotation_matrix(rot_angles)[:, :3, :3]  
        # R = r9_to_matrix(r9_params)
        R = r6_to_matrix(r6_params)
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


# =============================
# SETransformerR6NonSampling Module
# =============================
class SETransformerR6NonSampling(nn.Module):   
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
        print("R6 non-sampling")
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
            norm_layer=nn.InstanceNorm3d
        )

        self.feature_comparator = FeatureComparator(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim*2
        )
        
        self.pos_combiner = nn.Sequential(
            nn.Conv3d(hidden_dim * 2, hidden_dim, 1),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU()
        )

        self.pos_encoder = ShiftEquivariantPositionalEncoder(
            hidden_dim=hidden_dim,
            patch_size=patch_size
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
            
            # aligned_input = self.transformer(
            #     aligned_input,
            #     transformation_pred[:, :9],
            #     transformation_pred[:, 9:]
            # )

            aligned_input = self.transformer(
                aligned_input,
                transformation_pred[:, :6],
                transformation_pred[:, 6:]
            )
        return transformation_pred, aligned_input

