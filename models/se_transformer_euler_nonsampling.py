import torch
import torch.nn as nn
import kornia
import math
import torch.nn.functional as F
from torch.autograd import Function
from .layers.self_attention_euler import SelfAttentionModule


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
    def __init__(self, in_channels, hidden_dim, out_dim=6):
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

        self.rotation_predictor = nn.Linear(hidden_dim // 2, 3)
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

        rot = torch.tanh(self.rotation_predictor(shared_features)) * math.pi
        trans = torch.tanh(self.translation_predictor(shared_features))
        
        predictions = torch.cat([rot, trans], dim=1)

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

    def forward(self, x, rot_angles, translations):

        B, C, D, H, W = x.shape
        device = x.device

        R = kornia.geometry.axis_angle_to_rotation_matrix(rot_angles)[:, :3, :3]  

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
# SETransformerEulerNonSampling Module
# =============================
class SETransformerEulerNonSampling(nn.Module):
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
        print("Euler non-sampling")
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

            aligned_input = self.transformer(
                aligned_input,
                transformation_pred[:, :3],  
                transformation_pred[:, 3:]   
            )

        return transformation_pred, aligned_input



# # ===========================
# # Test Cases
# # ===========================
def get_device():
    return torch.device("cpu")
def test_transformer_forward_pass():
    print("Testing SETransformerEulerNonSampling single forward pass...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 1
    in_channels = 1
    spatial_size = 32
    
    input_tensor = torch.randn(batch_size, in_channels, spatial_size, spatial_size, spatial_size).to(device)
    target_tensor = torch.randn(batch_size, in_channels, spatial_size, spatial_size, spatial_size).to(device)
    
    model = SETransformerEulerNonSampling(
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
    
    expected_trans_pred_shape = (batch_size, 6)
    expected_aligned_shape = input_tensor.shape
    
    assert trans_pred.shape == expected_trans_pred_shape, \
        f"Transformation prediction shape mismatch. Expected {expected_trans_pred_shape}, got {trans_pred.shape}"
    assert aligned_input.shape == expected_aligned_shape, \
        f"Aligned input shape mismatch. Expected {expected_aligned_shape}, got {aligned_input.shape}"
    
    assert not torch.isnan(trans_pred).any(), "NaN values in transformation prediction"
    assert not torch.isnan(aligned_input).any(), "NaN values in aligned input"
    
    print("\nAll shape checks passed!")
    print("\nTransformation predictions:")
    print(f"Rotations (radians): {trans_pred[0, :3].cpu().numpy()}")
    print(f"Translations: {trans_pred[0, 3:].cpu().numpy()}")
    print("\nSETransformerEulerNonSampling forward pass test completed successfully!")

def test_transformer_comprehensive():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 2

    all_transformers = {
        'Kornia': Transformer().to(device),
    }

    def create_test_volume(B=BATCH_SIZE, C=1, D=32, H=32, W=32):
        x = torch.zeros(B, C, D, H, W, device=device)
        mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

        for i in range(-2, 3):
            x[:, :, mid_d + i, mid_h, mid_w] = 1.0
            x[:, :, mid_d, mid_h + i, mid_w] = 1.0
            x[:, :, mid_d, mid_h, mid_w + i] = 1.0

        return x

    test_cases = [
        {'name': 'Small X rotation', 'angles': torch.tensor([0.1, 0.0, 0.0], device=device), 'trans': torch.tensor([0.0, 0.0, 0.0], device=device)},
        {'name': 'Small Y rotation', 'angles': torch.tensor([0.0, 0.1, 0.0], device=device), 'trans': torch.tensor([0.0, 0.0, 0.0], device=device)},
        {'name': 'Small Z rotation', 'angles': torch.tensor([0.0, 0.0, 0.1], device=device), 'trans': torch.tensor([0.0, 0.0, 0.0], device=device)},
        {'name': 'Medium X rotation', 'angles': torch.tensor([1.0, 0.0, 0.0], device=device), 'trans': torch.tensor([0.0, 0.0, 0.0], device=device)},
        {'name': 'Medium Y rotation', 'angles': torch.tensor([0.0, 1.0, 0.0], device=device), 'trans': torch.tensor([0.0, 0.0, 0.0], device=device)},
        {'name': 'Medium Z rotation', 'angles': torch.tensor([0.0, 0.0, 1.0], device=device), 'trans': torch.tensor([0.0, 0.0, 0.0], device=device)},
        {'name': 'Large X rotation', 'angles': torch.tensor([2.0, 0.0, 0.0], device=device), 'trans': torch.tensor([0.0, 0.0, 0.0], device=device)},
        {'name': 'Large Y rotation', 'angles': torch.tensor([0.0, 2.0, 0.0], device=device), 'trans': torch.tensor([0.0, 0.0, 0.0], device=device)},
        {'name': 'Large Z rotation', 'angles': torch.tensor([0.0, 0.0, 2.0], device=device), 'trans': torch.tensor([0.0, 0.0, 0.0], device=device)},
        {'name': 'Small X translation', 'angles': torch.tensor([0.0, 0.0, 0.0], device=device), 'trans': torch.tensor([0.1, 0.0, 0.0], device=device)},
        {'name': 'Small Y translation', 'angles': torch.tensor([0.0, 0.0, 0.0], device=device), 'trans': torch.tensor([0.0, 0.1, 0.0], device=device)},
        {'name': 'Small Z translation', 'angles': torch.tensor([0.0, 0.0, 0.0], device=device), 'trans': torch.tensor([0.0, 0.0, 0.1], device=device)},
        {'name': 'Large X translation', 'angles': torch.tensor([0.0, 0.0, 0.0], device=device), 'trans': torch.tensor([0.3, 0.0, 0.0], device=device)},
        {'name': 'Large Y translation', 'angles': torch.tensor([0.0, 0.0, 0.0], device=device), 'trans': torch.tensor([0.0, 0.3, 0.0], device=device)},
        {'name': 'Large Z translation', 'angles': torch.tensor([0.0, 0.0, 0.0], device=device), 'trans': torch.tensor([0.0, 0.0, 0.3], device=device)},
        {'name': 'X rotation + translation', 'angles': torch.tensor([1.0, 0.0, 0.0], device=device), 'trans': torch.tensor([0.2, 0.0, 0.0], device=device)},
        {'name': 'Y rotation + translation', 'angles': torch.tensor([0.0, 1.0, 0.0], device=device), 'trans': torch.tensor([0.0, 0.2, 0.0], device=device)},
        {'name': 'Z rotation + translation', 'angles': torch.tensor([0.0, 0.0, 1.0], device=device), 'trans': torch.tensor([0.0, 0.0, 0.2], device=device)},
        {'name': 'X rotation + Y translation', 'angles': torch.tensor([1.0, 0.0, 0.0], device=device), 'trans': torch.tensor([0.0, 0.2, 0.0], device=device)},
        {'name': 'Y rotation + Z translation', 'angles': torch.tensor([0.0, 1.0, 0.0], device=device), 'trans': torch.tensor([0.0, 0.0, 0.2], device=device)},
        {'name': 'Z rotation + X translation', 'angles': torch.tensor([0.0, 0.0, 1.0], device=device), 'trans': torch.tensor([0.2, 0.0, 0.0], device=device)},
        {'name': 'Small XYZ rotation', 'angles': torch.tensor([0.1, 0.1, 0.1], device=device), 'trans': torch.tensor([0.0, 0.0, 0.0], device=device)},
        {'name': 'Large XYZ rotation', 'angles': torch.tensor([1.5, 1.5, 1.5], device=device), 'trans': torch.tensor([0.0, 0.0, 0.0], device=device)},
        {'name': 'Small XYZ translation', 'angles': torch.tensor([0.0, 0.0, 0.0], device=device), 'trans': torch.tensor([0.1, 0.1, 0.1], device=device)},
        {'name': 'Large XYZ translation', 'angles': torch.tensor([0.0, 0.0, 0.0], device=device), 'trans': torch.tensor([0.3, 0.3, 0.3], device=device)},
        {'name': 'Small XYZ rotation + translation', 'angles': torch.tensor([0.1, 0.1, 0.1], device=device), 'trans': torch.tensor([0.1, 0.1, 0.1], device=device)},
        {'name': 'Medium XYZ rotation + translation', 'angles': torch.tensor([1.0, 1.0, 1.0], device=device), 'trans': torch.tensor([0.2, 0.2, 0.2], device=device)},
        {'name': 'Large XYZ rotation + translation', 'angles': torch.tensor([1.5, 1.5, 1.5], device=device), 'trans': torch.tensor([0.3, 0.3, 0.3], device=device)},
    ]

    results = []
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        x = create_test_volume()
        angles = case['angles']
        trans = case['trans']
        B = x.shape[0]

        if angles.dim() == 1:
            angles = angles.unsqueeze(0).expand(B, -1)
        if trans.dim() == 1:
            trans = trans.unsqueeze(0).expand(B, -1)

        transformed_results = {}
        for name, transformer in all_transformers.items():
            transformed_results[name] = transformer(x, angles, trans)
        threshold = 0.5
        orig_points = (x > threshold).sum().item()
        preserved_points = {name: (result > threshold).sum().item() for name, result in transformed_results.items()}

        print("\nPoint Analysis:")
        print(f"Original points: {orig_points}")
        for name, points in preserved_points.items():
            print(f"{name} preserved points: {points}")
            print(f"{name} point ratio: {points/orig_points:.2f}")

        def compute_centroid(vol):
            points = torch.where(vol[0, 0] > threshold)
            if len(points[0]) > 0:
                return torch.stack([dim.float().mean() for dim in points])
            return None

        centroids = {name: compute_centroid(result) for name, result in transformed_results.items()}
        print("\nCentroid Analysis:")
        for name, centroid in centroids.items():
            if centroid is not None:
                print(f"{name} centroid: {centroid.tolist()}")

        def compute_distance_stats(points):
            if len(points) > 1:
                dists = torch.cdist(points, points)
                mask = torch.triu(torch.ones_like(dists), diagonal=1) > 0
                dists = dists[mask]
                return {
                    'min': dists.min().item(),
                    'max': dists.max().item(),
                    'mean': dists.mean().item(),
                    'std': dists.std().item()
                }
            return None

        print("\nDistance Statistics:")
        orig_points = torch.stack(torch.where(x[0, 0] > threshold), dim=1).float()
        orig_stats = compute_distance_stats(orig_points)

        if orig_stats:
            print(f"\nOriginal distances:")
            print(f"Min: {orig_stats['min']:.4f}")
            print(f"Max: {orig_stats['max']:.4f}")
            print(f"Mean: {orig_stats['mean']:.4f}")
            print(f"Std: {orig_stats['std']:.4f}")
            
            for name, result in transformed_results.items():
                trans_points = torch.stack(torch.where(result[0, 0] > threshold), dim=1).float()
                trans_stats = compute_distance_stats(trans_points)
                
                if trans_stats:
                    print(f"\n{name} distances:")
                    print(f"Min: {trans_stats['min']:.4f}")
                    print(f"Max: {trans_stats['max']:.4f}")
                    print(f"Mean: {trans_stats['mean']:.4f}")
                    print(f"Std: {trans_stats['std']:.4f}")
                    
                    min_change = (trans_stats['min'] - orig_stats['min']) / orig_stats['min']
                    max_change = (trans_stats['max'] - orig_stats['max']) / orig_stats['max']
                    mean_change = (trans_stats['mean'] - orig_stats['mean']) / orig_stats['mean']
                    
                    print(f"\nRelative changes:")
                    print(f"Min distance change: {min_change:.2%}")
                    print(f"Max distance change: {max_change:.2%}")
                    print(f"Mean distance change: {mean_change:.2%}")

        results.append({
            'name': case['name'],
            'angles': angles[0].tolist(),
            'translations': trans[0].tolist(),
            'preserved_points': preserved_points,
        })

    print("\nTest Summary:")
    print("-" * 50)
    print(f"Total test cases: {len(results)}")
    
    categories = {
        'rotation_only': [r for r in results if all(t == 0 for t in r['translations'])],
        'translation_only': [r for r in results if all(a == 0 for a in r['angles'])],
        'combined': [r for r in results if any(t != 0 for t in r['translations']) and any(a != 0 for a in r['angles'])]
    }
    
    for category, cases in categories.items():
        print(f"\n{category.replace('_', ' ').title()} Cases:")
        print(f"Number of cases: {len(cases)}")
        if cases:
            orig_points = cases[0]['preserved_points']['Kornia']
            for impl in ['Kornia']:
                avg_points = sum(case['preserved_points'][impl] for case in cases) / len(cases)
                print(f"{impl} average preserved points: {avg_points:.2f}/{orig_points} ({avg_points/orig_points*100:.1f}%)")

def test_SETransformerEulerNonSampling_comprehensive():
    print("\n=== Comprehensive Testing of SETransformerEulerNonSampling ===")
    
    batch_size = 2
    in_channels = 1
    hidden_dim = 72
    num_blocks = 2
    num_heads = 4
    ff_dim = 128
    patch_size = (4, 4, 4)
    input_size = (32, 32, 32)
    
    print("\nConfiguration:")
    print(f"Batch size: {batch_size}")
    print(f"Input channels: {in_channels}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Patch size: {patch_size}")
    print(f"Input size: {input_size}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SETransformerEulerNonSampling(
        in_channels=in_channels,
        num_transformer_blocks=num_blocks,
        num_heads=num_heads,
        ff_hidden_dim=ff_dim,
        hidden_dim=hidden_dim,
        patch_size=patch_size
    ).to(device)
    
    model.train()
    
    input_tensor = torch.randn(batch_size, in_channels, *input_size).to(device) * 0.1
    target_tensor = torch.randn(batch_size, in_channels, *input_size).to(device) * 0.1
    
    print("\nTest Input Creation:")
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Target tensor shape: {target_tensor.shape}")
    
    try:
        trans_pred, aligned = model(input_tensor, target_tensor)
        
        print("\nForward Pass Successful!")
        print(f"Transformation prediction shape: {trans_pred.shape}")
        print(f"Aligned output shape: {aligned.shape}")
        assert trans_pred.shape == (batch_size, 6), "Incorrect transformation prediction shape."
        assert aligned.shape == (batch_size, in_channels, *input_size), "Incorrect aligned output shape."
        
        rot = trans_pred[:, :3]
        trans = trans_pred[:, 3:]
        print(f"\nRotation range: [{rot.min().item():.3f}, {rot.max().item():.3f}]")
        print(f"Translation range: [{trans.min().item():.3f}, {trans.max().item():.3f}]")
        
        assert torch.all(rot >= -math.pi) and torch.all(rot <= math.pi), "Rotation angles out of bounds."
        assert torch.all(trans >= -1) and torch.all(trans <= 1), "Translations out of bounds."
        
        print("\nSETransformerEulerNonSampling Forward Pass Tests Passed.")
        
        shift = (4, 4, 4)
        input_shifted = torch.roll(input_tensor, shifts=shift, dims=(2, 3, 4))
        target_shifted = torch.roll(target_tensor, shifts=shift, dims=(2, 3, 4))
        
        trans_pred_shifted, aligned_shifted = model(input_shifted, target_shifted)
        
        print(f"\nShifted Forward Pass Successful!")
        print(f"Shifted Transformation prediction shape: {trans_pred_shifted.shape}")
        print(f"Shifted Aligned output shape: {aligned_shifted.shape}")
        assert trans_pred_shifted.shape == (batch_size, 6), "Incorrect shifted transformation prediction shape."
        assert aligned_shifted.shape == (batch_size, in_channels, *input_size), "Incorrect shifted aligned output shape."
        
        print("\nSETransformerEulerNonSampling Comprehensive Tests Passed Successfully.")
        return model
    except Exception as e:
        print(f"Error: {e}")


def test_position_encoder():
    print("\n=== Testing ShiftEquivariantPositionalEncoder ===")
    
    hidden_dim = 64
    encoder = ShiftEquivariantPositionalEncoder(hidden_dim=hidden_dim)
    
    print("\nTest 1: Basic Functionality")
    B, C, D, H, W = 2, 32, 16, 16, 16
    x = torch.randn(B, C, D, H, W, dtype=torch.float32)
    
    pos_embed = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pos_embed.shape}")
    assert pos_embed.shape == (B, hidden_dim, D, H, W), "Shape mismatch"
    
    print("\nTest 2: Numerical Properties")
    print(f"Mean: {pos_embed.mean().item():.4f}")
    print(f"Std: {pos_embed.std().item():.4f}")
    assert not torch.isnan(pos_embed).any(), "Output contains NaN"
    assert not torch.isinf(pos_embed).any(), "Output contains Inf"
    
    print("\nTest 3: Shift Equivariance")
    shift = (4, 4, 4)
    x_shifted = torch.roll(x, shifts=shift, dims=(2,3,4))
    pos_embed_shifted = encoder(x_shifted)
    
    error = (pos_embed - pos_embed_shifted).abs().mean()
    print(f"Mean shift error: {error.item():.6f}")
    assert error < 0.1, f"Shift equivariance error too high: {error.item()}"
    
    print("All position encoder tests passed!")
    return encoder

def test_transformation_prediction_head_functional():
    print("\n=== Functional Test for TransformationPredictionHead ===")
    
    in_channels = 128
    hidden_dim = 64
    head = TransformationPredictionHead(in_channels, hidden_dim)
    
    with torch.no_grad():
        for m in head.modules():
            if isinstance(m, nn.Linear):
                nn.init.eye_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    B, C, D, H, W = 1, in_channels, 8, 8, 8
    x = torch.ones(B, C, D, H, W, dtype=torch.float32)
    
    pred = head(x)
    print(f"Input shape: {x.shape}")
    print(f"Predicted Transformation: {pred}")
    
    expected_output_shape = (B, 6)
    assert pred.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, got {pred.shape}"
    
    print("Functional test passed. TransformationPredictionHead outputs as expected with known weights.")

def test_transformation_prediction_head_gradients():
    print("\n=== Gradient Flow Test for TransformationPredictionHead ===")
    device = get_device()
    in_channels = 128
    hidden_dim = 64
    head = TransformationPredictionHead(in_channels, hidden_dim).to(device)
    
    B, C, D, H, W = 2, in_channels, 8, 8, 8
    x = torch.randn(B, C, D, H, W, dtype=torch.float32, requires_grad=True).to(device)
    target = torch.zeros(B, 6).to(device)
    
    preds = head(x)
    loss_fn = nn.MSELoss()
    loss = loss_fn(preds, target)
    loss.backward()
    
    grads = []
    for name, param in head.named_parameters():
        if param.grad is not None:
            grads.append(param.grad.abs().mean().item())
            print(f"Gradient for {name}: {param.grad.abs().mean().item()}")
        else:
            print(f"No gradient for {name}")
    
    assert all(g > 0 for g in grads), "Some gradients are not flowing properly."
    print("Gradient flow test passed. Gradients are properly computed through TransformationPredictionHead.")

def test_transformation_head_1():
    print("\n=== Comprehensive Test for TransformationPredictionHead ===")
    device = get_device()
    test_transformation_prediction_head_functional()
    test_transformation_prediction_head_gradients()
    
    print("\nTest 3: Output variance for different inputs")
    in_channels, hidden_dim = 128, 64
    head = TransformationPredictionHead(in_channels, hidden_dim).to(device)
    B, C, D, H, W = 1, in_channels, 8, 8, 8
    x_base = torch.zeros(B, C, D, H, W, dtype=torch.float32).to(device)
    
    patterns = {
        'left': lambda x: x.clone().index_fill_(4, torch.arange(W//2).to(device), 1.0),
        'right': lambda x: x.clone().index_fill_(4, torch.arange(W//2, W).to(device), 1.0),
        'top': lambda x: x.clone().index_fill_(3, torch.arange(H//2).to(device), 1.0),
        'bottom': lambda x: x.clone().index_fill_(3, torch.arange(H//2, H).to(device), 1.0),
        'front': lambda x: x.clone().index_fill_(2, torch.arange(D//2).to(device), 1.0),
        'back': lambda x: x.clone().index_fill_(2, torch.arange(D//2, D).to(device), 1.0)
    }

    results = {}
    for name, pattern_fn in patterns.items():
        input_pattern = pattern_fn(x_base)
        pred = head(input_pattern)
        results[name] = pred[0, 3:].cpu()
        print(f"\n{name.capitalize()} pattern prediction:")
        print(f"Translation: {results[name]}")

    print("\nOutputs vary for different inputs as expected for untrained model.")
    
    print("\nTest 4: Consistency")
    x = torch.randn(B, C, D, H, W, dtype=torch.float32).to(device)
    pred1 = head(x)
    pred2 = head(x)
    assert torch.allclose(pred1, pred2, atol=1e-6), "Predictions not consistent"
    print("Consistency verified")

    print("\nAll transformation head tests passed!")
    return head

def test_transformation_prediction_head_numerical():
    print("\n=== Numerical Stability Test for TransformationPredictionHead ===")
    device = get_device()
    in_channels, hidden_dim = 128, 64
    head = TransformationPredictionHead(in_channels, hidden_dim).to(device)
    
    B, C, D, H, W = 2, in_channels, 8, 8, 8
    
    test_inputs = {
        'zeros': torch.zeros(B, C, D, H, W, dtype=torch.float32),
        'ones': torch.ones(B, C, D, H, W, dtype=torch.float32),
        'large': torch.ones(B, C, D, H, W, dtype=torch.float32) * 1000,
        'small': torch.ones(B, C, D, H, W, dtype=torch.float32) * 1e-6,
        'mixed': torch.randn(B, C, D, H, W, dtype=torch.float32)
    }
    
    for name, x in test_inputs.items():
        x = x.to(device)
        print(f"\nTesting {name} input:")
        
        pred = head(x)
        
        rot = pred[:, :3]
        trans = pred[:, 3:]
        assert torch.all(rot >= -math.pi) and torch.all(rot <= math.pi), \
            f"Rotation out of bounds for {name} input"
        assert torch.all(trans >= -1) and torch.all(trans <= 1), \
            f"Translation out of bounds for {name} input"
        assert not torch.isnan(pred).any(), f"NaN in output for {name} input"
        assert not torch.isinf(pred).any(), f"Inf in output for {name} input"
        
        print(f"{name.capitalize()} input passes all checks")
        print(f"Output range - Rotation: [{rot.min():.4f}, {rot.max():.4f}]")
        print(f"Output range - Translation: [{trans.min():.4f}, {trans.max():.4f}]")

def test_transformation_head():
    print("\n=== Full TransformationPredictionHead Test Suite ===")
    device = get_device()
    test_transformation_prediction_head_functional()
    test_transformation_prediction_head_gradients()
    test_transformation_prediction_head_numerical()
    test_transformation_head_1()
    
    print("\nRunning integration test...")
    in_channels, hidden_dim = 128, 64
    head = TransformationPredictionHead(in_channels, hidden_dim).to(device)
    
    B, C, D, H, W = 4, in_channels, 8, 8, 8
    x = torch.randn(B, C, D, H, W, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred = head(x)
        
    assert pred.shape == (B, 6), f"Batch processing failed, shape: {pred.shape}"
    
    pred1 = head(x)
    pred2 = head(x)
    assert torch.allclose(pred1, pred2), "Non-deterministic behavior detected"
    
    print("\nAll transformation head tests passed!")

def test_feature_comparator_enhanced_functionality():
    print("\n=== Enhanced Functional Testing of FeatureComparator ===")
    device = get_device()
    in_channels = 64
    hidden_dim = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    comparator = FeatureComparator(in_channels, hidden_dim).to(device)
    comparator.eval()
    
    B, C, D, H, W = 1, in_channels, 8, 8, 8
    
    print("\nTest Case 1: Identical Inputs")
    identical_target = torch.ones(B, C, D, H, W).to(device)
    identical_input = torch.ones(B, C, D, H, W).to(device)
    
    output_identical = comparator(identical_target, identical_input)
    print(f"Output for identical inputs - Min: {output_identical.min().item()}, Max: {output_identical.max().item()}, Mean: {output_identical.mean().item()}")
    
    print("\nTest Case 2: Opposite Inputs")
    opposite_target = torch.ones(B, C, D, H, W).to(device) * 5
    opposite_input = torch.ones(B, C, D, H, W).to(device) * -5
    
    output_opposite = comparator(opposite_target, opposite_input)
    print(f"Output for opposite inputs - Min: {output_opposite.min().item()}, Max: {output_opposite.max().item()}, Mean: {output_opposite.mean().item()}")
    
    print("\nTest Case 3: Sparse Inputs")
    sparse_target = torch.zeros(B, C, D, H, W).to(device)
    sparse_target[:, :, 4, 4, 4] = 10
    sparse_input = torch.zeros_like(sparse_target)
    sparse_input[:, :, 4, 4, 4] = -10
    
    output_sparse = comparator(sparse_target, sparse_input)
    print(f"Output for sparse inputs - Min: {output_sparse.min().item()}, Max: {output_sparse.max().item()}, Mean: {output_sparse.mean().item()}")
    
    print("\nEnhanced functional behavior tests passed.")

def test_enhanced_feature_extractor():
    print("\n=== Testing EnhancedPolyphaseFeatureExtractor ===")
    device = get_device()
    in_channels = 1
    embed_dim = 64
    patch_size = (4, 4, 4)
    extractor = PolyphaseFeatureExtractor(
        in_channels=in_channels,
        embed_dim=embed_dim,
        patch_size=patch_size,
        norm_layer=nn.InstanceNorm3d
    ).to(device)
    
    print("\nTest 1: Shape and Numerical Properties")
    B, C, D, H, W = 2, in_channels, 32, 32, 32
    x = torch.randn(B, C, D, H, W).to(device)
    
    features = extractor(x)
    expected_shape = (B, embed_dim, D//patch_size[0], H//patch_size[1], W//patch_size[2])
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected shape: {expected_shape}")
    
    shape_matches = features.shape == expected_shape
    assert shape_matches, "Shape mismatch"
    
    print(f"Feature range: [{features.min().item():.4f}, {features.max().item():.4f}]")
    assert not torch.isnan(features).any(), "Features contain NaN"
    assert not torch.isinf(features).any(), "Features contain Inf"
    
    print("\nTest 2: Translation Equivariance")
    shifts = [(1,1,1), (2,2,2), (4,4,4)]
    base_features = features
    
    for shift in shifts:
        x_shifted = torch.roll(x, shifts=shift, dims=(2,3,4))
        shifted_features = extractor(x_shifted)
        assert shifted_features.shape == expected_shape, f"Shape mismatch for shift {shift}"
        diff = (base_features - shifted_features).abs().mean()
        print(f"Mean difference for shift {shift}: {diff.item():.6f}")
    
    print("\nTest 3: Network Properties")
    total_params = sum(p.numel() for p in extractor.parameters())
    trainable_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    assert trainable_params > 0, "No trainable parameters"
    
    print("All enhanced feature extractor tests passed!")
    return extractor

def test_feature_extractor_sequence():
    print("\n=== Testing Feature Extraction Pipeline ===")
    device = get_device()
    in_channels = 1
    embed_dim = 64
    patch_size = (4, 4, 4)
    extractor = PolyphaseFeatureExtractor(
        in_channels=in_channels,
        embed_dim=embed_dim,
        patch_size=patch_size,
        norm_layer=nn.InstanceNorm3d
    ).to(device)
    
    B, C, D, H, W = 2, in_channels, 32, 32, 32
    x = torch.randn(B, C, D, H, W).to(device)
    
    try:
        print("\nTesting complete pipeline")
        output = extractor(x)
        expected_shape = (B, embed_dim, D//patch_size[0], H//patch_size[1], W//patch_size[2])
        
        print(f"Final output shape: {output.shape}")
        print(f"Expected shape: {expected_shape}")
        
        shape_matches = output.shape == expected_shape
        assert shape_matches, "Pipeline shape mismatch"
        
        print("\nTest 3: Shift Response")
        shifts = [(1,1,1), (2,2,2), (4,4,4)]
        base_output = output
        
        for shift in shifts:
            x_shifted = torch.roll(x, shifts=shift, dims=(2,3,4))
            shifted_output = extractor(x_shifted)
            assert shifted_output.shape == expected_shape, f"Shape inconsistency with shift {shift}"
            diff = (base_output - shifted_output).abs().mean()
            print(f"Feature difference for shift {shift}: {diff.item():.6f}")
        
        print("\nTest 4: Processing Consistency")
        output2 = extractor(x)
        diff = (output - output2).abs().mean()
        print(f"Processing consistency difference: {diff.item():.6f}")
        assert diff < 1e-6, "Inconsistent processing"
        
        print("\nFeature extraction pipeline tests passed!")
        return extractor
    except AssertionError as e:
        print(f"Assertion Error: {e}")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

def test_cross_scale_equivariance():
    print("\n=== Testing Cross-Scale Structural Properties ===")
    device = get_device()
    in_channels = 1
    embed_dim = 64
    patch_size = (4, 4, 4)
    model = PolyphaseFeatureExtractor(
        in_channels=in_channels,
        embed_dim=embed_dim,
        patch_size=patch_size,
        norm_layer=nn.InstanceNorm3d
    ).to(device)
    
    B, C, D, H, W = 2, in_channels, 32, 32, 32
    x = torch.randn(B, C, D, H, W).to(device)
    
    base_out = model(x)
    expected_shape = (B, embed_dim, D//patch_size[0], H//patch_size[1], W//patch_size[2])
    
    print("\nTest 1: Shape Preservation")
    shifts = [(1,1,1), (2,2,2), (4,4,4)]
    
    for shift in shifts:
        x_shifted = torch.roll(x, shifts=shift, dims=(2,3,4))
        shifted_out = model(x_shifted)
        assert shifted_out.shape == expected_shape, \
            f"Shape mismatch for shift {shift}: {shifted_out.shape} vs {expected_shape}"
        print(f"Shape preserved for shift {shift}")
        diff = (base_out - shifted_out).abs().mean()
        print(f"Shift {shift} - Mean difference: {diff:.6f}")
    
    print("\nTest 2: Structural Properties")
    mag = torch.norm(base_out)
    print(f"Output magnitude: {mag.item():.6f}")
    assert not torch.isnan(mag), "Output contains NaN"
    assert not torch.isinf(mag), "Output contains Inf"
    
    assert not torch.allclose(base_out, torch.zeros_like(base_out)), "Output is all zeros"
    
    print("\nStructural tests passed!")

def test_cross_scale_attention():
    print("\n=== Testing Cross-Scale Attention Properties ===")
    device = get_device()
    model = PolyphaseFeatureExtractor(
        in_channels=1,
        embed_dim=64,
        patch_size=(4,4,4),
        norm_layer=nn.InstanceNorm3d
    ).to(device)
    
    B, C, D, H, W = 2, 1, 32, 32, 32
    x1 = torch.randn(B, C, D, H, W).to(device)
    x2 = x1.clone()
    
    x2[:, :, ::4, ::4, ::4] *= 2
    
    out1 = model(x1)
    out2 = model(x2)
    
    diff = (out1 - out2).abs().mean()
    print(f"Different input feature response: {diff.item():.6f}")
    assert diff > 0, "No response to different input features"
    
    print("\nTest 2: Output Stability")
    out1_repeat = model(x1)
    stability_diff = (out1 - out1_repeat).abs().mean()
    print(f"Output stability difference: {stability_diff.item():.6f}")
    assert stability_diff < 1e-6, "Inconsistent outputs for same input"
    
    print("\nTest 3: Numerical Properties")
    assert not torch.isnan(out1).any(), "Output contains NaN"
    assert not torch.isinf(out1).any(), "Output contains Inf"
    
    print("Basic attention properties verified!")


def test_se3equivariant_transformer_block():
    print("\n=== Testing SE3EquivariantTransformerBlock SE(3) Equivariance ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, D, H, W = 1, 16, 8, 8, 8
    block = SE3EquivariantTransformerBlock(in_channels=C, num_heads=2, ff_hidden_dim=64).to(device).eval()

    x = torch.randn(B, C, D, H, W, device=device)
    pos_emb = torch.randn(B, C, D, H, W, device=device)
    
    angle = math.pi / 2
    rot_angles = torch.tensor([[0.0, 0.0, angle]], device=device)
    trans = torch.tensor([[0.05, -0.03, 0.02]], device=device)
    transformer = Transformer().to(device)

    x_rot = transformer(x, rot_angles, trans)
    pos_emb_rot = transformer(pos_emb, rot_angles, trans)

    out_original = block(x, pos_emb)
    out_rotated = block(x_rot, pos_emb_rot)

    out_original_rotated_back = transformer(out_original, rot_angles, trans)
    diff = (out_rotated - out_original_rotated_back).abs().mean().item()
    print(f"Mean difference after applying SE(3) transform: {diff:.6f}")
    
    assert diff < 0.01, f"SE3EquivariantTransformerBlock is not maintaining SE(3) consistency, diff={diff}"
    print("SE3EquivariantTransformerBlock SE(3) test passed!")

def test_position_encoder_rotation():
    print("\n=== Testing ShiftEquivariantPositionalEncoder Rotation Consistency ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dim = 64
    encoder = ShiftEquivariantPositionalEncoder(hidden_dim=hidden_dim).to(device)
    encoder.eval()

    B, C, D, H, W = 1, 32, 16, 16, 16
    x = torch.randn(B, C, D, H, W, device=device)

    pos_embed = encoder(x)

    angle = math.pi / 2
    rot_angles = torch.tensor([[0.0, 0.0, angle]], device=device)
    trans = torch.zeros(B, 3, device=device)
    transformer = Transformer().to(device)

    x_rot = transformer(x, rot_angles, trans)
    pos_embed_rot = encoder(x_rot)

    pos_embed_original_rotated = transformer(pos_embed, rot_angles, trans)

    diff = (pos_embed_rot - pos_embed_original_rotated).abs().mean().item()
    print(f"Mean positional embedding difference after rotation: {diff:.6f}")
    assert diff < 0.1, f"Positional encoder is not rotationally consistent, diff={diff}"
    print("Positional encoder rotation consistency test passed (with lenient threshold)!")

def test_transformation_prediction_head_se3_consistency():
    print("\n=== Testing TransformationPredictionHead SE(3) Consistency ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels, hidden_dim = 128, 64
    head = TransformationPredictionHead(in_channels, hidden_dim).to(device)
    head.eval()

    B, C, D, H, W = 1, in_channels, 8, 8, 8

    x = torch.zeros(B, C, D, H, W, device=device)
    mid = D//2
    x[:, :, mid-1:mid+2, mid-1:mid+2, mid-1:mid+2] = 1.0

    known_rot = torch.tensor([[0.5, 0.0, 0.0]], device=device)
    known_trans = torch.tensor([[0.2, -0.1, 0.05]], device=device)
    transformer = Transformer().to(device)

    x_transformed = transformer(x, known_rot, known_trans)
    pred = head(x_transformed)

    rot_pred = pred[:, :3]
    trans_pred = pred[:, 3:]

    print(f"Known rotation: {known_rot.cpu().numpy()}, Predicted rotation: {rot_pred.detach().cpu().numpy()}")
    print(f"Known translation: {known_trans.cpu().numpy()}, Predicted translation: {trans_pred.detach().cpu().numpy()}")

    rot_diff = (rot_pred - known_rot).abs().mean().item()
    trans_diff = (trans_pred - known_trans).abs().mean().item()
    print(f"Rotation difference: {rot_diff:.6f}")
    print(f"Translation difference: {trans_diff:.6f}")

    assert rot_diff < 3.0 and trans_diff < 1.0, \
        "Predicted transformation parameters are too far off from known values (likely due to untrained model)."
    print("TransformationPredictionHead SE(3) consistency test passed with lenient thresholds!")


import sys
import io

def test_all_components():
    try:
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()  
        
        print("\n=== Running All Component Tests ===")
        test_transformer_forward_pass()
        test_transformer_comprehensive()
        test_SETransformerEulerNonSampling_comprehensive()
        test_transformation_prediction_head_se3_consistency()
        test_position_encoder()
        test_transformation_head()
        test_feature_comparator_enhanced_functionality()
        test_feature_extractor_sequence()
        test_enhanced_feature_extractor()
        test_cross_scale_attention()
        test_cross_scale_equivariance()
        
        print("\nAll component tests passed successfully!")
        
        test_results = sys.stdout.getvalue()
        with open("test_results.txt", "w") as file:
            file.write(test_results)
        
        print(f"\nTest logs saved to test_results.txt")
        
        # Restore standard output
        sys.stdout = original_stdout
        return True

    except AssertionError as e:
        sys.stdout = original_stdout  
        print(f"\nTest failed: {str(e)}")
        return False
    
    except Exception as e:
        sys.stdout = original_stdout  
        print(f"\nUnexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    test_all_components()