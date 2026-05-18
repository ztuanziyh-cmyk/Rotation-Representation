import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def skew_symmetric(w):
    zero = torch.zeros_like(w[..., :1])
    w_x, w_y, w_z = w[..., 0:1], w[..., 1:2], w[..., 2:3]
    O = torch.cat([
        zero, -w_z, w_y,
        w_z, zero, -w_x,
        -w_y, w_x, zero
    ], dim=-1)
    O = O.view(*w.shape[:-1], 3, 3)
    return O

def exp_skew_symmetric(w):
    theta = torch.norm(w, dim=-1, keepdim=True)
    theta = theta.clamp(min=1e-8)
    w_unit = w / theta
    w_hat = skew_symmetric(w_unit)
    theta = theta.unsqueeze(-1)
    I = torch.eye(3, device=w.device).expand_as(w_hat)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    R = I + sin_theta * w_hat + (1 - cos_theta) * torch.matmul(w_hat, w_hat)
    return R

class PositionalProcessing(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        
        self.local_processor = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, 
                     padding=1, padding_mode='circular'),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, 1)
        )
        
        self.modulation = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 1),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.pos_attention = nn.Sequential(
            nn.Conv3d(hidden_dim, num_heads, 1),
            nn.InstanceNorm3d(num_heads),
            nn.GELU()
        )
        
    def forward(self, x, pos_emb):
        local_feat = self.local_processor(pos_emb)
        mod_factors = self.modulation(local_feat)
        x_mod = x * mod_factors
        
        pos_attn = self.pos_attention(pos_emb)
        
        return x_mod, pos_attn

def LIERE(X, p, A, feature_type='vector'):
    rotation_params = torch.matmul(p, A.transpose(-2, -1))
    
    R = exp_skew_symmetric(rotation_params)
    
    if feature_type == 'vector':
        assert X.size(-1) % 3 == 0, "Feature dimension must be divisible by 3"
        num_vectors = X.size(-1) // 3
        X_vectors = X.view(*X.shape[:-1], num_vectors, 3)
        X_rotated = torch.matmul(X_vectors, R.transpose(-2, -1))
        return X_rotated.view(*X.shape)
    return X

def apply_translation(p, t):
    return p + t

class AxialSelfAttentionModule(nn.Module):
    def __init__(self, in_channels, num_heads, feature_type='vector', max_relative_position=64):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.feature_type = feature_type
        self.max_relative_position = max_relative_position

        self.qkv = nn.Linear(in_channels, 3 * in_channels)
        
        self.pos_processor = PositionalProcessing(in_channels, num_heads)
        
        self.A_d = nn.Parameter(torch.randn(1, num_heads, 3, 3))
        self.A_h = nn.Parameter(torch.randn(1, num_heads, 3, 3))
        self.A_w = nn.Parameter(torch.randn(1, num_heads, 3, 3))
        
        self.t_d = nn.Parameter(torch.zeros(1, num_heads, 1, 3))  
        self.t_h = nn.Parameter(torch.zeros(1, num_heads, 1, 3))
        self.t_w = nn.Parameter(torch.zeros(1, num_heads, 1, 3))
        
        for A in [self.A_d, self.A_h, self.A_w]:
            nn.init.orthogonal_(A)
        
        for t_param in [self.t_d, self.t_h, self.t_w]:
            nn.init.normal_(t_param, std=0.01)

        self.proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.pos_bias_layer = nn.Linear(3, 1)

    def _axial_attention(self, x, pos_emb, axis, A, t):
        B, C, D, H, W = x.shape
        
        x_mod, pos_attn = self.pos_processor(x, pos_emb)
        
        if axis == 'depth':
            x_perm = x_mod.permute(0, 3, 4, 1, 2).contiguous()
            if pos_attn is not None:
                pos_attn = pos_attn.permute(0, 3, 4, 1, 2).contiguous()
            batch_dim = B * H * W
            L = D
        elif axis == 'height':
            x_perm = x_mod.permute(0, 2, 4, 1, 3).contiguous()
            if pos_attn is not None:
                pos_attn = pos_attn.permute(0, 2, 4, 1, 3).contiguous()
            batch_dim = B * D * W
            L = H
        else:  # width
            x_perm = x_mod.permute(0, 2, 3, 1, 4).contiguous()
            if pos_attn is not None:
                pos_attn = pos_attn.permute(0, 2, 3, 1, 4).contiguous()
            batch_dim = B * D * H
            L = W
            
        x_perm = x_perm.reshape(batch_dim, C, L)
        
        pos = self._get_axis_positions(L, axis, x.device)
        qkv = self.qkv(x_perm.transpose(-1, -2))
        qkv = qkv.reshape(batch_dim, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        
        pos = pos.unsqueeze(0).unsqueeze(1).expand(batch_dim, self.num_heads, L, 3)
        A = A.expand(batch_dim, -1, -1, -1)
        t = t.expand(batch_dim, -1, L, -1)
        
        q_rot = LIERE(q, pos, A, self.feature_type)
        k_rot = LIERE(k, pos, A, self.feature_type)
        
        attn = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * self.scale
        
        if pos_attn is not None:
            pos_attn = pos_attn.reshape(batch_dim, self.num_heads, L)  
            pos_attn = pos_attn.unsqueeze(-1).expand(-1, -1, -1, L)   
            attn = attn + pos_attn
        
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_dim, L, -1)
        
        if axis == 'depth':
            out = out.reshape(B, H, W, D, -1).permute(0, 4, 3, 1, 2)
        elif axis == 'height':
            out = out.reshape(B, D, W, H, -1).permute(0, 4, 1, 3, 2)
        else:  # width
            out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3)
            
        return out

    def _compute_positional_bias(self, pos_translated):
        rel_pos = pos_translated.unsqueeze(-2) - pos_translated.unsqueeze(-3)  
        rel_pos_flat = rel_pos.view(-1, 3) 
        bias = self.pos_bias_layer(rel_pos_flat)  
        batch_dim = pos_translated.size(0)
        num_heads = pos_translated.size(1)
        L = pos_translated.size(2)
        bias = bias.view(batch_dim, num_heads, L, L)  
        return bias

    def forward(self, x, pos_emb):
        out_d = self._axial_attention(x, pos_emb, 'depth', self.A_d, self.t_d)
        out_h = self._axial_attention(x, pos_emb, 'height', self.A_h, self.t_h)
        out_w = self._axial_attention(x, pos_emb, 'width', self.A_w, self.t_w)
        
        out_sum = out_d + out_h + out_w
        out = self.proj(out_sum)
        return out

    def _get_axis_positions(self, length, axis, device):
        pos = torch.linspace(-1, 1, steps=length, device=device)
        if axis == 'depth':
            return torch.stack([torch.zeros_like(pos), torch.zeros_like(pos), pos], dim=-1)
        elif axis == 'height':
            return torch.stack([torch.zeros_like(pos), pos, torch.zeros_like(pos)], dim=-1)
        else:  # width
            return torch.stack([pos, torch.zeros_like(pos), torch.zeros_like(pos)], dim=-1)


    def _get_relative_pos_bias(self, length, bias):
        pos = torch.arange(length, device=bias.device)
        rel_pos = pos.view(-1, 1) - pos.view(1, -1)  
        max_rel = self.max_relative_position - 1
        rel_pos = rel_pos.clamp(-max_rel, max_rel)
        rel_pos += max_rel
        return bias[rel_pos].unsqueeze(0)


# ===========================
# Self-Attention Module Wrapper
# ===========================
class SelfAttentionModule(nn.Module):
    def __init__(self, in_channels, num_heads, feature_type='vector'):
        super().__init__()
        self.attention = AxialSelfAttentionModule(
            in_channels=in_channels,
            num_heads=num_heads,
            feature_type=feature_type
        )

    def forward(self, x, pos_emb, mask=None):  
        return self.attention(x, pos_emb)
