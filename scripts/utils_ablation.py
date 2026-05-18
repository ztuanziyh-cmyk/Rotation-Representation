import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

# The following utilities are from the GUMNet PyTorch implementation -- some modifications done
def axis_angle_to_matrix(ax, ay, az, device):
    sx = torch.sin(torch.tensor(ax, device=device))
    cx = torch.cos(torch.tensor(ax, device=device))
    sy = torch.sin(torch.tensor(ay, device=device))
    cy = torch.cos(torch.tensor(ay, device=device))
    sz = torch.sin(torch.tensor(az, device=device))
    cz = torch.cos(torch.tensor(az, device=device))
    
    Rx = torch.tensor([
        [1,   0,    0],
        [0,   cx,  -sx],
        [0,   sx,   cx]
    ], device=device, dtype=torch.float32)
    
    Ry = torch.tensor([
        [ cy,   0,  sy],
        [  0,   1,   0],
        [-sy,   0,  cy]
    ], device=device, dtype=torch.float32)
    
    Rz = torch.tensor([
        [ cz, -sz,  0],
        [ sz,  cz,  0],
        [  0,   0,  1]
    ], device=device, dtype=torch.float32)
    
    R = torch.matmul(Rz, torch.matmul(Ry, Rx))
    return R

def rotate_tensor(tensor, angle_rad, axes, device):
    B = tensor.size(0)
    dtype = tensor.dtype  
    R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)  

    if axes == (2, 3):
        cos_a = torch.cos(angle_rad).type(dtype)
        sin_a = torch.sin(angle_rad).type(dtype)
        R[:,1,1] = cos_a
        R[:,1,2] = -sin_a
        R[:,2,1] = sin_a
        R[:,2,2] = cos_a
    elif axes == (2, 4):
        cos_a = torch.cos(angle_rad).type(dtype)
        sin_a = torch.sin(angle_rad).type(dtype)
        R[:,0,0] = cos_a
        R[:,0,2] = sin_a
        R[:,2,0] = -sin_a
        R[:,2,2] = cos_a
    elif axes == (3, 4):
        cos_a = torch.cos(angle_rad).type(dtype)
        sin_a = torch.sin(angle_rad).type(dtype)
        R[:,0,0] = cos_a
        R[:,0,1] = -sin_a
        R[:,1,0] = sin_a
        R[:,1,1] = cos_a
    else:
        raise ValueError("Invalid axes for rotation. Must be one of (2, 3), (2, 4), or (3, 4).")

    affine_matrix = torch.zeros(B, 3, 4, device=device, dtype=dtype)
    affine_matrix[:, :3, :3] = R
    affine_matrix[:, :3, 3] = 0 

    grid = F.affine_grid(affine_matrix, tensor.size(), align_corners=False)  

    if torch.isnan(grid).any() or torch.isinf(grid).any():
        print("NaNs or Infs detected in grid in rotate_tensor.")
        raise ValueError("Invalid grid in rotate_tensor.")

    rotated = F.grid_sample(tensor, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return rotated

def rotate_around_center(x, rotation_matrix, offset, device):
    B = x.size(0)
    dtype = x.dtype
    
    affine_matrix = torch.zeros(B, 3, 4, device=device, dtype=dtype)
    affine_matrix[:, :3, :3] = rotation_matrix.unsqueeze(0).repeat(B, 1, 1)
    affine_matrix[:, :3, 3] = offset
    
    grid = F.affine_grid(affine_matrix, x.size(), align_corners=True)
    
    if torch.isnan(grid).any() or torch.isinf(grid).any():
        print("NaNs or Infs detected in grid in rotate_around_center.")
        raise ValueError("Invalid grid in rotate_around_center.")
    
    rotated = F.grid_sample(
        x, 
        grid, 
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    return rotated


# def shift_tensor(tensor, shift, device):
#     B, C, D, H, W = tensor.size()
    
#     shifts_normalized = shift.clone()
#     shifts_normalized[:, 0] = shift[:, 0] * 2.0 / (W - 1)
#     shifts_normalized[:, 1] = shift[:, 1] * 2.0 / (H - 1)
#     shifts_normalized[:, 2] = shift[:, 2] * 2.0 / (D - 1)
    
#     affine_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
#     affine_matrix[:, :3, 3] = -shifts_normalized
#     affine_matrices = affine_matrix[:, :3, :]
    
#     grid = F.affine_grid(
#         affine_matrices, 
#         tensor.size(), 
#         align_corners=True
#     )

#     shifted = F.grid_sample(
#         tensor, 
#         grid,
#         mode='nearest',
#         padding_mode='border',
#         align_corners=True
#     )
#     return shifted

def shift_tensor(tensor, shift, device):
    B, C, D, H, W = tensor.size()
    sx = shift[:, 2] * 2.0 / (W - 1)
    sy = shift[:, 1] * 2.0 / (H - 1)
    sz = shift[:, 0] * 2.0 / (D - 1)
    s_new = torch.stack([sx, sy, sz], dim=1)

    A = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    A[:, :3, 3] = -s_new  
    M = A[:, :3, :]
    g = F.affine_grid(M, tensor.size(), align_corners=True)
    return F.grid_sample(tensor, g, mode='nearest', padding_mode='border', align_corners=True)


def compose_rotation(ax, ay, az, device):
    Rx = axis_angle_to_matrix(ax, 0.0, 0.0, device)
    Ry = axis_angle_to_matrix(0.0, ay, 0.0, device)
    Rz = axis_angle_to_matrix(0.0, 0.0, az, device)
    R_approx = Rz @ (Ry @ Rx)
    U, _, V = torch.linalg.svd(R_approx)
    det_val = torch.linalg.det(torch.matmul(U, V.transpose(-2, -1)))
    if det_val < 0:
        V[..., -1] *= -1.0
    R = torch.matmul(U, V.transpose(-2, -1))
    return R

def shear_tensor(x, shear_factors, device):
    B, C, D, H, W = x.shape
    sheared = []
    for i in range(B):
        shear_x, shear_y, shear_z = shear_factors[i]
        affine_matrix = torch.tensor([
            [1,       shear_x, shear_y, 0],
            [0,       1,       shear_z, 0],
            [0,       0,       1,       0]
        ], dtype=torch.float32, device=device)
        
        grid = F.affine_grid(
            affine_matrix.unsqueeze(0), 
            x[i:i+1].size(), 
            align_corners=True
        )
        
        sheared_sample = F.grid_sample(
            x[i:i+1], 
            grid, 
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        sheared.append(sheared_sample)
    sheared = torch.cat(sheared, dim=0)
    return sheared

def random_translations(B, D, H, W, translation_range, device):
    max_trans_d = D * translation_range
    max_trans_h = H * translation_range
    max_trans_w = W * translation_range
    translations_d = torch.FloatTensor(B).uniform_(-max_trans_d, max_trans_d).to(device)
    translations_h = torch.FloatTensor(B).uniform_(-max_trans_h, max_trans_h).to(device)
    translations_w = torch.FloatTensor(B).uniform_(-max_trans_w, max_trans_w).to(device)
    translations = torch.stack((translations_d, translations_h, translations_w), dim=1)
    return translations

def augment_tensors(x, y, device):
    angle_x = torch.FloatTensor(1).uniform_(-10, 10).to(device).squeeze()
    angle_y = torch.FloatTensor(1).uniform_(-10, 10).to(device).squeeze()
    angle_z = torch.FloatTensor(1).uniform_(-10, 10).to(device).squeeze()
    x = rotate_tensor(x, angle_x.unsqueeze(0), axes=(2, 3), device=device)  
    x = rotate_tensor(x, angle_y.unsqueeze(0), axes=(2, 4), device=device)  
    x = rotate_tensor(x, angle_z.unsqueeze(0), axes=(3, 4), device=device)  
    y = rotate_tensor(y, angle_x.unsqueeze(0), axes=(2, 3), device=device)
    y = rotate_tensor(y, angle_y.unsqueeze(0), axes=(2, 4), device=device)
    y = rotate_tensor(y, angle_z.unsqueeze(0), axes=(3, 4), device=device)
    B = x.size(0)
    translate_x = torch.FloatTensor(B).uniform_(-5, 5).to(device)
    translate_y = torch.FloatTensor(B).uniform_(-5, 5).to(device)
    translate_z = torch.FloatTensor(B).uniform_(-5, 5).to(device)
    shifts = torch.stack((translate_x, translate_y, translate_z), dim=1)

    x = shift_tensor(x, shifts, device)
    y = shift_tensor(y, shifts, device)

    noise_x = Normal(0, 0.01).sample(x.size()).to(device)
    noise_y = Normal(0, 0.01).sample(y.size()).to(device)
    x = x + noise_x
    y = y + noise_y
    return x, y


def transformation_loss(predictions, ground_truths, transform_type='r9'):
    # print(f"transformation_loss Called with transform_type={transform_type}")
    alpha = 1.0
    beta = 3.0

    if transform_type == 'r9':
        pred_rot = predictions[:, :9]
        pred_trans = predictions[:, 9:]
        gt_rot = ground_truths[:, :9]
        gt_trans = ground_truths[:, 9:]
        rot_loss = F.mse_loss(pred_rot, gt_rot)
        trans_loss = F.mse_loss(pred_trans, gt_trans)
        return alpha * rot_loss + beta * trans_loss

    elif transform_type == 'r6':
        pred_rot = predictions[:, :6]
        pred_trans = predictions[:, 6:]
        gt_rot = ground_truths[:, :6]
        gt_trans = ground_truths[:, 6:]

        angle_loss = F.mse_loss(pred_rot, gt_rot)
        trans_loss = F.mse_loss(pred_trans, gt_trans)
        return alpha * angle_loss + beta * trans_loss
    
    # elif transform_type == 'quat':
    #     import kornia
    #     pred_quat_unnorm = predictions[:, :4]
    #     pred_trans = predictions[:, 4:]

    #     gt_quat_unnorm = ground_truths[:, :4]
    #     gt_trans = ground_truths[:, 4:]

    #     pred_quat = pred_quat_unnorm / (torch.norm(pred_quat_unnorm, dim=1, keepdim=True) + 1e-8)
    #     gt_quat   = gt_quat_unnorm   / (torch.norm(gt_quat_unnorm,   dim=1, keepdim=True) + 1e-8)

    #     R_pred = kornia.geometry.quaternion_to_rotation_matrix(pred_quat)[:, :3, :3]
    #     R_gt   = kornia.geometry.quaternion_to_rotation_matrix(gt_quat)[:, :3, :3]
    #     rot_loss = F.mse_loss(R_pred.reshape(-1, 9), R_gt.reshape(-1, 9))
    #     trans_loss = F.mse_loss(pred_trans, gt_trans)
    #     return alpha * rot_loss + beta * trans_loss
    else: 
        pred_angles = predictions[:, :3]
        pred_trans = predictions[:, 3:]
        gt_angles = ground_truths[:, :3]
        gt_trans = ground_truths[:, 3:]
        angle_loss = F.mse_loss(pred_angles, gt_angles)
        trans_loss = F.mse_loss(pred_trans, gt_trans)
        return alpha * angle_loss + beta * trans_loss


def angle_difference(angle1, angle2):
    diff = (angle1 - angle2) % (2 * torch.pi)
    return torch.min(diff, 2 * torch.pi - diff)

def matrix_to_euler(R):
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.zeros_like(y)
    
    return torch.stack([x, y, z])

def compute_angle_diff(eul_true, eul_pred):
    return torch.sqrt(sum(angle_difference(t, p)**2 for t, p in zip(eul_true, eul_pred)))

def alignment_eval(y_true, y_pred, image_size, scale=True, transform_type='r9'):
    if isinstance(image_size, (list, tuple)):
        image_size = float(image_size[-1])
    else:
        image_size = float(image_size)
        
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    if transform_type == 'r9':
        angle_d = []
        trans_d = []
        for i in range(len(y_true)):
            R_true = y_true[i][:9].reshape(3, 3)
            R_pred = y_pred[i][:9].reshape(3, 3)
            
            eul_true = matrix_to_euler(R_true)
            eul_pred = matrix_to_euler(R_pred)
            
            T_true = y_true[i][9:]
            T_pred = y_pred[i][9:]
            if scale:
                T_true = T_true * image_size
                T_pred = T_pred * image_size
            
            angle_diff = compute_angle_diff(eul_true, eul_pred)
            trans_diff = torch.norm(T_true - T_pred)
            
            angle_d.append(angle_diff.item())
            trans_d.append(trans_diff.item())
            
        angle_mean, angle_std = torch.tensor(angle_d).mean().item(), torch.tensor(angle_d).std().item()
        trans_mean, trans_std = torch.tensor(trans_d).mean().item(), torch.tensor(trans_d).std().item()
        print(f"R9 Rotation error: {angle_mean:.4f} ± {angle_std:.4f}, "
              f"Trans error: {trans_mean:.4f} ± {trans_std:.4f}")
              
    elif transform_type == 'r6':
        angle_d = []
        trans_d = []
        for i in range(len(y_true)):
            r6_true = y_true[i][:6]
            r6_pred = y_pred[i][:6]
            
            R_true = r6_to_matrix(r6_true.unsqueeze(0))[0]
            R_pred = r6_to_matrix(r6_pred.unsqueeze(0))[0]
            
            eul_true = matrix_to_euler(R_true)
            eul_pred = matrix_to_euler(R_pred)
            
            T_true = y_true[i][6:]
            T_pred = y_pred[i][6:]
            if scale:
                T_true = T_true * image_size
                T_pred = T_pred * image_size
            
            angle_diff = compute_angle_diff(eul_true, eul_pred)
            trans_diff = torch.norm(T_true - T_pred)
            
            angle_d.append(angle_diff.item())
            trans_d.append(trans_diff.item())
            
        angle_mean, angle_std = torch.tensor(angle_d).mean().item(), torch.tensor(angle_d).std().item()
        trans_mean, trans_std = torch.tensor(trans_d).mean().item(), torch.tensor(trans_d).std().item()
        print(f"R6 Rotation error: {angle_mean:.4f} ± {angle_std:.4f}, "
              f"Trans error: {trans_mean:.4f} ± {trans_std:.4f}")

    # elif transform_type == 'quat':
    #     import kornia
    #     angle_d = []
    #     trans_d = []
    #     for i in range(len(y_true)):
    #         gt_quat_unnorm = y_true[i][:4]
    #         gt_trans = y_true[i][4:]
    #         pred_quat_unnorm = y_pred[i][:4]
    #         pred_trans = y_pred[i][4:]
    #         gt_quat = gt_quat_unnorm / (torch.norm(gt_quat_unnorm) + 1e-8)
    #         pr_quat = pred_quat_unnorm / (torch.norm(pred_quat_unnorm) + 1e-8)
    #         R_gt = kornia.geometry.quaternion_to_rotation_matrix(gt_quat.unsqueeze(0))[:, :3, :3][0]
    #         R_pr = kornia.geometry.quaternion_to_rotation_matrix(pr_quat.unsqueeze(0))[:, :3, :3][0]
    #         eul_true = matrix_to_euler(R_gt)
    #         eul_pred = matrix_to_euler(R_pr)
    #         if scale:
    #             gt_trans = gt_trans * image_size
    #             pred_trans = pred_trans * image_size

    #         angle_diff = compute_angle_diff(eul_true, eul_pred)
    #         trans_diff = torch.norm(gt_trans - pred_trans)
            
    #         angle_d.append(angle_diff.item())
    #         trans_d.append(trans_diff.item())

    #     angle_mean, angle_std = torch.tensor(angle_d).mean().item(), torch.tensor(angle_d).std().item()
    #     trans_mean, trans_std = torch.tensor(trans_d).mean().item(), torch.tensor(trans_d).std().item()
    #     print(f"QUAT Rotation error: {angle_mean:.4f} ± {angle_std:.4f}, "
    #           f"Trans error: {trans_mean:.4f} ± {trans_std:.4f}")
    
    else:
        angle_d = []
        trans_d = []
        for i in range(len(y_true)):
            eul_true = y_true[i][:3]
            eul_pred = y_pred[i][:3]
            
            T_true = y_true[i][3:]
            T_pred = y_pred[i][3:]
            if scale:
                T_true = T_true * image_size
                T_pred = T_pred * image_size
            
            angle_diff = compute_angle_diff(eul_true, eul_pred)
            trans_diff = torch.norm(T_true - T_pred)
            
            angle_d.append(angle_diff.item())
            trans_d.append(trans_diff.item())
            
        angle_mean, angle_std = torch.tensor(angle_d).mean().item(), torch.tensor(angle_d).std().item()
        trans_mean, trans_std = torch.tensor(trans_d).mean().item(), torch.tensor(trans_d).std().item()
        print(f"Euler Rotation error: {angle_mean:.4f} ± {angle_std:.4f}, "
              f"Trans error: {trans_mean:.4f} ± {trans_std:.4f}")

        
def angle_zyz_difference(ang1=torch.zeros(3), ang2=torch.zeros(3)):
    rm1 = rotation_matrix_zyz(ang1)
    rm2 = rotation_matrix_zyz(ang2)
    dif_m = rm1 - rm2
    dif_d = torch.sqrt(torch.square(dif_m).sum())
    return dif_d.item()

def compute_rotation_matrices(angles):
    return torch.stack([rotation_matrix_zyz(angle) for angle in angles])

def rotation_matrix_zyz(ang):
    phi = ang[0]
    theta = ang[1]
    psi_t = ang[2]
    a1 = rotation_matrix_axis(2, psi_t)
    a2 = rotation_matrix_axis(1, theta)
    a3 = rotation_matrix_axis(2, phi)
    rm = torch.matmul(a3, torch.matmul(a2, a1))
    return rm

def generate_masks(x, tilt_angle=30):
    batches, channels, depth, height, width = x.shape
    tilt_radians = np.radians(tilt_angle)
    kz = np.fft.fftfreq(depth)
    ky = np.fft.fftfreq(height)
    kx = np.fft.fftfreq(width)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    angles = np.abs(np.arcsin(KZ))
    mask = np.ones((depth, height, width), dtype=np.float32)
    mask[angles > tilt_radians] = 0
    mask = torch.tensor(mask, dtype=x.dtype, device=x.device)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batches, channels, depth, height, width)
    return mask, 1 - mask

def rotation_matrix_axis(dim, theta):
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    if dim == 0:
        rm = torch.stack([
            torch.stack([torch.ones_like(cos_theta), torch.zeros_like(cos_theta), torch.zeros_like(cos_theta)], dim=-1),
            torch.stack([torch.zeros_like(cos_theta), cos_theta, -sin_theta], dim=-1),
            torch.stack([torch.zeros_like(cos_theta), sin_theta, cos_theta], dim=-1)
        ], dim=1)
    elif dim == 1:
        rm = torch.stack([
            torch.stack([cos_theta, torch.zeros_like(cos_theta), sin_theta], dim=-1),
            torch.stack([torch.zeros_like(cos_theta), torch.ones_like(cos_theta), torch.zeros_like(cos_theta)], dim=-1),
            torch.stack([-sin_theta, torch.zeros_like(cos_theta), cos_theta], dim=-1)
        ], dim=1)
    elif dim == 2:
        rm = torch.stack([
            torch.stack([cos_theta, -sin_theta, torch.zeros_like(cos_theta)], dim=-1),
            torch.stack([sin_theta, cos_theta, torch.zeros_like(cos_theta)], dim=-1),
            torch.stack([torch.zeros_like(cos_theta), torch.zeros_like(cos_theta), torch.ones_like(cos_theta)], dim=-1)
        ], dim=1)
    else:
        raise ValueError("Invalid axis. Must be 0, 1, or 2.")
    
    return rm

def cross_correlation_loss(aligned, target):
    aligned_flat = aligned.view(aligned.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    aligned_norm = F.normalize(aligned_flat, p=2, dim=1)
    target_norm = F.normalize(target_flat, p=2, dim=1)
    cc = torch.sum(aligned_norm * target_norm, dim=1)
    cc_loss = 1 - cc  
    return cc_loss.mean()

def ssim_loss(x, y, window_size=11, size_average=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool3d(x, window_size, stride=1, padding=window_size//2)
    mu_y = F.avg_pool3d(y, window_size, stride=1, padding=window_size//2)
    sigma_x = F.avg_pool3d(x * x, window_size, stride=1, padding=window_size//2) - mu_x * mu_x
    sigma_y = F.avg_pool3d(y * y, window_size, stride=1, padding=window_size//2) - mu_y * mu_y
    sigma_xy = F.avg_pool3d(x * y, window_size, stride=1, padding=window_size//2) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean([1,2,3,4])
    
    
def gradient_difference_loss(aligned_input, targets):
    sobel_x = torch.tensor([[[[-1, -1, -1],
                              [0,  0,  0],
                              [1,  1,  1]],
                             [[-3, -3, -3],
                              [0,  0,  0],
                              [3,  3,  3]],
                             [[-1, -1, -1],
                              [0,  0,  0],
                              [1,  1,  1]]]], dtype=torch.float32, device=aligned_input.device)
    
    sobel_y = torch.tensor([[[[-1, -3, -1],
                              [-1, -3, -1],
                              [-1, -3, -1]],
                             [[0,  0,  0],
                              [0,  0,  0],
                              [0,  0,  0]],
                             [[1,  3,  1],
                              [1,  3,  1],
                              [1,  3,  1]]]], dtype=torch.float32, device=aligned_input.device)
    
    sobel_z = torch.tensor([[[[-1,  0, 1],
                              [-3,  0, 3],
                              [-1,  0, 1]],
                             [[-1,  0, 1],
                              [-3,  0, 3],
                              [-1,  0, 1]],
                             [[-1,  0, 1],
                              [-3,  0, 3],
                              [-1,  0, 1]]]], dtype=torch.float32, device=aligned_input.device)
    
    C = aligned_input.shape[1]
    sobel_x = sobel_x.repeat(C, 1, 1, 1, 1)
    sobel_y = sobel_y.repeat(C, 1, 1, 1, 1)
    sobel_z = sobel_z.repeat(C, 1, 1, 1, 1)
    
    grad_aligned_x = F.conv3d(aligned_input, sobel_x, padding=1, stride=1)
    grad_aligned_y = F.conv3d(aligned_input, sobel_y, padding=1, stride=1)
    grad_aligned_z = F.conv3d(aligned_input, sobel_z, padding=1, stride=1)
    
    grad_target_x = F.conv3d(targets, sobel_x, padding=1, stride=1)
    grad_target_y = F.conv3d(targets, sobel_y, padding=1, stride=1)
    grad_target_z = F.conv3d(targets, sobel_z, padding=1, stride=1)
    diff_x = torch.abs(grad_aligned_x - grad_target_x)
    diff_y = torch.abs(grad_aligned_y - grad_target_y)
    diff_z = torch.abs(grad_aligned_z - grad_target_z)
    
    total_diff = diff_x.sum() + diff_y.sum() + diff_z.sum()
    num_elements = diff_x.numel() + diff_y.numel() + diff_z.numel()
    gdl = total_diff / num_elements
    
    return gdl

