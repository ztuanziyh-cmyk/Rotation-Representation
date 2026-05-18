import os
import random
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler
import traceback

# Models for ablation study
# Euler
from models.se_transformer_euler_nonsampling import SETransformerEulerNonSampling 
from models.se_transformer_euler_sampling import SETransformerEulerSampling


# R9
from models.se_transformer_r9_nonsampling import SETransformerR9NonSampling
from models.se_transformer_r9_sampling import SETransformerR9Sampling

# R6
from models.se_transformer_r6_sampling import SETransformerR6Sampling
from models.se_transformer_r6_nonsampling import SETransformerR6NonSampling


import wandb
import time
######## MultiProc ##########
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
######## MultiProc ##########

from scripts.utils_ablation import (
    rotate_tensor, shift_tensor,
    transformation_loss, cross_correlation_loss,
    alignment_eval, gradient_difference_loss,
    shear_tensor, compose_rotation, matrix_to_r6, r6_to_matrix 
)

import argparse
if dist.is_initialized() and dist.get_rank() != 0:
    os.environ['WANDB_MODE'] = 'disabled'
    
# ===========================
# Hyperparameters and Settings
# ===========================
def debug_print(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)
        
parser = argparse.ArgumentParser(description='Training script for SETransformer model.')

# Hyperparameters
parser.add_argument('--pretrain_batch_size', type=int, default=4, help='Batch size for pretraining.')
parser.add_argument('--finetune_batch_size', type=int, default=4, help='Batch size for fine-tuning.')
parser.add_argument('--test_batch_size', type=int, default=4, help='Batch size for testing.')
parser.add_argument('--patch_size', type=int, nargs=3, default=(4, 4, 4), help='Patch size for the model.')

parser.add_argument('--pretrain_epochs', type=int, default=50, help='Number of epochs for pretraining.')
parser.add_argument('--finetune_epochs', type=int, default=50, help='Number of epochs for fine-tuning.')

parser.add_argument('--learning_rate_pretrain', type=float, default=1e-5, help='Learning rate for pretraining.')
parser.add_argument('--weight_decay_pretrain', type=float, default=2e-8, help='Weight decay for pretraining.')
parser.add_argument('--learning_rate_finetune', type=float, default=1e-5, help='Learning rate for fine-tuning.')
parser.add_argument('--weight_decay_finetune', type=float, default=2e-8, help='Weight decay for fine-tuning.')
parser.add_argument('--best_model_path_pretrain', type=str, default='best_model_pretrain.pth', help='Path to save the best pre-trained model.')
parser.add_argument('--best_model_path_finetune', type=str, default='best_finetune.pth', help='Path to save the best fine-tuned model.')
parser.add_argument('--run_name', type=str, default='Test_Run', help='Name of the run for logging.')
parser.add_argument('--data_dir', type=str, default='./data', help='Base directory for dataset')
parser.add_argument('--output_dir', type=str, default='./logs', help='Directory for saving outputs')
parser.add_argument('--pretrained_dir', type=str, default='./pretrained', help='Directory for pretrained models')

# Flags to determine architecture variants
parser.add_argument('--use_liere', action='store_true', help='Use LIERE/MARE in model architecture.')
parser.add_argument('--use_polyshift', action='store_true', help='Use polyphase anchoring in model architecture.')
parser.add_argument('--use_cross_attention', action='store_true', help='Use cross attention if true and self attention if false.')
parser.add_argument('--use_v2', action='store_true', help='Use MARE v2 in model architecture.')

# Model parameters
parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels.')
parser.add_argument('--num_transformer_blocks', type=int, default=4, help='Number of transformer blocks.')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads.')
parser.add_argument('--ff_hidden_dim', type=int, default=256, help='Hidden dimension of the feedforward network.')
parser.add_argument('--hidden_dim', type=int, default=60, help='Hidden dimension of the transformer network.')

# Augmentation parameters
parser.add_argument('--num_augmentations_pretrain', type=int, default=10, help='Number of augmentations for pre-training.')
parser.add_argument('--rotation_range_pretrain', type=float, default=90.0, help='Rotation range in degrees for pre-training.')
parser.add_argument('--translation_range_pretrain', type=float, default=0.3, help='Translation range as fraction of D, H, W for pre-training.')
parser.add_argument('--shearing_range_pretrain', type=float, default=0.05, help='Shearing range for pre-training.')
parser.add_argument('--num_augmentations_finetune', type=int, default=10, help='Number of augmentations for fine-tuning.')
parser.add_argument('--rotation_range_finetune', type=float, default=15.0, help='Rotation range in degrees for fine-tuning.')
parser.add_argument('--translation_range_finetune', type=float, default=0.2, help='Translation range as fraction of D, H, W for fine-tuning.')
parser.add_argument('--shearing_range_finetune', type=float, default=0.025, help='Shearing range for fine-tuning.')
parser.add_argument('--rotation_range_test', type=float, default=30.0, help='Rotation range in degrees for test set.')
parser.add_argument('--translation_range_test', type=float, default=0.3, help='Translation range as fraction of D, H, W for test set.')

parser.add_argument('--resume_pretrain', type=str, default=None, 
                   help='Path to resume pretraining from checkpoint.')
parser.add_argument('--resume_finetune', type=str, default=None, 
                   help='Path to resume finetuning from checkpoint.')
parser.add_argument('--force_train', action='store_true', help='Force training even if best model exists')

parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)), 
                   help='Local rank for distributed training')
parser.add_argument('--world_size', type=int, default=2, help='Number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', help='Distributed backend')

parser.add_argument('--wandb_project', type=str, default='setransformer_r9', 
                   help='WandB project name')
parser.add_argument('--wandb_entity', type=str, default=None, 
                   help='WandB entity/team name')

parser.add_argument('--transform_type', type=str, choices=['r9', 'euler', 'r6'], default='r9',
                    help='Which rotation representation to use: r9, euler, or r6.')
parser.add_argument('--use_sampling', action='store_true',
                        help='Use the sampling-based variant of SETransformer.')

args = parser.parse_args()

args.rotation_range_test = args.rotation_range_finetune
args.translation_range_test = args.translation_range_finetune

PRETRAIN_BATCH_SIZE = args.pretrain_batch_size
FINETUNE_BATCH_SIZE = args.finetune_batch_size
TEST_BATCH_SIZE = args.test_batch_size
PATCH_SIZE = tuple(args.patch_size)

PRETRAIN_EPOCHS = args.pretrain_epochs
FINETUNE_EPOCHS = args.finetune_epochs

LEARNING_RATE_PRETRAIN = args.learning_rate_pretrain
WEIGHT_DECAY_PRETRAIN = args.weight_decay_pretrain
LEARNING_RATE_FINETUNE = args.learning_rate_finetune
WEIGHT_DECAY_FINETUNE = args.weight_decay_finetune

BEST_MODEL_PATH_PRETRAIN = os.path.join(args.output_dir, args.best_model_path_pretrain)
BEST_MODEL_PATH_FINETUNE = os.path.join(args.output_dir, args.best_model_path_finetune)

RUN_NAME = args.run_name

USE_LIERE = args.use_liere
USE_POLYSHIFT = args.use_polyshift
USE_CROSS_ATTENTION = args.use_cross_attention
USE_V2 = args.use_v2

DEVICE = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')

EARLY_STOPPING_PATIENCE_PRETRAIN = 3
EARLY_STOPPING_PATIENCE_FINETUNE = 3

if args.local_rank == -1:
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = 0

def init_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print(f'Not using distributed mode')
        return

    torch.cuda.set_device(args.local_rank)
    args.dist_url = 'env://'
    args.dist_backend = 'gloo'
    dist.init_process_group(backend=args.dist_backend, 
                          init_method=args.dist_url,
                          world_size=args.world_size, 
                          rank=args.rank)
    dist.barrier()

def init_wandb(args):
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                config={
                    **vars(args),
                    'num_gpus': torch.cuda.device_count(),
                    'device_name': torch.cuda.get_device_name(args.local_rank),
                    'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
                }
            )
        except Exception as e:
            print(f"Warning: wandb initialization failed: {e}")
            print("Training will continue without wandb logging")

SEED = 2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
debug_print(f"Random seeds set to {SEED}")  

class DistributedMetricsTracker:
    def __init__(self, device):
        self.device = device
        self.reset()
        
    def reset(self):
        self.running_loss = torch.zeros(1, device=self.device)
        self.running_trans_loss = torch.zeros(1, device=self.device)
        self.count = torch.zeros(1, device=self.device)
        self.step = 0

    def update(self, loss, trans_loss, batch_size):
        if not torch.isfinite(loss):
            debug_print(f"Warning: Loss is {loss}, skipping update")
            return
        self.running_loss += loss.item() * batch_size
        self.running_trans_loss += trans_loss.item() * batch_size
        self.count += batch_size
        self.step += 1 
        
    def synchronize(self):
        try:
            if dist.is_initialized():
                dist.all_reduce(self.running_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.running_trans_loss, op=dist.ReduceOp.SUM)
                
                if dist.get_rank() == 0 and self.step % 5 == 0:
                    metrics = {
                        'loss': (self.running_loss / self.count).item(),
                        'trans_loss': (self.running_trans_loss / self.count).item()
                    }
                    if wandb.run is not None:
                        wandb.log(metrics)
        except Exception as e:
            print(f"Rank {dist.get_rank()}: Error in synchronize: {e}")
            raise
        
    def get_metrics(self):
        return {
            'loss': (self.running_loss / self.count).item(),
            'trans_loss': (self.running_trans_loss / self.count).item()
        }
def cleanup():
    if dist.is_initialized():
        try:
            dist.barrier()
            dist.destroy_process_group()
            if dist.get_rank() == 0:
                print("Distributed process group destroyed")
                wandb.finish()
        except Exception as e:
            print(f"Rank {dist.get_rank()}: Error during cleanup: {e}")
            dist.destroy_process_group()
            if dist.get_rank() == 0:
                print("Distributed process group destroyed")
                wandb.finish()
                
# ===========================
# Data Loading and Preparation
# ===========================
def load_data(file_path):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Starting to load data from {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Successfully loaded {len(data)} samples from {file_path}")
    except Exception as e:
        print(f"Error loading data from {file_path}: {str(e)}")
        raise
        
    subtomograms = [d['v'] for d in data]
    subtomograms = torch.tensor(np.stack(subtomograms, axis=0), dtype=torch.float32).unsqueeze(1)
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Processed data shape: {subtomograms.shape}")
    
    return subtomograms    

def custom_collate_fn(batch):
    filtered_batch = []
    for sample in batch:
        input_sample, target_sample, params = sample
        if torch.isnan(input_sample).any() or torch.isinf(input_sample).any():
            continue
        if torch.isnan(target_sample).any() or torch.isinf(target_sample).any():
            continue
        if torch.isnan(params).any() or torch.isinf(params).any():
            continue
        filtered_batch.append(sample)
    if not filtered_batch:
        raise ValueError("All samples in the batch are invalid.")
    inputs, targets, params = zip(*filtered_batch)
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    params = torch.stack(params, dim=0)
    return inputs, targets, params

def shear_tensor(x, shear_factors, device):
    B, C, D, H, W = x.shape
    sheared = []

    for i in range(B):
        shear_x, shear_y, shear_z = shear_factors[i]
        affine_matrix = torch.tensor([
            [1, shear_x, shear_y, 0],
            [0, 1, shear_z, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32, device=device)
        grid = F.affine_grid(affine_matrix.unsqueeze(0), x[i:i+1].size(), align_corners=True)
        sheared_sample = F.grid_sample(x[i:i+1], grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        sheared.append(sheared_sample)

    sheared = torch.cat(sheared, dim=0)
    return sheared


# ===========================
# Transformation Functions
# ===========================
def apply_pretrain_transformations(x, device, max_rotation_angle, translation_range, shearing_range, args):
    B, C, D, H, W = x.shape

    angles_deg = torch.FloatTensor(B, 3).uniform_(-max_rotation_angle, max_rotation_angle).to(device)
    angles_rad = angles_deg * math.pi / 180.0

    x_rot = rotate_tensor(x, angles_rad[:, 0], axes=(2, 3), device=device)
    x_rot = rotate_tensor(x_rot, angles_rad[:, 1], axes=(2, 4), device=device)
    x_rot = rotate_tensor(x_rot, angles_rad[:, 2], axes=(3, 4), device=device)

    #shear_factors = torch.FloatTensor(B, 3).uniform_(-shearing_range, shearing_range).to(device)
    #x_sheared = shear_tensor(x_rot, shear_factors, device=device)
    x_sheared = x_rot
    max_trans_d = D * translation_range
    max_trans_h = H * translation_range
    max_trans_w = W * translation_range
    trans_d = torch.FloatTensor(B).uniform_(-max_trans_d, max_trans_d).to(device)
    trans_h = torch.FloatTensor(B).uniform_(-max_trans_h, max_trans_h).to(device)
    trans_w = torch.FloatTensor(B).uniform_(-max_trans_w, max_trans_w).to(device)
    translations = torch.stack((trans_d, trans_h, trans_w), dim=1)
    x_translated = shift_tensor(x_sheared, translations, device=device)

    # noise_std = 0.05
    # noise = torch.randn_like(x_translated) * noise_std
    x_noisy = x_translated  # + noise

    transformed_input = x_noisy
    targets = x

    if args.transform_type == 'euler':
        params = torch.cat([angles_rad, translations], dim=1)

    elif args.transform_type == 'r9':
        r_mats = []
        for i in range(B):
            ax = angles_rad[i, 0].item()
            ay = angles_rad[i, 1].item()
            az = angles_rad[i, 2].item()
            Rb = compose_rotation(ax, ay, az, device)
            r_mats.append(Rb.reshape(1, 9))
        r9 = torch.cat(r_mats, dim=0)
        params = torch.cat([r9, translations], dim=1)

    elif args.transform_type == 'r6':
        r_mats = []
        for i in range(B):
            ax = angles_rad[i, 0].item()
            ay = angles_rad[i, 1].item()
            az = angles_rad[i, 2].item()
            Rb = compose_rotation(ax, ay, az, device)

            r6_once = matrix_to_r6(Rb.unsqueeze(0))            
            R_again = r6_to_matrix(r6_once)                    
            r6_final = matrix_to_r6(R_again)                   
        
            r_mats.append(r6_final)  

        r6 = torch.cat(r_mats, dim=0)
        params = torch.cat([r6, translations], dim=1)

    else:
        raise ValueError(f"Unsupported transform_type: {args.transform_type}")

    return transformed_input, targets, params

def apply_finetune_transformations(x, device, max_rotation_angle, translation_range, shearing_range, args):
    B, C, D, H, W = x.shape

    angles_deg = torch.FloatTensor(B, 3).uniform_(-max_rotation_angle, max_rotation_angle).to(device)
    angles_rad = angles_deg * math.pi / 180.0

    x_rot = rotate_tensor(x, angles_rad[:, 0], axes=(2, 3), device=device)
    x_rot = rotate_tensor(x_rot, angles_rad[:, 1], axes=(2, 4), device=device)
    x_rot = rotate_tensor(x_rot, angles_rad[:, 2], axes=(3, 4), device=device)

    # shear_factors = torch.FloatTensor(B, 3).uniform_(-shearing_range, shearing_range).to(device)
    # x_sheared = shear_tensor(x_rot, shear_factors, device=device)
    x_sheared = x_rot
    max_trans_d = D * translation_range
    max_trans_h = H * translation_range
    max_trans_w = W * translation_range
    trans_d = torch.FloatTensor(B).uniform_(-max_trans_d, max_trans_d).to(device)
    trans_h = torch.FloatTensor(B).uniform_(-max_trans_h, max_trans_h).to(device)
    trans_w = torch.FloatTensor(B).uniform_(-max_trans_w, max_trans_w).to(device)
    translations = torch.stack((trans_d, trans_h, trans_w), dim=1)
    x_translated = shift_tensor(x_sheared, translations, device=device)

    # noise_std = 0.01
    # noise = torch.randn_like(x_translated) * noise_std
    x_noisy = x_translated  # + noise

    transformed_input = x_noisy
    targets = x

    if args.transform_type == 'euler':
        params = torch.cat([angles_rad, translations], dim=1)
    elif args.transform_type == 'r9':
        r_mats = []
        for i in range(B):
            ax = angles_rad[i, 0].item()
            ay = angles_rad[i, 1].item()
            az = angles_rad[i, 2].item()
            Rb = compose_rotation(ax, ay, az, device)
            r_mats.append(Rb.view(1, 9))
        r9 = torch.cat(r_mats, dim=0)
        params = torch.cat([r9, translations], dim=1)
    elif args.transform_type == 'r6':
        r_mats = []
        for i in range(B):
            ax = angles_rad[i, 0].item()
            ay = angles_rad[i, 1].item()
            az = angles_rad[i, 2].item()
            Rb = compose_rotation(ax, ay, az, device)
            r6_once = matrix_to_r6(Rb.unsqueeze(0))
            R_again = r6_to_matrix(r6_once)
            r6_final = matrix_to_r6(R_again)
            

            r_mats.append(r6_final)
        r6 = torch.cat(r_mats, dim=0)
        params = torch.cat([r6, translations], dim=1)
    else:
        raise ValueError(f"Unsupported transform_type: {args.transform_type}")

    return transformed_input, targets, params

def apply_testset_transformations(x, device, max_rotation_angle, translation_range, args):
    B, C, D, H, W = x.shape

    angles_deg = torch.FloatTensor(B, 3).uniform_(-max_rotation_angle, max_rotation_angle).to(device)
    angles_rad = angles_deg * math.pi / 180.0

    x_rot = rotate_tensor(x, angles_rad[:, 0], axes=(2, 3), device=device)
    x_rot = rotate_tensor(x_rot, angles_rad[:, 1], axes=(2, 4), device=device)
    x_rot = rotate_tensor(x_rot, angles_rad[:, 2], axes=(3, 4), device=device)

    max_trans_d = D * translation_range
    max_trans_h = H * translation_range
    max_trans_w = W * translation_range

    trans_d = torch.FloatTensor(B).uniform_(-max_trans_d, max_trans_d).to(device)
    trans_h = torch.FloatTensor(B).uniform_(-max_trans_h, max_trans_h).to(device)
    trans_w = torch.FloatTensor(B).uniform_(-max_trans_w, max_trans_w).to(device)
    translations = torch.stack((trans_d, trans_h, trans_w), dim=1)

    x_translated = shift_tensor(x_rot, translations, device=device)
    transformed_input = x_translated
    targets = x

    if args.transform_type == 'euler':
        params = torch.cat([angles_rad, translations], dim=1)
    elif args.transform_type == 'r9':
        r_mats = []
        for i in range(B):
            ax = angles_rad[i, 0].item()
            ay = angles_rad[i, 1].item()
            az = angles_rad[i, 2].item()
            Rb = compose_rotation(ax, ay, az, device)
            r_mats.append(Rb.reshape(1, 9))
        r9 = torch.cat(r_mats, dim=0)
        params = torch.cat([r9, translations], dim=1)
    elif args.transform_type == 'r6':
        r_mats = []
        for i in range(B):
            ax = angles_rad[i, 0].item()
            ay = angles_rad[i, 1].item()
            az = angles_rad[i, 2].item()
            Rb = compose_rotation(ax, ay, az, device)
            r6_once = matrix_to_r6(Rb.unsqueeze(0))
            R_again = r6_to_matrix(r6_once)
            r6_final = matrix_to_r6(R_again)
            r_mats.append(r6_final)
        r6 = torch.cat(r_mats, dim=0)
        params = torch.cat([r6, translations], dim=1)
    else:
        raise ValueError(f"Unsupported transform_type: {args.transform_type}")

    return transformed_input, targets, params

def create_self_supervised_pairs(subtomograms, pretrain, num_augmentations, test, max_rotation_angle, translation_range, shearing_range):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Creating pairs with {num_augmentations} augmentations...")
    
    batch_size = 5
    num_batches = (num_augmentations + batch_size - 1) // batch_size
    
    transformed_inputs_list = []
    targets_list = []
    transformation_params_list = []

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_augmentations)
        current_batch_size = end_idx - start_idx
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Processing batch {batch + 1}/{num_batches} (augmentations {start_idx}-{end_idx})")
        batch_transformed_list = []
        batch_targets_list = []
        batch_params_list = []
        
        for aug_idx in range(current_batch_size):
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Processing augmentation {start_idx + aug_idx}/{num_augmentations}")
                
            if test:
                transformed_inputs, targets, transformation_params = apply_testset_transformations(
                    subtomograms, device='cpu', max_rotation_angle=max_rotation_angle, translation_range=translation_range, args = args
                )
            else:
                if pretrain:
                    transformed_inputs, targets, transformation_params = apply_pretrain_transformations(
                        subtomograms, device='cpu', max_rotation_angle=max_rotation_angle, 
                        translation_range=translation_range, shearing_range=shearing_range, args = args
                    )
                else:
                    transformed_inputs, targets, transformation_params = apply_finetune_transformations(
                        subtomograms, device='cpu', max_rotation_angle=max_rotation_angle, 
                        translation_range=translation_range, shearing_range=shearing_range, args = args
                    )
            
            batch_transformed_list.append(transformed_inputs)
            batch_targets_list.append(targets)
            batch_params_list.append(transformation_params)
            
            torch.cuda.empty_cache()
            
            if dist.is_initialized():
                dist.barrier()
                
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"  Augmentation {start_idx + aug_idx} completed")
        
        transformed_inputs_list.extend(batch_transformed_list)
        targets_list.extend(batch_targets_list)
        transformation_params_list.extend(batch_params_list)
        
        del batch_transformed_list
        del batch_targets_list
        del batch_params_list
        torch.cuda.empty_cache()
        
        if dist.is_initialized():
            dist.barrier()
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Batch {batch + 1}/{num_batches} completed")


    if transformed_inputs_list:
        transformed_inputs = torch.cat(transformed_inputs_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        transformation_params = torch.cat(transformation_params_list, dim=0)
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Data augmentation completed. Final shapes: {transformed_inputs.shape}")
        
        return transformed_inputs, targets, transformation_params
    else:
        raise RuntimeError("No data was processed in create_self_supervised_pairs")

def prepare_datasets(train_file_path, valid_file_path, test_file_paths, split_ratio=0.4, seed=2):
    if dist.is_initialized():
        dist.barrier()

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("\nInitial GPU Memory Status:")
        print(torch.cuda.memory_summary())
    #debug_print("prepare_datasets() called")  
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_subtomograms = load_data(train_file_path)
    valid_subtomograms = load_data(valid_file_path)

    test_subtomograms_list = []
    for test_path in test_file_paths:
        test_subtomograms = load_data(test_path)
        test_subtomograms_list.append(test_subtomograms)

    def split_tensor(tensor, ratio, seed):
        torch.manual_seed(seed)
        shuffled_indices = torch.randperm(tensor.size(0))
        split_point = int(ratio * tensor.size(0))
        first_indices = shuffled_indices[:split_point]
        second_indices = shuffled_indices[split_point:]
        return tensor[first_indices], tensor[second_indices]
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Starting to split data...")
    pretrain_train_subtomograms, finetune_train_subtomograms = split_tensor(train_subtomograms, split_ratio, seed)
    pretrain_valid_subtomograms, finetune_valid_subtomograms = split_tensor(valid_subtomograms, split_ratio, seed)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Data split completed.")
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating self-supervised pairs for pretraining...")

    transformed_pretrain_train, targets_pretrain_train, params_pretrain_train = create_self_supervised_pairs(
        pretrain_train_subtomograms, pretrain=True, num_augmentations=args.num_augmentations_pretrain, test=False,
        max_rotation_angle=args.rotation_range_pretrain, translation_range=args.translation_range_pretrain, 
        shearing_range=args.shearing_range_pretrain
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating self-supervised pairs for pretraining valid...")
    transformed_pretrain_valid, targets_pretrain_valid, params_pretrain_valid = create_self_supervised_pairs(
        pretrain_valid_subtomograms, pretrain=True, num_augmentations=args.num_augmentations_pretrain, test=False,
        max_rotation_angle=args.rotation_range_pretrain, translation_range=args.translation_range_pretrain, 
        shearing_range=args.shearing_range_pretrain
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating self-supervised pairs for finetuning...")
    transformed_finetune_train, targets_finetune_train, params_finetune_train = create_self_supervised_pairs(
        finetune_train_subtomograms, pretrain=False, num_augmentations=args.num_augmentations_finetune, test=False,
        max_rotation_angle=args.rotation_range_finetune, translation_range=args.translation_range_finetune, 
        shearing_range=args.shearing_range_finetune
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating self-supervised pairs for finetuning valid...")
    transformed_finetune_valid, targets_finetune_valid, params_finetune_valid = create_self_supervised_pairs(
        finetune_valid_subtomograms, pretrain=False, num_augmentations=args.num_augmentations_finetune, test=False,
        max_rotation_angle=args.rotation_range_finetune, translation_range=args.translation_range_finetune, 
        shearing_range=args.shearing_range_finetune
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Pretraining pairs created.")
    transformed_finetune_test_list = []
    targets_finetune_test_list = []
    params_finetune_test_list = []
    for test_subtomograms in test_subtomograms_list:
        transformed_test, targets_test, params_test = create_self_supervised_pairs(
            test_subtomograms, pretrain=False, num_augmentations=1, test=True,
            max_rotation_angle=args.rotation_range_test, translation_range=args.translation_range_test, 
            shearing_range=args.shearing_range_finetune
        )
        transformed_finetune_test_list.append(transformed_test)
        targets_finetune_test_list.append(targets_test)
        params_finetune_test_list.append(params_test)

    
    debug_print("\n===== Augmented Dataset Sizes ======")
    debug_print(f"Pretraining Training Set: {transformed_pretrain_train.shape[0]} samples")
    debug_print(f"Pretraining Validation Set: {transformed_pretrain_valid.shape[0]} samples")
    debug_print(f"Fine-Tuning Training Set: {transformed_finetune_train.shape[0]} samples")
    debug_print(f"Fine-Tuning Validation Set: {transformed_finetune_valid.shape[0]} samples")
    for idx, test_set in enumerate(transformed_finetune_test_list):
        debug_print(f"Fine-Tuning Test Set {idx+1}: {test_set.shape[0]} samples")
    debug_print("====================================\n")
    
    trans_pre_mean = params_pretrain_train.mean(dim=0, keepdim=True)
    trans_pre_std = params_pretrain_train.std(dim=0, keepdim=True)
    
    if dist.is_initialized():
        dist.all_reduce(trans_pre_mean, op=dist.ReduceOp.SUM)
        dist.all_reduce(trans_pre_std, op=dist.ReduceOp.SUM)
        
        world_size = dist.get_world_size()
        trans_pre_mean /= world_size
        trans_pre_std /= world_size
    
    trans_fine_mean = params_finetune_train.mean(dim=0, keepdim=True)
    trans_fine_std = params_finetune_train.std(dim=0, keepdim=True)
    
    if dist.is_initialized():
        dist.all_reduce(trans_fine_mean, op=dist.ReduceOp.SUM)
        dist.all_reduce(trans_fine_std, op=dist.ReduceOp.SUM)
        
        world_size = dist.get_world_size()
        trans_fine_mean /= world_size
        trans_fine_std /= world_size
    
    params_pretrain_train_normalized = (params_pretrain_train - trans_pre_mean) / trans_pre_std
    params_pretrain_valid_normalized = (params_pretrain_valid - trans_pre_mean) / trans_pre_std
    
    params_finetune_train_normalized = (params_finetune_train - trans_fine_mean) / trans_fine_std
    params_finetune_valid_normalized = (params_finetune_valid - trans_fine_mean) / trans_fine_std
    
    params_finetune_test_normalized_list = []
    for params_test in params_finetune_test_list:
        params_test_normalized = (params_test - trans_fine_mean) / trans_fine_std
        params_finetune_test_normalized_list.append(params_test_normalized)
    
    mean_pre = transformed_pretrain_train.mean()
    std_pre = transformed_pretrain_train.std()
    
    if dist.is_initialized():
        mean_pre = torch.tensor([mean_pre], device=transformed_pretrain_train.device)
        std_pre = torch.tensor([std_pre], device=transformed_pretrain_train.device)
        
        dist.all_reduce(mean_pre, op=dist.ReduceOp.SUM)
        dist.all_reduce(std_pre, op=dist.ReduceOp.SUM)
        
        world_size = dist.get_world_size()
        mean_pre = mean_pre.item() / world_size
        std_pre = std_pre.item() / world_size
    
    transformed_pretrain_train = (transformed_pretrain_train - mean_pre) / std_pre
    targets_pretrain_train = (targets_pretrain_train - mean_pre) / std_pre
    transformed_pretrain_valid = (transformed_pretrain_valid - mean_pre) / std_pre
    targets_pretrain_valid = (targets_pretrain_valid - mean_pre) / std_pre
    
    mean_fine = transformed_finetune_train.mean()
    std_fine = transformed_finetune_train.std()
    
    if dist.is_initialized():
        mean_fine = torch.tensor([mean_fine], device=transformed_finetune_train.device)
        std_fine = torch.tensor([std_fine], device=transformed_finetune_train.device)
        
        dist.all_reduce(mean_fine, op=dist.ReduceOp.SUM)
        dist.all_reduce(std_fine, op=dist.ReduceOp.SUM)
        
        world_size = dist.get_world_size()
        mean_fine = mean_fine.item() / world_size
        std_fine = std_fine.item() / world_size
    
    transformed_finetune_train = (transformed_finetune_train - mean_fine) / std_fine
    targets_finetune_train = (targets_finetune_train - mean_fine) / std_fine
    transformed_finetune_valid = (transformed_finetune_valid - mean_fine) / std_fine
    targets_finetune_valid = (targets_finetune_valid - mean_fine) / std_fine

    transformed_finetune_test_normalized_list = []
    targets_finetune_test_normalized_list = []
    for transformed_test, targets_test in zip(transformed_finetune_test_list, targets_finetune_test_list):
        mean_test = transformed_test.mean()
        std_test = transformed_test.std()
        transformed_test_norm = (transformed_test - mean_test) / std_test
        targets_test_norm = (targets_test - mean_test) / std_test
        transformed_finetune_test_normalized_list.append(transformed_test_norm)
        targets_finetune_test_normalized_list.append(targets_test_norm)
    
    train_dataset_pretrain = TensorDataset(transformed_pretrain_train, targets_pretrain_train, params_pretrain_train_normalized)
    valid_dataset_pretrain = TensorDataset(transformed_pretrain_valid, targets_pretrain_valid, params_pretrain_valid_normalized)
    
    train_dataset_finetune = TensorDataset(transformed_finetune_train, targets_finetune_train, params_finetune_train_normalized)
    valid_dataset_finetune = TensorDataset(transformed_finetune_valid, targets_finetune_valid, params_finetune_valid_normalized)

    test_datasets_finetune = []
    for transformed_test, targets_test, params_test in zip(
        transformed_finetune_test_normalized_list,
        targets_finetune_test_normalized_list,
        params_finetune_test_normalized_list
    ):
        test_dataset = TensorDataset(transformed_test, targets_test, params_test)
        test_datasets_finetune.append(test_dataset)
        
    if dist.is_initialized():
        dist.barrier()
        
    return (
        train_dataset_pretrain,
        valid_dataset_pretrain,
        train_dataset_finetune,
        valid_dataset_finetune,
        test_datasets_finetune,
        trans_pre_mean, trans_pre_std,
        trans_fine_mean, trans_fine_std,
        mean_pre, std_pre,
        mean_fine, std_fine
    )

def get_dataloaders(train_pretrain, valid_pretrain, train_finetune, valid_finetune, test_datasets, 
                    pretrain_batch_size=4, finetune_batch_size=4, test_batch_size=4):
    if dist.is_initialized():
        train_pretrain_sampler = torch.utils.data.distributed.DistributedSampler(train_pretrain, shuffle=True)
        val_pretrain_sampler = torch.utils.data.distributed.DistributedSampler(valid_pretrain, shuffle=True)
        train_finetune_sampler = torch.utils.data.distributed.DistributedSampler(train_finetune, shuffle=True)
        val_finetune_sampler = torch.utils.data.distributed.DistributedSampler(valid_finetune, shuffle=True)

        train_pretrain_batch_sampler = torch.utils.data.BatchSampler(train_pretrain_sampler, pretrain_batch_size, drop_last=True)
        # val_pretrain_batch_sampler = torch.utils.data.BatchSampler(val_pretrain_sampler, pretrain_batch_size, drop_last=True)
        val_pretrain_batch_sampler = torch.utils.data.BatchSampler(val_pretrain_sampler, pretrain_batch_size, drop_last=False)
        train_finetune_batch_sampler = torch.utils.data.BatchSampler(train_finetune_sampler, finetune_batch_size, drop_last=True)
        # val_finetune_batch_sampler = torch.utils.data.BatchSampler(val_finetune_sampler, finetune_batch_size, drop_last=True)
        val_finetune_batch_sampler = torch.utils.data.BatchSampler(val_finetune_sampler, finetune_batch_size, drop_last=False)
        
        pretrain_train_loader = DataLoader(train_pretrain, batch_sampler=train_pretrain_batch_sampler, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        pretrain_valid_loader = DataLoader(valid_pretrain, batch_sampler=val_pretrain_batch_sampler, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        finetune_train_loader = DataLoader(train_finetune, batch_sampler=train_finetune_batch_sampler, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        finetune_valid_loader = DataLoader(valid_finetune, batch_sampler=val_finetune_batch_sampler, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

        test_loaders = []
        for test_dataset in test_datasets:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, shuffle=False)
            test_batch_sampler = torch.utils.data.BatchSampler(
                test_sampler, test_batch_size, drop_last=False)
            test_loader = DataLoader(
                test_dataset,
                batch_sampler=test_batch_sampler,
                num_workers=4,
                pin_memory=True,
                collate_fn=custom_collate_fn
            )
            test_loaders.append(test_loader)            
            
    else:
        pretrain_train_loader = DataLoader(train_pretrain, batch_size=pretrain_batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        pretrain_valid_loader = DataLoader(valid_pretrain, batch_size=pretrain_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        finetune_train_loader = DataLoader(train_finetune, batch_size=finetune_batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        finetune_valid_loader = DataLoader(valid_finetune, batch_size=finetune_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        test_loaders = []
        for test_dataset in test_datasets:
            test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, 
                                     num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
            test_loaders.append(test_loader)

    return pretrain_train_loader, pretrain_valid_loader, finetune_train_loader, finetune_valid_loader, test_loaders

# Different model initialization for different transform types
def initialize_model(args, patch_size, device):
    debug_print(f"Initializing model with transform_type={args.transform_type}, use_sampling={args.use_sampling}")

    if args.transform_type == 'r9':
        if args.use_sampling:
            debug_print("SETransformerR9Sampling (R9 + sampling)")
            model = SETransformerR9Sampling(
                in_channels=1,
                num_transformer_blocks=4,
                num_heads=8,
                ff_hidden_dim=512,
                hidden_dim=120,
                feature_type='vector',
                patch_size=patch_size
            )
        else:
            debug_print("SETransformerR9NonSampling (R9 + no sampling)")
            model = SETransformerR9NonSampling(
                in_channels=1,
                num_transformer_blocks=4,
                num_heads=8,
                ff_hidden_dim=512,
                hidden_dim=120,
                feature_type='vector',
                patch_size=patch_size
            )
    elif args.transform_type == 'euler':  
        if args.use_sampling:
            debug_print("SETransformerEulerSampling (Euler + sampling)")
            model = SETransformerEulerSampling(
                in_channels=1,
                num_transformer_blocks=4,
                num_heads=8,
                ff_hidden_dim=512,
                hidden_dim=120,
                feature_type='vector',
                patch_size=patch_size
            )
        else:
            debug_print("SETransformerEulerNonSampling (Euler + no sampling)")
            model = SETransformerEulerNonSampling(
                in_channels=1,
                num_transformer_blocks=4,
                num_heads=8,
                ff_hidden_dim=512,
                hidden_dim=120,
                feature_type='vector',
                patch_size=patch_size
            )
    elif args.transform_type == 'r6':
        if args.use_sampling:
            debug_print("SETransformerR6Sampling (R6 + sampling)")
            model = SETransformerR6Sampling(
                in_channels=1,
                num_transformer_blocks=4,
                num_heads=8,
                ff_hidden_dim=512,
                hidden_dim=120,
                feature_type='vector',
                patch_size=patch_size
            )
        else:
            debug_print("SETransformerR6NonSampling (R6 + no sampling)")
            model = SETransformerR6NonSampling(
                in_channels=1,
                num_transformer_blocks=4,
                num_heads=8,
                ff_hidden_dim=512,
                hidden_dim=120,
                feature_type='vector',
                patch_size=patch_size
            )
    else:
        raise ValueError(f"Unsupported transform_type: {args.transform_type}")

    model = model.to(device)
    if dist.is_initialized():
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=True
        )
    return model


def get_optimizer_and_loss(model, pretrain=True, transform_type='r9'):
    if pretrain:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_PRETRAIN, weight_decay=WEIGHT_DECAY_PRETRAIN)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_FINETUNE, weight_decay=WEIGHT_DECAY_FINETUNE)

    mse_loss_fn = lambda predictions, ground_truths: transformation_loss(predictions, ground_truths, transform_type=transform_type)
    cc_loss_fn = cross_correlation_loss
    alignment_loss_fn = gradient_difference_loss

    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=5, 
        factor=0.5, 
        verbose=(not dist.is_initialized() or dist.get_rank() == 0)
    )      
    return optimizer, mse_loss_fn, cc_loss_fn, alignment_loss_fn, scaler, scheduler

# ===========================
# Check points
# ===========================
def save_checkpoint(state, filename, rank):
    if rank == 0:
        torch.save(state, filename)
        debug_print(f"Checkpoint saved to '{filename}'")
        
# ============================================
# TRAIN_MODEL 
# ============================================
def train_model(
    model,
    train_loader,
    optimizer,
    mse_loss_fn,
    cc_loss_fn,
    alignment_loss_fn,
    scaler,
    scheduler,
    device,
    epoch_losses,
    trans_pre_mean,  
    trans_pre_std,
    phase='Training',
    transform_type='r9'
    ):
    model.train()
    metrics = DistributedMetricsTracker(device)
    all_trans_predictions = []

    alpha = 3.0   
    trans_pre_mean = trans_pre_mean.to(device)
    trans_pre_std = trans_pre_std.to(device)
    if hasattr(train_loader.batch_sampler, 'sampler'):
        train_loader.batch_sampler.sampler.set_epoch(len(epoch_losses['total']))
    torch.distributed.barrier()

    for batch_idx, (inputs, targets, params) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        params = params.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            transformation_pred, aligned_input = model(input=inputs, target=targets)
            if transform_type == 'r6':
                gt_transformation = params[:, :9].to(device)
            elif transform_type == 'r9':
                gt_transformation = params[:, :12].to(device)
            elif transform_type == 'euler':
                gt_transformation = params[:, :6].to(device)
            else:
                raise ValueError(f"Unsupported transform_type: {transform_type}")
            loss_transformation = mse_loss_fn(transformation_pred, gt_transformation)
            loss = alpha * loss_transformation
            
        all_trans_predictions.append(transformation_pred.detach())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if not (torch.isnan(loss).any() or 
                any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)):
            scaler.step(optimizer)
        scaler.update()

        metrics.update(loss.detach(), loss_transformation.detach(), inputs.size(0))
        torch.distributed.barrier()

        current_metrics = metrics.get_metrics()
        debug_print(f"{phase} - GPU{dist.get_rank()}: "
                  f"Batch {batch_idx+1}/{len(train_loader)} "
                  f"Loss: {current_metrics['loss']:.4f}")

    metrics.synchronize()
    final_metrics = metrics.get_metrics()

    if all_trans_predictions:
        all_trans_predictions = torch.cat(all_trans_predictions, dim=0)
        gathered_predictions = [torch.zeros_like(all_trans_predictions) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(gathered_predictions, all_trans_predictions)
        global_trans_predictions = torch.cat(gathered_predictions, dim=0)
        
        if dist.get_rank() == 0:
            trans_mean = global_trans_predictions.mean(dim=0)
            trans_std = global_trans_predictions.std(dim=0)
            print(f"{phase} - Epoch {len(epoch_losses['total']) + 1}\n"
                  f"Trans Mean: {trans_mean}\nTrans Std: {trans_std}")

    if dist.get_rank() == 0:
        epoch_losses['total'].append(final_metrics['loss'])
        epoch_losses['transformation'].append(final_metrics['trans_loss'])
        print(f"{phase} - Epoch Summary:\n"
              f"Avg Loss: {final_metrics['loss']:.4f}\n"
              f"Trans Loss: {final_metrics['trans_loss']:.4f}")

    torch.distributed.barrier()
    return final_metrics['loss']

# ============================================
# VALIDATE_MODEL 
# ============================================
def validate_model(
    model,
    valid_loader,
    mse_loss_fn,
    cc_loss_fn,
    alignment_loss_fn,
    device,
    epoch_losses,
    scheduler,
    trans_pre_mean,
    trans_pre_std,
    phase='Validation',
    transform_type='r9'
):
    model.eval()
    metrics = DistributedMetricsTracker(device)

    alpha = 3.0   
    trans_pre_mean = trans_pre_mean.to(device)
    trans_pre_std = trans_pre_std.to(device)

    torch.distributed.barrier()

    with torch.no_grad():
        for batch_idx, (inputs, targets, params) in enumerate(valid_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            params = params.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                transformation_pred, aligned_input = model(input=inputs, target=targets)
                if transform_type == 'r6':
                    gt_transformation = params[:, :9].to(device)
                elif transform_type == 'r9':
                    gt_transformation = params[:, :12].to(device)
                elif transform_type == 'euler':
                    gt_transformation = params[:, :6].to(device)
                else:
                    raise ValueError(f"Unsupported transform_type: {transform_type}")
                loss_transformation = mse_loss_fn(transformation_pred, gt_transformation)
                loss = alpha * loss_transformation
            
            metrics.update(loss.detach(), loss_transformation.detach(), inputs.size(0))

    metrics.synchronize()
    final_metrics = metrics.get_metrics()

    if dist.get_rank() == 0:
        epoch_losses['total'].append(final_metrics['loss'])
        epoch_losses['transformation'].append(final_metrics['trans_loss'])
        print(f"{phase} - Epoch Summary:\n"
              f"Avg Loss: {final_metrics['loss']:.4f}\n"
              f"Trans Loss: {final_metrics['trans_loss']:.4f}")

    scheduler.step(final_metrics['loss'])
    torch.distributed.barrier()
    return final_metrics['loss']

def test_model(model, test_loader, device, trans_fine_mean, trans_fine_std, test_file_name, transform_type='r9'):
    model.eval()
    local_size = 0
    predicted_trans = []
    ground_truth_trans = []

    trans_fine_mean = trans_fine_mean.to(device)
    trans_fine_std = trans_fine_std.to(device)
    
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: Device for trans_fine_mean: {trans_fine_mean.device}")
        print(f"Rank {dist.get_rank()}: Device for trans_fine_std: {trans_fine_std.device}")        
        print(f"test_model transform_type={transform_type}")
    
    try:
        with torch.no_grad():
            for batch_idx, (inputs, targets, params) in enumerate(test_loader):
                if dist.is_initialized():
                    print(f"Rank {dist.get_rank()}: Processing batch {batch_idx}")
                
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                params = params.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    transformation_pred, aligned_input = model(input=inputs, target=targets)
                    if transform_type == 'r6':
                        gt_transformation = params[:, :9]
                    elif transform_type == 'r9':
                        gt_transformation = params[:, :12]
                    elif transform_type == 'euler':
                        gt_transformation = params[:, :6]

                if torch.isnan(transformation_pred).any():
                    print(f"Rank {dist.get_rank()}: NaN detected in batch {batch_idx}")
                    continue

                transformation_pred_unnorm = transformation_pred * trans_fine_std + trans_fine_mean  
                gt_transformation_unnorm = gt_transformation * trans_fine_std + trans_fine_mean  

                predicted_trans.append(transformation_pred_unnorm.cpu())
                ground_truth_trans.append(gt_transformation_unnorm.cpu())
                local_size += transformation_pred_unnorm.size(0)

            print(f"Rank {dist.get_rank()}: Total processed {local_size} samples")

            if dist.is_initialized():
                if predicted_trans:
                    all_predictions = torch.cat(predicted_trans, dim=0)
                    all_ground_truths = torch.cat(ground_truth_trans, dim=0)
                    
                    print(f"Rank {dist.get_rank()}: Local tensor shapes - predictions: {all_predictions.shape}")

                    size_tensor = torch.tensor([all_predictions.shape[0]], device=device)
                    dist.all_reduce(size_tensor, op=dist.ReduceOp.MAX)
                    max_size = size_tensor.item()
                    
                    print(f"Rank {dist.get_rank()}: Max size across processes: {max_size}")

                    if all_predictions.shape[0] < max_size:
                        pad_size = max_size - all_predictions.shape[0]
                        print(f"Rank {dist.get_rank()}: Padding {pad_size} samples")
                        pad_predictions = torch.zeros((pad_size,) + all_predictions.shape[1:], 
                                                    dtype=all_predictions.dtype)
                        pad_ground_truths = torch.zeros((pad_size,) + all_ground_truths.shape[1:], 
                                                      dtype=all_ground_truths.dtype)
                        all_predictions = torch.cat([all_predictions, pad_predictions], dim=0)
                        all_ground_truths = torch.cat([all_ground_truths, pad_ground_truths], dim=0)

                    gathered_predictions = [torch.zeros_like(all_predictions) for _ in range(dist.get_world_size())]
                    gathered_ground_truths = [torch.zeros_like(all_ground_truths) for _ in range(dist.get_world_size())]
                    
                    dist.barrier()
                    print(f"Rank {dist.get_rank()}: Starting all_gather")
                    
                    dist.all_gather(gathered_predictions, all_predictions)
                    dist.all_gather(gathered_ground_truths, all_ground_truths)
                    
                    print(f"Rank {dist.get_rank()}: Completed all_gather")
                    
                    predicted_trans = gathered_predictions
                    ground_truth_trans = gathered_ground_truths
                else:
                    print(f"Rank {dist.get_rank()}: No predictions to gather")

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Testing for {test_file_name} completed.")

    except Exception as e:
        print(f"Error in test_model on rank {dist.get_rank()}: {str(e)}")
        print(f"Rank {dist.get_rank()}: Exception traceback:", traceback.format_exc())
        return None, None

    return predicted_trans, ground_truth_trans

def evaluate_model(predicted_trans, ground_truth_trans, image_size=32, test_file_name="Test", transform_type='r9'):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
        
    print(f"\nStarting evaluation of alignment quality for '{test_file_name}'...")
    # debug_print(f"evaluate_model using transform_type={transform_type}")

    
    # Validate predictions
    if not predicted_trans:
        print(f"No predictions to evaluate for '{test_file_name}'.")
        return
        
    try:
        predicted_trans = torch.cat(predicted_trans, dim=0).float()
        ground_truth_trans = torch.cat(ground_truth_trans, dim=0).float()
    except Exception as e:
        print(f"Error concatenating tensors: {e}")
        return

    if transform_type == 'r6':
        expected_dim = 9
    elif transform_type == 'r9':
        expected_dim = 12
    elif transform_type == 'euler':
        expected_dim = 6
    else:
        raise ValueError(f"Unsupported transform_type: {transform_type}")

    if ground_truth_trans.size(1) != expected_dim or predicted_trans.size(1) != expected_dim:
        print(f"[ERROR] Expected {expected_dim} transformation parameters, but got "
              f"{ground_truth_trans.size(1)} ground truth and {predicted_trans.size(1)} predicted "
              f"for '{test_file_name}'. Evaluation aborted.")
        return
    
    # Validate class distribution
    class_names = ['5LQW', '1I6V', '6A5L', '5T2C', '5MPA']
    total_samples = predicted_trans.size(0)
    
    if total_samples % len(class_names) != 0:
        print(f"Warning: Number of samples ({total_samples}) is not divisible by number of classes ({len(class_names)})")
        samples_per_class = total_samples // len(class_names)
        remainder = total_samples % len(class_names)
        print(f"Dropping last {remainder} samples to maintain balance")
    else:
        samples_per_class = total_samples // len(class_names)

    # Evaluate each class
    for i, class_name in enumerate(class_names):
        start_idx = i * samples_per_class
        end_idx = (i + 1) * samples_per_class
        
        if end_idx > total_samples:
            print(f"Warning: Skipping evaluation for {class_name} due to insufficient samples")
            continue
            
        predicted_trans_class = predicted_trans[start_idx:end_idx]
        ground_truth_trans_class = ground_truth_trans[start_idx:end_idx]

        print(f"\nEvaluating alignment quality for '{test_file_name}/{class_name}'...")
        try:
            alignment_results = alignment_eval(
                ground_truth_trans_class, 
                predicted_trans_class, 
                image_size=image_size, 
                scale=False,
                transform_type=transform_type
            )
        except Exception as e:
            print(f"Error in alignment evaluation for {class_name}: {e}")
            continue
# ===========================
# Main Execution Block 
# ===========================
def main():
    torch.backends.cudnn.enabled = True
    init_wandb(args)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        print(f"Distributed learning enabled: GPU{args.local_rank}.")
    else:
        print(f'Not using distributed mode')
        return

    torch.cuda.set_device(args.local_rank)

    torch.backends.cudnn.enabled = True
    if dist.get_rank()==0:
        print("cuDNN enabled.")

    # TODO:: Change paths
    TRAIN_FILE_PATH = '/home/runminj@andrew.cmu.edu/dataset/gum_jim/1000_0_100.pickle'
    VALID_FILE_PATH = '/home/runminj@andrew.cmu.edu/dataset/gum_jim/valid_0_100.pickle'
    TEST_FILE_PATHS = [
        '/home/runminj@andrew.cmu.edu/dataset/gum_jim/test_0_100.pickle',
        '/home/runminj@andrew.cmu.edu/dataset/gum_jim/test_0_01.pickle',
        '/home/runminj@andrew.cmu.edu/dataset/gum_jim/test_0_001.pickle',
        '/home/runminj@andrew.cmu.edu/dataset/gum_jim/test_0_003.pickle',
        '/home/runminj@andrew.cmu.edu/dataset/gum_jim/test_0_005.pickle',
    ]
    # TRAIN_FILE_PATH = '/shared/scratch/0/home/c_xingjia2/gum-net/data_xiangrui_folder/simulated/1000_0_100.pickle'
    # VALID_FILE_PATH = '/shared/scratch/0/home/c_xingjia2/gum-net/data_xiangrui_folder/simulated/valid_0_100.pickle'
    # TEST_FILE_PATHS = [
    #     '/shared/scratch/0/home/c_xingjia2/gum-net/data_xiangrui_folder/simulated/test_0_100.pickle',
    #     '/shared/scratch/0/home/c_xingjia2/gum-net/data_xiangrui_folder/simulated/test_0_01.pickle',
    #     '/shared/scratch/0/home/c_xingjia2/gum-net/data_xiangrui_folder/simulated/test_0_001.pickle',
    #     '/shared/scratch/0/home/c_xingjia2/gum-net/data_xiangrui_folder/simulated/test_0_003.pickle',
    #     '/shared/scratch/0/home/c_xingjia2/gum-net/data_xiangrui_folder/simulated/test_0_005.pickle',
    # ]

    if dist.get_rank() == 0:
        try:
            print("Successfully logged into Weights & Biases (mock).")
        except Exception as e:
            print(f"Error logging into Weights & Biases: {e}")
            print("wandb logging will be skipped.")

    os.makedirs(args.output_dir, exist_ok=True)
    
    (train_dataset_pretrain,
     valid_dataset_pretrain,
     train_dataset_finetune,
     valid_dataset_finetune,
     test_datasets_finetune,
     trans_pre_mean, trans_pre_std,
     trans_fine_mean, trans_fine_std,
     mean_pre, std_pre,
     mean_fine, std_fine) = prepare_datasets(
        train_file_path=TRAIN_FILE_PATH,
        valid_file_path=VALID_FILE_PATH,
        test_file_paths=TEST_FILE_PATHS,
        split_ratio=0.4,
        seed=SEED
    )
     
    pretrain_train_loader, pretrain_valid_loader, \
    finetune_train_loader, finetune_valid_loader, \
    test_loaders = get_dataloaders(
        train_pretrain=train_dataset_pretrain,
        valid_pretrain=valid_dataset_pretrain,
        train_finetune=train_dataset_finetune,
        valid_finetune=valid_dataset_finetune,
        test_datasets=test_datasets_finetune,
        pretrain_batch_size=PRETRAIN_BATCH_SIZE,
        finetune_batch_size=FINETUNE_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE
    )
    
    torch.cuda.empty_cache()
    debug_print(f"GPU{dist.get_rank()}: GPU memory allocated before model definition: {torch.cuda.memory_allocated(DEVICE) // 1e6} MiB")

    model = initialize_model(args=args, patch_size=PATCH_SIZE, device=DEVICE)
    debug_print(f"GPU{dist.get_rank()}: GPU memory allocated after model definition: {torch.cuda.memory_allocated(DEVICE) // 1e6} MiB")
    pretrain_exists = os.path.exists(BEST_MODEL_PATH_PRETRAIN)
    finetune_exists = os.path.exists(BEST_MODEL_PATH_FINETUNE)


    # ===========================
    # Pretraining Phase
    # ===========================
    pretrain_start_time = time.time() if torch.cuda.is_available() else None
    pretrain_runtime = 0.0
    
    optimizer_pretrain, mse_loss_fn_pretrain, cc_loss_fn_pretrain, alignment_loss_fn_pretrain, scaler_pretrain, scheduler_pretrain = get_optimizer_and_loss(model, pretrain=True, transform_type=args.transform_type)
    pretrain_epoch_losses = {'total': [], 'transformation': []}
    start_epoch = 0
    best_val_loss_pretrain = float('inf')
    epochs_no_improve_pretrain = 0
    
    if args.resume_pretrain and os.path.exists(args.resume_pretrain):
        debug_print(f"Resuming pretraining from checkpoint: {args.resume_pretrain}")
        checkpoint = torch.load(args.resume_pretrain, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer_pretrain.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler_pretrain.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss_pretrain = checkpoint.get('best_val_loss', float('inf'))
            epochs_no_improve_pretrain = checkpoint.get('epochs_no_improve', 0)
            pretrain_epoch_losses = checkpoint.get('epoch_losses', {'total': [], 'transformation': []})
            pretrain_runtime = checkpoint.get('runtime', 0.0)
            pretrain_start_time = time.time() if torch.cuda.is_available() else None
        else:
            debug_print(f"Warning: Invalid checkpoint format in {args.resume_pretrain}")
    elif pretrain_exists and not args.force_train:
        debug_print(f"Found an existing best pre-trained model at '{BEST_MODEL_PATH_PRETRAIN}'. Skipping pretraining.")
        checkpoint = torch.load(BEST_MODEL_PATH_PRETRAIN, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            best_val_loss_pretrain = checkpoint.get('best_val_loss', float('inf'))
        else:
            debug_print(f"Warning: Invalid checkpoint format in {BEST_MODEL_PATH_PRETRAIN}")
    else:
        debug_print("Starting Pre-Training Phase from scratch...")
    
    if args.resume_pretrain or not pretrain_exists or args.force_train:
        for epoch in range(start_epoch, PRETRAIN_EPOCHS):
            epoch_start_time = time.time() if torch.cuda.is_available() else None
            debug_print(f"\nPre-Training Epoch {epoch+1}/{PRETRAIN_EPOCHS}")
    
            train_loss = train_model(
                model=model,
                train_loader=pretrain_train_loader,
                optimizer=optimizer_pretrain,
                mse_loss_fn=mse_loss_fn_pretrain,
                cc_loss_fn=cc_loss_fn_pretrain,
                alignment_loss_fn=alignment_loss_fn_pretrain,
                scaler=scaler_pretrain,
                scheduler=scheduler_pretrain,
                device=DEVICE,
                epoch_losses=pretrain_epoch_losses,
                trans_pre_mean=trans_pre_mean,
                trans_pre_std=trans_pre_std,
                phase='Pre-Training',
                transform_type=args.transform_type  

            )
    
            val_loss = validate_model(
                model=model,
                valid_loader=pretrain_valid_loader,
                mse_loss_fn=mse_loss_fn_pretrain,
                cc_loss_fn=cc_loss_fn_pretrain,
                alignment_loss_fn=alignment_loss_fn_pretrain,
                device=DEVICE,
                epoch_losses=pretrain_epoch_losses,
                scheduler=scheduler_pretrain,
                trans_pre_mean=trans_pre_mean,
                trans_pre_std=trans_pre_std,
                phase='Pre-Training Validation',
                transform_type=args.transform_type  
            )
    
            if dist.get_rank() == 0:
                current_runtime = time.time() - epoch_start_time if epoch_start_time else 0
                pretrain_runtime += current_runtime
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer_pretrain.state_dict(),
                    'scheduler_state_dict': scheduler_pretrain.state_dict(),
                    'best_val_loss': best_val_loss_pretrain,
                    'epochs_no_improve': epochs_no_improve_pretrain,
                    'epoch_losses': pretrain_epoch_losses,
                    'runtime': pretrain_runtime
                }
                torch.save(checkpoint, os.path.join(args.output_dir, f"pretrain_epoch_{epoch+1}.pth"))
    
                if val_loss < best_val_loss_pretrain:
                    best_val_loss_pretrain = val_loss
                    epochs_no_improve_pretrain = 0
                    torch.save({
                        'model_state_dict': model.module.state_dict(),
                        'best_val_loss': best_val_loss_pretrain
                    }, BEST_MODEL_PATH_PRETRAIN)
                    print(f"Validation loss improved to {best_val_loss_pretrain:.4f}. Saving best pretrain model.")
                else:
                    epochs_no_improve_pretrain += 1
                    print(f"No improvement in validation loss for {epochs_no_improve_pretrain} epoch(s).")
    
                if epochs_no_improve_pretrain >= EARLY_STOPPING_PATIENCE_PRETRAIN:
                    print(f"Early stopping triggered after {epoch+1} epochs of no improvement.")
                    break
    
            torch.distributed.barrier()
        
        if dist.get_rank() == 0:
            if pretrain_runtime > 0:
                hours = pretrain_runtime // 3600
                minutes = (pretrain_runtime % 3600) // 60
                seconds = pretrain_runtime % 60
                debug_print(f"Pre-Training Phase Completed at epoch {epoch+1}. Total runtime: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
    
    torch.distributed.barrier()
    # ===========================
    # Fine-Tuning Phase
    # ===========================

    if dist.get_rank() == 0:
        if os.path.exists(BEST_MODEL_PATH_PRETRAIN):
            checkpoint = torch.load(BEST_MODEL_PATH_PRETRAIN, map_location='cpu', weights_only=True)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else {}
        else:
            state_dict = {}
    else:
        state_dict = {}
    
    state_dict_list = [state_dict]
    dist.broadcast_object_list(state_dict_list, src=0)
    state_dict = state_dict_list[0]
    
    if len(state_dict) > 0:
        model.module.load_state_dict(state_dict)
    
    torch.distributed.barrier()
    
    optimizer_finetune, mse_loss_fn_finetune, cc_loss_fn_finetune, alignment_loss_fn_finetune, scaler_finetune, scheduler_finetune = get_optimizer_and_loss(model, pretrain=False, transform_type=args.transform_type)
    finetune_epoch_losses = {'total': [], 'transformation': []}
    start_epoch = 0
    best_val_loss_finetune = float('inf')
    epochs_no_improve_finetune = 0
    finetune_runtime = 0.0
    finetune_start_time = time.time() if torch.cuda.is_available() else None
    
    if args.resume_finetune and os.path.exists(args.resume_finetune):
        debug_print(f"Resuming finetuning from checkpoint: {args.resume_finetune}")
        checkpoint = torch.load(args.resume_finetune, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer_finetune.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler_finetune.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss_finetune = checkpoint.get('best_val_loss', float('inf'))
            epochs_no_improve_finetune = checkpoint.get('epochs_no_improve', 0)
            finetune_epoch_losses = checkpoint.get('epoch_losses', {'total': [], 'transformation': []})
            finetune_runtime = checkpoint.get('runtime', 0.0)
            finetune_start_time = time.time() if torch.cuda.is_available() else None
        else:
            debug_print(f"Warning: Invalid checkpoint format in {args.resume_finetune}")
    elif finetune_exists and not args.force_train:
        debug_print(f"Found an existing best fine-tuned model at '{BEST_MODEL_PATH_FINETUNE}'. Skipping fine-tuning.")
        checkpoint = torch.load(BEST_MODEL_PATH_FINETUNE, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            best_val_loss_finetune = checkpoint.get('best_val_loss', float('inf'))
        else:
            debug_print(f"Warning: Invalid checkpoint format in {BEST_MODEL_PATH_FINETUNE}")
    else:
        debug_print("Starting Fine-Tuning Phase from scratch...")
    
    if args.resume_finetune or not finetune_exists or args.force_train:
        for epoch in range(start_epoch, FINETUNE_EPOCHS):
            epoch_start_time = time.time() if torch.cuda.is_available() else None
            debug_print(f"\nFine-Tuning Epoch {epoch+1}/{FINETUNE_EPOCHS}")
    
            train_loss = train_model(
                model=model,
                train_loader=finetune_train_loader,
                optimizer=optimizer_finetune,
                mse_loss_fn=mse_loss_fn_finetune,
                cc_loss_fn=cc_loss_fn_finetune,
                alignment_loss_fn=alignment_loss_fn_finetune,
                scaler=scaler_finetune,
                scheduler=scheduler_finetune,
                device=DEVICE,
                epoch_losses=finetune_epoch_losses,
                trans_pre_mean=trans_fine_mean,
                trans_pre_std=trans_fine_std,
                phase='Fine-Tuning',
                transform_type=args.transform_type  
            )
    
            val_loss = validate_model(
                model=model,
                valid_loader=finetune_valid_loader,
                mse_loss_fn=mse_loss_fn_finetune,
                cc_loss_fn=cc_loss_fn_finetune,
                alignment_loss_fn=alignment_loss_fn_finetune,
                device=DEVICE,
                epoch_losses=finetune_epoch_losses,
                scheduler=scheduler_finetune,
                trans_pre_mean=trans_fine_mean,
                trans_pre_std=trans_fine_std,
                phase='Fine-Tuning Validation',
                transform_type=args.transform_type  
            )
    
            if dist.get_rank() == 0:
                current_runtime = time.time() - epoch_start_time if epoch_start_time else 0
                finetune_runtime += current_runtime
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer_finetune.state_dict(),
                    'scheduler_state_dict': scheduler_finetune.state_dict(),
                    'best_val_loss': best_val_loss_finetune,
                    'epochs_no_improve': epochs_no_improve_finetune,
                    'epoch_losses': finetune_epoch_losses,
                    'runtime': finetune_runtime
                }
                torch.save(checkpoint, os.path.join(args.output_dir, f"finetune_epoch_{epoch+1}.pth"))
    
                if val_loss < best_val_loss_finetune:
                    best_val_loss_finetune = val_loss
                    epochs_no_improve_finetune = 0
                    torch.save({
                        'model_state_dict': model.module.state_dict(),
                        'best_val_loss': best_val_loss_finetune
                    }, BEST_MODEL_PATH_FINETUNE)
                    print(f"Validation loss improved to {best_val_loss_finetune:.4f}. Saving best finetune model.")
                else:
                    epochs_no_improve_finetune += 1
                    print(f"No improvement in validation loss for {epochs_no_improve_finetune} epoch(s).")
    
                if epochs_no_improve_finetune >= EARLY_STOPPING_PATIENCE_FINETUNE:
                    print(f"Early stopping triggered after {epoch+1} epochs of no improvement.")
                    break
    
            torch.distributed.barrier()
        
        if dist.get_rank() == 0:
            hours = finetune_runtime // 3600
            minutes = (finetune_runtime % 3600) // 60
            seconds = finetune_runtime % 60
            debug_print(f"Fine-Tuning Phase Completed. Total runtime: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
    
    torch.distributed.barrier()

    # ===========================
    # Testing and Evaluation
    # ==========================z
    if not (os.path.exists(BEST_MODEL_PATH_FINETUNE) or os.path.exists(BEST_MODEL_PATH_PRETRAIN)):
        print(f"GPU{dist.get_rank()}: No valid checkpoint found. Evaluation aborted.")
        return
    
    if os.path.exists(BEST_MODEL_PATH_FINETUNE):
        checkpoint = torch.load(BEST_MODEL_PATH_FINETUNE, map_location=f'cuda:{args.local_rank}', weights_only=True)
        if 'model_state_dict' not in checkpoint:
            print(f"GPU{dist.get_rank()}: Invalid checkpoint format in fine-tuned model")
            return
        model.module.load_state_dict(checkpoint['model_state_dict'])
        print(f"GPU{dist.get_rank()}: Loaded the best fine-tuned model from checkpoint for testing.")
    else:
        checkpoint = torch.load(BEST_MODEL_PATH_PRETRAIN, map_location=f'cuda:{args.local_rank}', weights_only=True)
        if 'model_state_dict' not in checkpoint:
            print(f"GPU{dist.get_rank()}: Invalid checkpoint format in pre-trained model")
            return
        model.module.load_state_dict(checkpoint['model_state_dict'])
        print(f"GPU{dist.get_rank()}: Loaded the best pre-trained model from checkpoint for testing.")
    for idx, test_loader in enumerate(test_loaders):
        test_file_path = TEST_FILE_PATHS[idx]
        test_file_name = os.path.basename(test_file_path)
        
        # Clear any previous state
        torch.cuda.empty_cache()
        
        try:
            predicted_trans_test, ground_truth_trans_test = test_model(
                model=model,
                test_loader=test_loader,
                device=torch.device(f'cuda:{args.local_rank}'),
                trans_fine_mean=trans_fine_mean.to(f'cuda:{args.local_rank}'),
                trans_fine_std=trans_fine_std.to(f'cuda:{args.local_rank}'),
                test_file_name=test_file_name,
                transform_type=args.transform_type  
            )
    
            # Ensure all processes have completed testing
            torch.distributed.barrier()
    
            if dist.get_rank() == 0 and predicted_trans_test is not None:
                evaluate_model(
                    predicted_trans=predicted_trans_test,
                    ground_truth_trans=ground_truth_trans_test,
                    image_size=32,
                    test_file_name=test_file_name,
                    transform_type=args.transform_type  
                )
        except Exception as e:
            print(f"Error processing test set {test_file_name}: {e}")
            continue
    
        # Ensure all processes are synchronized before next test set
        torch.distributed.barrier()
        
    if dist.get_rank() == 0:
        print("All processes completed successfully.")

if __name__ == "__main__":
    init_distributed()
    main()
