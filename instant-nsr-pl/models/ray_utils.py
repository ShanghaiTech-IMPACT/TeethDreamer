import torch
import numpy as np


def cast_rays(ori, dir, z_vals):
    return ori[..., None, :] + z_vals[..., None] * dir[..., None, :]


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
        rays_o = c2w[:,:,3].expand(rays_d.shape)
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
            rays_o = c2w[None,None,:,3].expand(rays_d.shape)
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = (directions[None,:,:,None,:] * c2w[:,None,None,:3,:3]).sum(-1) # (B, H, W, 3)
            rays_o = c2w[:,None,None,:,3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d
    
def rotate_ray(xyz, angle, offset):
    xyz = xyz.reshape(angle.shape[0], -1 ,3)
    xyz_shape = xyz.shape
    angle_rad = torch.deg2rad(angle)
    trans_mat = torch.zeros((angle.shape[0],3,3)).float().to(angle.device)
    # x axis
    trans_mat[:,0,0] = 1
    trans_mat[:,1,1] = torch.cos(angle_rad)
    trans_mat[:,1,2] = -torch.sin(angle_rad)
    trans_mat[:,2,1] = torch.sin(angle_rad)
    trans_mat[:,2,2] = torch.cos(angle_rad)
    '''
    # z axis
    trans_mat[:,0,0] = torch.cos(angle_rad)
    trans_mat[:,1,1] = torch.cos(angle_rad)
    trans_mat[:,0,1] = -torch.sin(angle_rad)
    trans_mat[:,1,0] = torch.sin(angle_rad)
    trans_mat[:,2,2] = 1
    '''
    '''
    trans_mat = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0, 0, 1]
        ]
    )
    
    xyz = xyz.reshape(-1, 3)
    center = np.array([0, offset, 0],dtype=np.float32)[None].repeat(xyz.shape[0],0)
    xyz -= center
    xyz = (np.dot(xyz, trans_mat.T) + center).reshape(xyz_shape)
    '''
    center = torch.tensor([0, 0, 0]).float().to(angle.device).expand(*xyz_shape)
    xyz = torch.matmul(xyz-center, trans_mat.transpose(1,2)) + center
    return xyz.reshape(-1, 3)
