import os
import json
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank
import pickle

from skimage.io import imread, imsave

def camNormal2worldNormal(rot_c2w, camNormal):
    H,W,_ = camNormal.shape
    normal_img = np.matmul(rot_c2w[None, :, :], camNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

    return normal_img

def worldNormal2camNormal(rot_w2c, normal_map_world):
    H,W,_ = normal_map_world.shape
    # normal_img = np.matmul(rot_w2c[None, :, :], worldNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

    # faster version
    # Reshape the normal map into a 2D array where each row represents a normal vector
    normal_map_flat = normal_map_world.reshape(-1, 3)

    # Transform the normal vectors using the transformation matrix
    normal_map_camera_flat = np.dot(normal_map_flat, rot_w2c.T)

    # Reshape the transformed normal map back to its original shape
    normal_map_camera = normal_map_camera_flat.reshape(normal_map_world.shape)

    return normal_map_camera

def trans_normal(normal, RT_w2c_src, RT_w2c_tgt):

    normal_world = camNormal2worldNormal(np.linalg.inv(RT_w2c_src[:3,:3]), normal)
    normal_target_cam = worldNormal2camNormal(RT_w2c_tgt[:3,:3], normal_world)

    return normal_target_cam
    

def normal2img(normal):
    return ((normal*0.5+0.5)*255).to(torch.uint8)

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
 
class BlenderDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True
                
        with open(os.path.join(self.config.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)

        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 800, 800

        if 'img_wh' in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config:
            w, h = W // self.config.img_downscale, H // self.config.img_downscale
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")
        
        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)
        
        if self.config.view_weights!=None:
            self.view_weights = torch.from_numpy(np.array(self.config.view_weights)).float().to(self.rank).view(-1)
        else:
            self.view_weights = torch.from_numpy(np.ones(len(meta['frames']))).float().to(self.rank).view(-1)
        self.view_weights = self.view_weights.view(-1,1,1).repeat(1, self.h, self.w)
        
        self.near, self.far = self.config.near_plane, self.config.far_plane

        self.focal = 0.5 * w / math.tan(0.5 * meta['camera_angle_x']) # scaled focal length

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.w, self.h, self.focal, self.focal, self.w//2, self.h//2).to(self.rank) # (h, w, 3)           

        if not self.config.has_normal:
            self.all_c2w, self.all_images, self.all_fg_masks = [], [], []
            for i, frame in enumerate(meta['frames']):
                c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
                self.all_c2w.append(c2w)
    
                img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}")
                img = Image.open(img_path)
                img = img.resize(self.img_wh, Image.BICUBIC)
                img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
    
                self.all_fg_masks.append(img[..., -1]) # (h, w)
                self.all_images.append(img[...,:3])
    
            self.all_c2w, self.all_images, self.all_fg_masks = \
                torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
                torch.stack(self.all_images, dim=0).float().to(self.rank), \
                torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)
            
        else:
            self.all_c2w, self.all_images, self.all_normals, self.all_fg_masks, self.all_color_masks = [], [], [], [], []
            for i, frame in enumerate(meta['frames']):
                c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
                normal_path = os.path.join(self.config.root_dir, f"{frame['file_path']}")
                normal_path = normal_path.replace('.png','_normal.png')
                img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}")
                normal = Image.open(normal_path).resize(self.img_wh, Image.BICUBIC)
                img = Image.open(img_path).resize(self.img_wh, Image.BICUBIC)
                normal = TF.to_tensor(normal).permute(1, 2, 0)
                img = TF.to_tensor(img).permute(1, 2, 0)
                
                color_mask = img[:,:,-1]
                invalid_color_mask = color_mask < 0.5
                threshold =  torch.ones_like(img[:, :, 0]) * 250 / 255
                invalid_white_mask = (img[:, :, 0] > threshold) & (img[:, :, 1] > threshold) & (img[:, :, 2] > threshold)
                invalid_color_mask_final = invalid_color_mask & invalid_white_mask
                color_mask = (1 - invalid_color_mask_final.long()) > 0
                
                mask = normal[:,:,-1]
                normal = 2*normal[:,:,:-1]-1
                normal[mask==0] = torch.tensor([0,0,0],dtype=normal.dtype)
                mask = mask > 0.5
                
                if 'lower' in self.config.root_dir:
                    _,_,_,_,poses = read_pickle(f'/public/home/v-xuchf/Data/Teeth_recon/SyncDreamer/instant-nsr-pl/datasets/lower.pkl')
                else:
                    _,_,_,_,poses = read_pickle(f'/public/home/v-xuchf/Data/Teeth_recon/SyncDreamer/instant-nsr-pl/datasets/upper.pkl')
                
                #normal_world = camNormal2worldNormal(c2w[:3,:3], normal)
                normal_world = torch.clamp(camNormal2worldNormal(c2w[:3,:3], trans_normal(normal, poses[8], poses[i+8])),-1,1)
                normal_img = normal2img(normal_world)
                imsave(normal_path.replace('_normal.png','_world_normal.png'), normal_img.numpy())
                
                self.all_c2w.append(c2w)
                self.all_images.append(img[:,:,:3])
                self.all_normals.append(normal_world)
                self.all_fg_masks.append(mask)
                self.all_color_masks.append(color_mask)
                
            self.all_c2w, self.all_images, self.all_normals, self.all_fg_masks, self.all_color_masks = \
                torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
                torch.stack(self.all_images, dim=0).float().to(self.rank), \
                torch.stack(self.all_normals, dim=0).float().to(self.rank), \
                torch.stack(self.all_fg_masks, dim=0).float().to(self.rank), \
                torch.stack(self.all_color_masks, dim=0).float().to(self.rank)
                
class BlenderDataset(Dataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class BlenderIterableDataset(IterableDataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('blender')
class BlenderDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = BlenderIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BlenderDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = BlenderDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = BlenderDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=self.config.num_workers, 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
