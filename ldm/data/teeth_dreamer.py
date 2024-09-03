import pytorch_lightning as pl
import numpy as np
import torch
import PIL
import os
from skimage.io import imread
import webdataset as wds
import PIL.Image as Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from ldm.base_utils import read_pickle, pose_inverse, trans_normal
import torchvision.transforms as transforms
import torchvision
from einops import rearrange

from ldm.util import prepare_inputs
import random

class TeethDreamerTrainData(Dataset):
    def __init__(self, target_dir, input_dir, uid_set_pkl, normal_predict=False, image_size=256, bg_color='white'):
        self.default_image_size = 256
        self.image_size = image_size
        self.target_dir = Path(target_dir)
        self.input_dir = Path(input_dir)
        self.normal_predict = normal_predict
        self.bg_color = bg_color
        if 'mv' in uid_set_pkl:
            self.single_cond=False
        else:
            self.single_cond=True
        self.uids = read_pickle(uid_set_pkl)['train']
        
        print('============= length of train_dataset %d =============' % len(self.uids))

        image_transforms = []
        image_transforms.extend([transforms.ToTensor(), transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)
        self.num_images = 16

    def __len__(self):
        # return len(self.uids)
        return self.uids.shape[0]
    
    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif self.bg_color == 'three_choices':
            white = np.array([1., 1., 1.], dtype=np.float32)
            black = np.array([0., 0., 0.], dtype=np.float32)
            gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            bg_color = random.choice([white, black, gray])
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color
    
    def load_im(self, path, bg_color, normal=False, **kwargs):
        if normal:
            img = imread(path)
            img = img.astype(np.float32) / 255.0
            mask = img[:,:,3:].copy()
            normal = trans_normal(2*img[:,:,:3]-1, kwargs['RT_w2c_src'], kwargs['RT_w2c_tgt'])
            img = (normal*0.5 + 0.5).astype(np.float32)  # [0, 1]
            img = img * mask + (1 - mask) * bg_color 
            img = Image.fromarray(np.uint8(img * 255.))
        else:
            img = imread(path)
            img = img.astype(np.float32) / 255.0
            mask = img[:,:,3:]
            img[:,:,:3] = img[:,:,:3] * mask + (1 - mask) * bg_color 
            img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img, mask

    def process_im(self, im):
        im = im.convert("RGB")
        im = im.resize((self.image_size, self.image_size), resample=PIL.Image.BICUBIC)
        return self.image_transforms(im)

    def load_index(self, filename, index, bg_color, normal=False, **kwargs):
        if normal:
            img, _ = self.load_im(os.path.join(filename, '%03d_normal.png' % index), bg_color, normal=True, **kwargs)
        else:
            img, _ = self.load_im(os.path.join(filename, '%03d.png' % index), bg_color)
        img = self.process_im(img)
        return img
   
    def get_data_for_index(self, index):
        if self.single_cond:
            target_dir = os.path.join(self.target_dir, self.uids[index])
            input_dir = os.path.join(self.input_dir, self.uids[index])
        else:
            target_dir = os.path.join(self.target_dir, self.uids[index]+'_front')
            if 'lower' in self.uids[index]:
                ext=['_front', '_left', '_right', '_up']
            else:
                ext=['_front', '_left', '_right', '_down']
            input_dir = [os.path.join(self.input_dir, self.uids[index]+ext[i]) for i in range(4)]

        views = np.arange(0, self.num_images)
        bg_color = self.get_bg_color()
        # start_view_index = np.random.randint(0, self.num_images)
        # views = (views + start_view_index) % self.num_images
        target_images = []
        target_K, target_az, target_el, target_dis, target_pose = read_pickle(os.path.join(target_dir, f'meta.pkl'))
        
        if self.normal_predict:
            views = np.arange(8, self.num_images)
            for si, target_index in enumerate(views):
                img = self.load_index(target_dir, target_index, bg_color)
                target_images.append(img)
            
            for si, target_index in enumerate(views):
                img = self.load_index(target_dir.replace('target','normal'), target_index, bg_color, normal=True, RT_w2c_src=target_pose[target_index], RT_w2c_tgt=target_pose[views[0]])
                target_images.append(img)
            
            target_K, target_az, target_el, target_dis, target_pose = target_K, target_az[views], target_el[views], target_dis[views], target_pose[views]
        else:
            for si, target_index in enumerate(views):
                img = self.load_index(target_dir, target_index, bg_color)
                target_images.append(img)
        
        target_images = torch.stack(target_images, 0)
        # input_img = self.load_index(input_dir, start_view_index)
        if self.single_cond:
            idx = np.random.choice(self.num_images,1)
            input_img = self.load_index(input_dir, idx, bg_color)
        else:
            idx = np.random.choice(self.num_images,4,replace=False)
            input_img = torch.cat([self.load_index(input_dir[i], idx[i], bg_color) for i in range(4)],-1)

        # K, azimuths, elevations, distances, cam_poses = read_pickle(os.path.join(input_dir, f'meta.pkl'))
        if self.single_cond:
            meta_path=os.path.join(input_dir, f'meta.pkl')
        else:
            meta_path=os.path.join(input_dir[-1], f'meta.pkl')
        meta = read_pickle(meta_path)
        input_az=np.array([meta[1][idx[-1]]]).astype(np.float32)
        input_el=np.array([meta[2][idx[-1]]]).astype(np.float32)
        input_dis=np.array([meta[3][idx[-1]]]).astype(np.float32)
        
        return {"target_image": target_images, "input_image": input_img, "input_az":input_az, "input_el":input_el, "input_dis":input_dis, "target_K":target_K, "target_az":target_az, "target_el":target_el, "target_dis":target_dis, "target_pose":target_pose}

    def __getitem__(self, index):
        data = self.get_data_for_index(index)
        return data

class TeethDreamerEvalData(Dataset):
    def __init__(self, image_dir, uid_set_pkl, image_size, crop_size, normal_predict):
        self.image_size = image_size
        self.image_dir = Path(image_dir)
        self.crop_size = crop_size
        self.uids = read_pickle(uid_set_pkl)['val']
        
        self.num_images = 16
        self.normal_predict = normal_predict
        if 'mv' in uid_set_pkl:
            self.view_idx = np.random.randint(0, self.num_images, (self.uids.shape[0], 4))

        else:
            self.view_idx = np.random.randint(0, self.num_images, (self.uids.shape[0], 1))

        self.target_cams = [read_pickle(f'meta_info/upper.pkl'), read_pickle(f'meta_info/lower.pkl')]
        
        if self.normal_predict and (not 'mv' in uid_set_pkl):
            occ=[]
            for i in self.uids:
                if i.split('_')[-1]=='down' or i.split('_')[-1]=='up':
                    occ.append(i)
            self.uids=np.array(occ)

        print('============= length of val_dataset %d =============' % self.uids.shape[0])

    def __len__(self):
        return self.uids.shape[0]

    def get_data_for_index(self, index):
        name = self.uids[index]
        if self.view_idx.shape[1] == 4:
            if 'lower' in name:
                ext=['_front', '_left', '_right', '_up']
            else:
                ext=['_front', '_left', '_right', '_down']
            input_img_fn = [os.path.join(self.image_dir, name+ext[i], f'{self.view_idx[index][i]:03d}.png') for i in range(4)]
            meta = read_pickle((self.image_dir) / (name+ext[-1]) / 'meta.pkl')
        else:
            input_img_fn = [(self.image_dir) / name / f"{self.view_idx[index][-1]:03d}.png"]
            meta = read_pickle((self.image_dir) / name / 'meta.pkl')
        input_az=np.array([meta[1][self.view_idx[index][-1]]]).astype(np.float32)
        input_el=np.array([meta[2][self.view_idx[index]][-1]]).astype(np.float32)
        input_dis=np.array([meta[3][self.view_idx[index][-1]]]).astype(np.float32)
        input_cam = {"input_az":input_az, "input_el":input_el, "input_dis":input_dis}
        target_K, target_az, target_el, target_dis, target_pose = self.target_cams[0] if 'upper' in name else self.target_cams[1]
        
        if self.normal_predict:
            views = np.arange(8, self.num_images)
            #views = np.arange(0, self.num_images)
            target_K, target_az, target_el, target_dis, target_pose = target_K, target_az[views], target_el[views], target_dis[views], target_pose[views]
        
        target_cams = {"target_K":target_K, "target_az":target_az, "target_el":target_el, "target_dis":target_dis, "target_pose":target_pose}
        data=prepare_inputs(input_img_fn, input_cam, target_cams, crop_size=self.crop_size, image_size=self.image_size)
        
        #data.update({'name':np.array([index],dtype=np.int64), 'cond':self.view_idx[index].astype(dtype=np.int64)})
        
        return data 

    def __getitem__(self, index):
        return self.get_data_for_index(index)
        
        
class TeethDreamerTestData(Dataset):
    def __init__(self, image_dir, uid_set_npy, image_size, crop_size, normal_predict=False, single_cond=True):
        self.image_size = image_size
        self.image_dir = Path(image_dir)
        self.crop_size = crop_size

        self.uids = np.array(os.listdir(image_dir))
        self.num_images = 16
        self.normal_predict = normal_predict
        self.single_cond = single_cond
        self.view_idx = np.zeros((self.uids.shape[0], 4)).astype(np.int32)
        self.target_cams = [read_pickle(f'meta_info/upper.pkl'), read_pickle(f'meta_info/lower.pkl')]
        
        print('============= length of test_dataset %d =============' % self.uids.shape[0])

    def __len__(self):
        return self.uids.shape[0]

    def get_data_for_index(self, index):
        name = self.uids[index]
        
        if 'lower' in name:
            ext=['_front', '_left', '_right', '_up']
        else:
            ext=['_front', '_left', '_right', '_down']

        input_img_fn = [os.path.join(self.image_dir, name, f'{self.view_idx[index][i]:03d}.png') for i in range(4)]
        meta = read_pickle('meta_info/cond.pkl')
        if 'upper' in name:
            input_az=np.array([meta[1][3]]).astype(np.float32)
            input_el=np.array([meta[2][3]]).astype(np.float32)
            input_dis=np.array([meta[3][3]]).astype(np.float32)
        else:
            input_az=np.array([meta[1][4]]).astype(np.float32)
            input_el=np.array([meta[2][4]]).astype(np.float32)
            input_dis=np.array([meta[3][4]]).astype(np.float32)
        
        input_cam = {"input_az":input_az, "input_el":input_el, "input_dis":input_dis}
        target_K, target_az, target_el, target_dis, target_pose = self.target_cams[0] if 'upper' in name else self.target_cams[1]
        
        if self.normal_predict:
            views = np.arange(8, self.num_images)
            target_K, target_az, target_el, target_dis, target_pose = target_K, target_az[views], target_el[views], target_dis[views], target_pose[views]
        
        target_cams = {"target_K":target_K, "target_az":target_az, "target_el":target_el, "target_dis":target_dis, "target_pose":target_pose}
        data=prepare_inputs(input_img_fn, input_cam, target_cams, crop_size=self.crop_size, image_size=self.image_size)
        
        if not self.single_cond:
            data.update({'name':np.array([index],dtype=np.int64), 'cond':self.view_idx[index].astype(dtype=np.int64)})
        else:
            data.update({'name':np.array([index],dtype=np.int64), 'cond':self.view_idx[index][[-1]].astype(dtype=np.int64)})
        
        return data 

    def __getitem__(self, index):
        return self.get_data_for_index(index)
                


class TeethDreamerDataset(pl.LightningDataModule):
    def __init__(self, target_dir, input_dir, validation_dir, test_dir, batch_size, uid_set_pkl, test_set_npy, num_workers=4, normal_predict=False, single_cond=True, image_size=256, val_crop_size=200, seed=0, bg_color='white', **kwargs):
        super().__init__()
        self.target_dir = target_dir
        self.input_dir = input_dir
        self.validation_dir = validation_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normal_predict = normal_predict
        self.uid_set_pkl = uid_set_pkl
        self.seed = seed
        self.additional_args = kwargs
        self.image_size = image_size
        self.val_crop_size = val_crop_size
        self.bg_color = bg_color
        self.test_dir = test_dir
        self.test_set_npy = test_set_npy
        self.single_cond = single_cond

    def setup(self, stage):
        if stage in ['fit']:
            self.train_dataset = TeethDreamerTrainData(self.target_dir, self.input_dir, uid_set_pkl=self.uid_set_pkl, normal_predict=self.normal_predict, image_size=self.image_size, bg_color=self.bg_color)
            self.val_dataset = TeethDreamerEvalData(image_dir=self.validation_dir, uid_set_pkl=self.uid_set_pkl, image_size=self.image_size, crop_size=self.val_crop_size, normal_predict=self.normal_predict)
        else:
            self.test_dataset = TeethDreamerTestData(image_dir=self.test_dir, uid_set_npy=self.test_set_npy, image_size=self.image_size, crop_size=self.val_crop_size, normal_predict=self.normal_predict, single_cond=self.single_cond)

    def train_dataloader(self):
        #sampler = DistributedSampler(self.train_dataset, seed=self.seed)
        #return wds.WebLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)
        return wds.WebLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        loader = wds.WebLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return loader

    def test_dataloader(self):
        return wds.WebLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
