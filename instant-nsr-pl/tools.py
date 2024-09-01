from skimage.io import imread, imsave
import numpy as np
import os
import trimesh
import pickle
from collections import OrderedDict
import json
from rembg import remove
import shutil
import argparse

def merge(path):
    imgs=[]
    for i in range(16):
        imgs.append(imread(os.path.join(path,f"{i:03d}.png")))
    data=np.concatenate(imgs,1)
    imsave(os.path.join(path,'merge.png'),data)

def split(path):
    data=imread(path)
    image_size=data.shape[0]
    for i in range(16):
        imsave(os.path.join(os.path.dirname(path),f"{i:03d}.png"),data[:,i*image_size:(i+1)*image_size,:])

def normalize(path):
    try:
        l = trimesh.load(os.path.join(path, 'lower.stl'))
        u = trimesh.load(os.path.join(path, 'upper.stl'))
        t = trimesh.Scene([l, u])
        scale = (t.bounds[1] - t.bounds[0]).max()
        center = (t.bounds[1] + t.bounds[0]) / 2
        l.apply_translation(-center)
        u.apply_translation(-center)
        l.apply_scale(1 / scale)
        u.apply_scale(1 / scale)
        l.export(os.path.join(path, 'norm_lower.stl'))
        u.export(os.path.join(path, 'norm_upper.stl'))
        transform = {'center': center, 'scale': scale}
        np.save(os.path.join(path, 'transform.npy'), transform)
    except:
        l = trimesh.load(os.path.join(path, 'lower.ply'))
        u = trimesh.load(os.path.join(path, 'upper.ply'))
        t = trimesh.Scene([l, u])
        scale = (t.bounds[1] - t.bounds[0]).max()
        center = (t.bounds[1] + t.bounds[0]) / 2
        l.apply_translation(-center)
        u.apply_translation(-center)
        l.apply_scale(1 / scale)
        u.apply_scale(1 / scale)
        l.export(os.path.join(path, 'norm_lower.ply'))
        u.export(os.path.join(path, 'norm_upper.ply'))
        transform = {'center':center, 'scale':scale}
        np.save(os.path.join(path, 'transform.npy'), transform)

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
        
def get_c2w(R,t):
    R = R.T
    t = -R @ t
    R[:,1]=-R[:,1]
    R[:,2]=-R[:,2]
    return np.concatenate([np.concatenate([R, t[:, None]], 1), np.array([0, 0, 0, 1])[None, :]], 0)
    
def pkl2json(path, image_size, normal=False, real=''):
    json_dict = OrderedDict()
    json_dict['camera_angle_x'] = 0.8575560450553894
    json_dict['h'] = image_size
    json_dict['w'] = image_size
    json_dict['frames'] = []
    _,_,_,_,poses = read_pickle(path)
    #poses=poses[8:]
    if real=='':
        for i in range(8,poses.shape[0]):
        #for i in range(8,poses.shape[0],2):
            transform = get_c2w(poses[i,:,:3].copy(),poses[i,:,3].copy())
            json_dict['frames'].append({'file_path':f'train/{i:03}.png','transform_matrix':transform.tolist()})
    else:
        if real=='upper':
            idx=[0,1,2,3]
        else:
            idx=[0,1,2,4]
        for i,j in enumerate(idx):
            transform = get_c2w(poses[j,:,:3].copy(),poses[j,:,3].copy())
            json_dict['frames'].append({'file_path':f'train/{i:03}.png','transform_matrix':transform.tolist()})
    return json_dict
   
def prepare_masked_img(img_path, out_path, rembg=True, normal=False, real=False):
    if real:
        for i,name in enumerate(os.listdir(img_path)):
            data = imread(os.path.join(img_path,name))
            if rembg:
                rgb = data.copy()
                masked_img = remove(rgb)
            else:
                if data.shape[-1]!=4:
                    raise ValueError('There is no alpha channel (mask)!!!')
                else:
                    masked_img = data.copy()
            
            imsave(os.path.join(out_path,name),masked_img)
        
        return masked_img.shape[0]
    
    data=imread(img_path)
    image_size=data.shape[0]
    if not normal:
        for i in range(8,16):
        #for i in range(0,8):
            if rembg:
                rgb = np.copy(data[:,i*image_size:(i+1)*image_size,:3])
                masked_img = remove(rgb)
            else:
                if data.shape[-1]!=4:
                    raise ValueError('There is no alpha channel (mask)!!!')
                else:
                    masked_img = np.copy(data[:,i*image_size:(i+1)*image_size,:])
            
            imsave(os.path.join(out_path,f"{i:03d}.png"),masked_img)
            #imsave(os.path.join(out_path,f"{(i+8):03d}.png"),masked_img)
    else:
        for i in range(0,16):
        #for i in range(0,16,2):
            if rembg:
                rgb = np.copy(data[:,i*image_size:(i+1)*image_size,:3])
                masked_img = remove(rgb)
            else:
                if data.shape[-1]!=4:
                    raise ValueError('There is no alpha channel (mask)!!!')
                else:
                    masked_img = np.copy(data[:,i*image_size:(i+1)*image_size,:])
            if i>=8:
                imsave(os.path.join(out_path,f"{i:03d}_normal.png"),masked_img)
            else:
                imsave(os.path.join(out_path,f"{(i+8):03d}.png"),masked_img)
    
    return image_size

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)  # generated multi-view image merged by SyncDreamer
    parser.add_argument('--output', type=str, required=True) # path to save images to reconstruct
    parser.add_argument('--pose', type=str, required=True)   # 'lower.pkl' or 'upper.pkl' or 'cond.pkl'
    parser.add_argument('--rembg', action="store_true")
    parser.add_argument('--normal', action="store_true")
    parser.add_argument('--real', action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.output,exist_ok=True)
    os.makedirs(os.path.join(args.output, 'train'),exist_ok=True)
    os.makedirs(os.path.join(args.output, 'test'),exist_ok=True)
    os.makedirs(os.path.join(args.output, 'val'),exist_ok=True)
    
    image_size = prepare_masked_img(args.input, os.path.join(args.output, 'train'), args.rembg, args.normal, args.real)
    if args.real:
        if 'upper' in args.input:
            transforms = pkl2json(args.pose, image_size, args.normal, 'upper')
        else:
            transforms = pkl2json(args.pose, image_size, args.normal, 'lower')
    else:
        transforms = pkl2json(args.pose, image_size, args.normal)
    
    with open(os.path.join(args.output, "transforms_train.json"), 'w') as f:
        json.dump(transforms, f, indent=4, sort_keys=True)
    
    with open(os.path.join(args.output, "transforms_test.json"), 'w') as f:
        json.dump(transforms, f, indent=4, sort_keys=True)
    
    with open(os.path.join(args.output, "transforms_val.json"), 'w') as f:
        json.dump(transforms, f, indent=4, sort_keys=True)
    
    for i in os.listdir(os.path.join(args.output, 'train')):
        shutil.copy(os.path.join(args.output, 'train', i), os.path.join(args.output, 'test'))
        shutil.copy(os.path.join(args.output, 'train', i), os.path.join(args.output, 'val'))
    