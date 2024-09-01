import os, sys
import numpy as np
import argparse

def recon(path, out_path, normal=False, rembg=False, real=False, cpu=8):
    img_path=path
    #out_path=os.path.join(os.path.dirname(path),'recon',os.path.basename(path).split('.')[0])
    out_path=os.path.join(out_path,os.path.basename(path).split('.')[0])
    if 'lower' in os.path.basename(path):
        pose='../meta_info/lower.pkl'
    else:
        pose='../meta_info/upper.pkl'
    if rembg:
        rembg=' --rembg'
    else:
        rembg=''
    if real:
        real=' --real'
        pose='../meta_info/cond.pkl'
    else:
        real=''
    if normal:
        config='configs/neus-blender-normal.yaml'
        os.system(f'python tools.py --input {img_path} --output {out_path} --pose {pose} --normal'+rembg+real)
    else:
        config='configs/neus-blender.yaml'
        os.system(f'python tools.py --input {img_path} --output {out_path} --pose {pose}'+rembg+real)
    
    runs_dir=os.path.join(out_path,'log')
    exp_dir=out_path
    root_dir=out_path
    name=os.path.basename(path).split('.')[0]
    if real:
        os.system(f'python launch.py --config {config} --gpu 0 --train --runs_dir={runs_dir} dataset.scene={name} dataset.root_dir={root_dir} trial_name=neus exp_dir={exp_dir} dataset.img_wh=[1024,1024] dataset.num_workers={cpu}')
    else:
        os.system(f'python launch.py --config {config} --gpu 0 --train --runs_dir={runs_dir} dataset.scene={name} dataset.root_dir={root_dir} trial_name=neus exp_dir={exp_dir} trainer.max_steps=20000 dataset.num_workers={cpu}')
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='path to img file')
    parser.add_argument('--cpu', required=True, help='number of CPUs')
    parser.add_argument('--dir', required=True, help='path to reconstruction')
    parser.add_argument('--normal', action='store_true', help='normal maps exist or not')
    parser.add_argument('--rembg', action='store_true', help='remove background of img or not')
    parser.add_argument('--real', action='store_true', help='four intra-oral photos (False) or generated images (True)')
    args = parser.parse_args()
    
    sys.path.append(os.getcwd())
    os.environ['OMP_NUM_THREADS'] = args.cpu
    recon(args.img, args.dir, args.normal, args.rembg, args.real, int(args.cpu))
    
    