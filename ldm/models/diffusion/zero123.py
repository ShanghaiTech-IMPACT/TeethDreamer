from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.io import imsave
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ldm.base_utils import read_pickle, concat_images_list, camNormal2worldNormal
from ldm.models.diffusion.sync_dreamer_utils import get_warp_coordinates, create_target_volume
from ldm.models.diffusion.sync_dreamer_network import NoisyTargetViewEncoder, SpatialTime3DNet, FrustumTV3DNet
from ldm.modules.diffusionmodules.util import make_ddim_timesteps, timestep_embedding
from ldm.modules.encoders.modules import FrozenCLIPImageEmbedder
from ldm.util import instantiate_from_config

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def disable_training_module(module: nn.Module):
    module = module.eval()
    module.train = disabled_train
    for para in module.parameters():
        para.requires_grad = False
    return module

def repeat_to_batch(tensor, B, VN):
    t_shape = tensor.shape
    ones = [1 for _ in range(len(t_shape)-1)]
    tensor_new = tensor.view(B,1,*t_shape[1:]).repeat(1,VN,*ones).view(B*VN,*t_shape[1:])
    return tensor_new

class Zero123Wrapper(nn.Module):
    def __init__(self, diff_model_config, drop_conditions=False, drop_scheme='default', use_zero_123=True):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.drop_conditions = drop_conditions
        self.drop_scheme=drop_scheme
        self.use_zero_123 = use_zero_123


    def drop(self, cond, mask):
        shape = cond.shape
        B = shape[0]
        cond = mask.view(B,*[1 for _ in range(len(shape)-1)]) * cond
        return cond

    def get_trainable_parameters(self):
        return self.diffusion_model.get_trainable_parameters()

    def get_drop_scheme(self, B, device):
        if self.drop_scheme=='default':
            random = torch.rand(B, dtype=torch.float32, device=device)
            drop_clip = (random > 0.15) & (random <= 0.2)
            drop_volume = (random > 0.1) & (random <= 0.15)
            drop_concat = (random > 0.05) & (random <= 0.1)
            drop_all = random <= 0.05
        else:
            raise NotImplementedError
        return drop_clip, drop_volume, drop_concat, drop_all

    def forward(self, x, t, clip_embed, x_concat, is_train=False, **kwargs):
        """

        @param x:             B,4,H,W
        @param t:             B,
        @param clip_embed:    B,M,768
        @param volume_feats:  B,C,D,H,W
        @param x_concat:      B,C,H,W
        @param is_train:
        @return:
        """
        if self.drop_conditions and is_train:
            B = x.shape[0]
            drop_clip, drop_volume, drop_concat, drop_all = self.get_drop_scheme(B, x.device)

            clip_mask = 1.0 - (drop_clip | drop_all).float()
            clip_embed = self.drop(clip_embed, clip_mask)

            concat_mask = 1.0 - (drop_concat | drop_all).float()
            x_concat = self.drop(x_concat, concat_mask)

        if self.use_zero_123:
            # zero123 does not multiply this when encoding, maybe a bug for zero123
            first_stage_scale_factor = 0.18215
            x_concat_ = x_concat * 1.0 / first_stage_scale_factor     #!!!!!
            #x_concat_[:, :4] = x_concat_[:, :4] / first_stage_scale_factor
        else:
            x_concat_ = x_concat

        x = torch.cat([x, x_concat_], 1)
        pred = self.diffusion_model(x, t, clip_embed, **kwargs)
        return pred

    def predict_with_unconditional_scale(self, x, t, clip_embed, x_concat, unconditional_scale, **kwargs):
        x_ = torch.cat([x] * 2, 0)
        t_ = torch.cat([t] * 2, 0)
        clip_embed_ = torch.cat([clip_embed, torch.zeros_like(clip_embed)], 0)
        

        x_concat_ = torch.cat([x_concat, torch.zeros_like(x_concat)], 0)
        
        if self.use_zero_123:
            first_stage_scale_factor = 0.18215
            x_concat_ = x_concat_ * 1.0 / first_stage_scale_factor    
        
        #print(x_.shape)
        #print(x_concat_.shape)
        
        x_ = torch.cat([x_, x_concat_], 1)
        # s, s_uc = self.diffusion_model(x_, t_, clip_embed_, source_dict=v_).chunk(2)
        s, s_uc = self.diffusion_model(x_, t_, clip_embed_, **kwargs).chunk(2)
        s = s_uc + unconditional_scale * (s - s_uc)
        return s

class Zero123(pl.LightningModule):
    def __init__(self, unet_config, scheduler_config,
                 finetune_unet=False, finetune_projection=True, use_zero_123=True,
                 view_num=16, normal_predict=False, image_size=256,
                 cfg_scale=3.0, output_num=8, batch_view_num=4,
                 drop_conditions=False, drop_scheme='default',
                 clip_image_encoder_path="/apdcephfs/private_rondyliu/projects/clip/ViT-L-14.pt",
                 sample_type='ddim', sample_steps=200):
        super().__init__()

        self.finetune_unet = finetune_unet
        self.finetune_projection = finetune_projection

        self.view_num = view_num
        self.normal_predict = normal_predict
        self.viewpoint_dim = 4
        self.output_num = output_num
        self.image_size = image_size

        self.batch_view_num = batch_view_num
        self.cfg_scale = cfg_scale

        self.clip_image_encoder_path = clip_image_encoder_path

        self._init_first_stage()
        self._init_schedule()
        self._init_clip_image_encoder()
        
        if self.normal_predict:
            self.viewpoint_dim = 6
        
        self._init_clip_projection()
        
        self.model = Zero123Wrapper(unet_config, drop_conditions=drop_conditions, drop_scheme=drop_scheme, use_zero_123=use_zero_123)
        self.scheduler_config = scheduler_config
        
        latent_size = image_size//8
        
        self.sampler = Zero123DDIMSampler(self, sample_steps , "uniform", 1.0, latent_size=latent_size)
        
    def _init_clip_projection(self):
        self.cc_projection = nn.Linear(768+self.viewpoint_dim, 768)
        nn.init.eye_(list(self.cc_projection.parameters())[0][:768, :768])
        nn.init.zeros_(list(self.cc_projection.parameters())[1])
        self.cc_projection.requires_grad_(True)

        if not self.finetune_projection:
            disable_training_module(self.cc_projection)
    
    def get_viewpoint_embedding(self, dp):
        """
        @param dp {da,de,dz}
        @return: cam_pose_embedding
        """
        if self.normal_predict:
            embedding = torch.stack([dp['de'], torch.sin(dp['da']), torch.cos(dp['da']), dp['dz']], -1) # B,N,4
            embed = torch.tensor([1,0], dtype=embedding.dtype, device=embedding.device)
            embed = embed[None,None].repeat(embedding.shape[0], 2*embedding.shape[1], 1)
            embed[:,embedding.shape[1]:,0] = 0
            embed[:,embedding.shape[1]:,1] = 1
            embedding = torch.cat([embedding.repeat(1,2,1), embed], -1) # B,2*N,6
        else:
            embedding = torch.stack([dp['de'], torch.sin(dp['da']), torch.cos(dp['da']), dp['dz']], -1) # B,N,4
        return embedding
    
    
    def _init_first_stage(self):
        first_stage_config={
            "target": "ldm.models.autoencoder.AutoencoderKL",
            "params": {
                "embed_dim": 4,
                "monitor": "val/rec_loss",
                "ddconfig":{
                  "double_z": True,
                  "z_channels": 4,
                  "resolution": self.image_size,
                  "in_channels": 3,
                  "out_ch": 3,
                  "ch": 128,
                  "ch_mult": [1,2,4,4],
                  "num_res_blocks": 2,
                  "attn_resolutions": [],
                  "dropout": 0.0
                },
                "lossconfig": {"target": "torch.nn.Identity"},
            }
        }
        self.first_stage_scale_factor = 0.18215
        self.first_stage_model = instantiate_from_config(first_stage_config)
        self.first_stage_model = disable_training_module(self.first_stage_model)

    def _init_clip_image_encoder(self):
        self.clip_image_encoder = FrozenCLIPImageEmbedder(model=self.clip_image_encoder_path)
        self.clip_image_encoder = disable_training_module(self.clip_image_encoder)

    def _init_schedule(self):
        self.num_timesteps = 1000
        linear_start = 0.00085
        linear_end = 0.0120
        num_timesteps = 1000
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2 # T
        assert betas.shape[0] == self.num_timesteps

        # all in float64 first
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # T
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # T
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_log_variance_clipped = torch.clamp(posterior_log_variance_clipped, min=-10)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).float())
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped.float())
        
    def encode_first_stage(self, x, sample=True):
        with torch.no_grad():
            posterior = self.first_stage_model.encode(x)  # b,4,h//8,w//8
            if sample:
                return posterior.sample().detach() * self.first_stage_scale_factor
            else:
                return posterior.mode().detach() * self.first_stage_scale_factor

    def decode_first_stage(self, z):
        with torch.no_grad():
            z = 1. / self.first_stage_scale_factor * z
            return self.first_stage_model.decode(z)

    def prepare(self, batch):
        # encode target
        if 'target_image' in batch:
            image_target = batch['target_image'].permute(0, 1, 4, 2, 3) # b,n,3,h,w
            N = image_target.shape[1]
            x = [self.encode_first_stage(image_target[:,ni], True) for ni in range(N)]
            x = torch.stack(x, 1) # b,n,4,h//8,w//8
        else:
            x = None
        
        image_input = batch['input_image'].permute(0, 3, 1, 2)
        x_input = torch.cat([self.encode_first_stage(image_input[:,i*3:(i+1)*3,:,:]) for i in range(image_input.shape[1]//3)],1)
        # x_input = torch.cat([self.encode_first_stage(image_input[:,i*3:(i+1)*3,:,:]) for i in [image_input.shape[1]//3-1]],1)

        azimuth_input = batch['input_az']
        azimuth_target = batch['target_az']
        elevation_input = -batch['input_el']
        elevation_target = -batch['target_el']
        distance_input = batch['input_dis']
        distance_target = batch['target_dis']
        N = azimuth_target.shape[1]
        
        # print(elevation_target.shape)
        # print(elevation_input.shape)
        d_e = elevation_target - elevation_input.repeat(1, N)
        d_a = azimuth_target - azimuth_input.repeat(1, N)
        d_z = distance_target - distance_input.repeat(1, N)
        d_p = {'da': d_a, 'de': d_e, 'dz': d_z}
        
        input_info = {'image': image_input, 'dp': d_p, 'x': x_input}
        with torch.no_grad():
            clip_embed = torch.cat([self.clip_image_encoder.encode(image_input[:,i*3:(i+1)*3,:,:]) for i in range(image_input.shape[1]//3)],1)
        return x, clip_embed, input_info
        
    def training_step(self, batch):
        '''
        print(batch['target_image'].shape)
        print(batch['input_image'].shape)
        print()
        print(batch['target_az'].shape)
        print(batch['target_el'].shape)
        print(batch['target_dis'].shape)
        print(batch['target_pose'].shape)
        print(batch['target_K'].shape)
        print()
        print(batch['input_az'].shape)
        print(batch['input_el'].shape)
        print(batch['input_dis'].shape)
        print(batch['input_pose'].shape)
        print(batch['input_K'].shape)
        '''
        B = batch['target_image'].shape[0]
        
        self.poses = batch['target_pose'].to(torch.float32)
        self.Ks = batch['target_K'][:,None].repeat(1,self.view_num,1,1).to(torch.float32)
        
        time_steps = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()

        x, clip_embed, input_info = self.prepare(batch)
        x_noisy, noise = self.add_noise(x, time_steps)  # B,N,4,H,W

        N = self.view_num
        target_index = torch.randint(0, N, (B, 1), device=self.device).long() # B, 1
        v_embed = self.get_viewpoint_embedding(input_info['dp']) # B,N,v_dim
        
        TN = target_index.shape[1]
        clip_embed_ = clip_embed.unsqueeze(1).repeat(1,TN,1,1).view(B*TN,clip_embed.shape[1],768)
        v_embed_ = v_embed[torch.arange(B)[:,None], target_index].view(B*TN, self.viewpoint_dim) # B*TN,v_dim
        clip_embed_ = self.cc_projection(torch.cat([clip_embed_, v_embed_.unsqueeze(1).repeat(1,clip_embed.shape[1],1)], -1))  # B*TN,1,768
        
        B, _, H, W = input_info['x'].shape
        x_input_ = input_info['x'].unsqueeze(1).repeat(1, TN, 1, 1, 1).view(B * TN, input_info['x'].shape[1], H, W)
        x_concat = x_input_
        
        x_noisy_ = x_noisy[torch.arange(B)[:,None],target_index][:,0] # B,4,H,W
        noise_predict = self.model(x_noisy_, time_steps, clip_embed_, x_concat, is_train=True) # B,4,H,W
        noise_target = noise[torch.arange(B)[:,None],target_index][:,0] # B,4,H,W
        # loss simple for diffusion
        loss = torch.nn.functional.mse_loss(noise_target, noise_predict, reduction='none').mean()
        
        if self.normal_predict:
            clip_embed_ = clip_embed.unsqueeze(1).repeat(1,TN,1,1).view(B*TN,clip_embed.shape[1],768)
            v_embed_ = v_embed[torch.arange(B)[:,None], target_index+N].view(B*TN, self.viewpoint_dim) # B*TN,v_dim
            clip_embed_ = self.cc_projection(torch.cat([clip_embed_, v_embed_.unsqueeze(1).repeat(1,clip_embed.shape[1],1)], -1))  # B*TN,1,768
            
            B, _, H, W = input_info['x'].shape
            x_input_ = input_info['x'].unsqueeze(1).repeat(1, TN, 1, 1, 1).view(B * TN, input_info['x'].shape[1], H, W)
            x_concat = x_input_
            
            x_noisy_ = x_noisy[torch.arange(B)[:,None],target_index+N][:,0] # B,4,H,W
            noise_predict = self.model(x_noisy_, time_steps, clip_embed_, x_concat, is_train=True) # B,4,H,W
            noise_target = noise[torch.arange(B)[:,None],target_index+N][:,0] # B,4,H,W
            loss += torch.nn.functional.mse_loss(noise_target, noise_predict, reduction='none').mean()
        
        self.log('sim', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, rank_zero_only=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        self.log("step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        
        return loss
        
    def add_noise(self, x_start, t):
        """
        @param x_start: B,*
        @param t:       B,
        @return:
        """
        B = x_start.shape[0]
        noise = torch.randn_like(x_start) # B,*

        sqrt_alphas_cumprod_  = self.sqrt_alphas_cumprod[t] # B,
        sqrt_one_minus_alphas_cumprod_ = self.sqrt_one_minus_alphas_cumprod[t] # B
        sqrt_alphas_cumprod_ = sqrt_alphas_cumprod_.view(B, *[1 for _ in range(len(x_start.shape)-1)])
        sqrt_one_minus_alphas_cumprod_ = sqrt_one_minus_alphas_cumprod_.view(B, *[1 for _ in range(len(x_start.shape)-1)])
        x_noisy = sqrt_alphas_cumprod_ * x_start + sqrt_one_minus_alphas_cumprod_ * noise
        return x_noisy, noise

    def sample(self, sampler, batch, cfg_scale, batch_view_num, return_inter_results=False, inter_interval=50, inter_view_interval=2):
        _, clip_embed, input_info = self.prepare(batch)
        self.poses = batch['target_pose'].to(torch.float32)
        self.Ks = batch['target_K'][:,None].repeat(1,self.view_num,1,1).to(torch.float32)
        
        x_sample, inter = sampler.sample(input_info, clip_embed, unconditional_scale=cfg_scale, log_every_t=inter_interval, batch_view_num=batch_view_num)

        N = x_sample.shape[1]
        x_sample = torch.stack([self.decode_first_stage(x_sample[:, ni]) for ni in range(N)], 1)
        if return_inter_results:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            inter = torch.stack(inter['x_inter'], 2) # # B,N,T,C,H,W
            B,N,T,C,H,W = inter.shape
            inter_results = []
            for ni in tqdm(range(0, N, inter_view_interval)):
                inter_results_ = []
                for ti in range(T):
                    inter_results_.append(self.decode_first_stage(inter[:, ni, ti]))
                inter_results.append(torch.stack(inter_results_, 1)) # B,T,3,H,W
            inter_results = torch.stack(inter_results,1) # B,N,T,3,H,W
            return x_sample, inter_results
        else:
            return x_sample

    def log_image(self,  x_sample, batch, step, output_dir):
        process = lambda x: ((torch.clip(x, min=-1, max=1).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        B = x_sample.shape[0]
        N = x_sample.shape[1]
        image_cond = []
        output_dir = Path(output_dir)

        for bi in range(B):
            img_pr_ = concat_images_list(*[process(batch['input_image'][bi,:,:,i*3:(i+1)*3]) for i in range(batch['input_image'].shape[-1]//3)],*[process(x_sample[bi, ni].permute(1, 2, 0)) for ni in range(N)])
            image_cond.append(img_pr_)

        imsave(str(output_dir/f'{step}.jpg'), concat_images_list(*image_cond, vert=True))

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if batch_idx==0 and self.global_rank==0:
            self.eval()
            step = self.global_step
            
            batch_ = {}
            for k, v in batch.items(): batch_[k] = v[:self.output_num]
            self.poses = batch_['target_pose'].to(torch.float32)
            self.Ks = batch_['target_K'][:,None].repeat(1,self.view_num,1,1).to(torch.float32)
            x_sample = self.sample(self.sampler, batch_, self.cfg_scale, self.batch_view_num)
            output_dir = Path(self.image_dir) / 'images' / 'val'
            output_dir.mkdir(exist_ok=True, parents=True)
            self.log_image(x_sample, batch, step, output_dir=output_dir)
     
    def save_image(self,  x_sample, batch, output_dir):
        process = lambda x: ((torch.clip(x, min=-1, max=1).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        B = x_sample.shape[0]
        N = x_sample.shape[1]
        image_cond = []
        output_dir = Path(output_dir)
        
        '''
        uids = np.concatenate([np.load('/public_bme/data/v-xuchf/test/splits.npy',allow_pickle=True).item()['sync'],np.load('/public_bme/data/v-xuchf/test/splits.npy',allow_pickle=True).item()['real']])
        names = []
        for i in uids:
            names.append(i+'_norm_lower')
            names.append(i+'_norm_upper')
        '''
        
        import os
        names = os.listdir('/public_bme/data/v-xuchf/test/real/')
        
        for bi in range(B):
            img_pr_ = concat_images_list(*[process(x_sample[bi, ni].permute(1, 2, 0)) for ni in range(N)])
            name = names[batch['name'][bi][0]]
            conds = '_'.join([f'{cond:03d}' for cond in batch['cond'][bi]])
            output_fn = output_dir / f'{name}_cond_{conds}.png'
            imsave(output_fn, img_pr_)
           
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.eval()
        self.poses = batch['target_pose'].to(torch.float32)
        self.Ks = batch['target_K'][:,None].repeat(1,self.view_num,1,1).to(torch.float32)
        x_sample = self.sample(self.sampler, batch, self.cfg_scale, self.batch_view_num)
        output_dir = Path(self.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        self.save_image(x_sample, batch, output_dir=output_dir)
        
    def configure_optimizers(self):
        lr = self.learning_rate
        print(f'setting learning rate to {lr:.4f} ...')
        paras = []
        if self.finetune_projection:
            paras.append({"params": self.cc_projection.parameters(), "lr": 10*lr},)
        if self.finetune_unet:
            paras.append({"params": self.model.parameters(), "lr": lr},)
        else:
            paras.append({"params": self.model.get_trainable_parameters(), "lr": lr},)
        
        opt = torch.optim.AdamW(paras, lr=lr)

        scheduler = instantiate_from_config(self.scheduler_config)
        print("Setting up LambdaLR scheduler...")
        scheduler = [{'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1}]
        return [opt], scheduler
        
class Zero123DDIMSampler:
    def __init__(self, model: Zero123, ddim_num_steps, ddim_discretize="uniform", ddim_eta=1.0, latent_size=32):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.latent_size = latent_size
        self._make_schedule(ddim_num_steps, ddim_discretize, ddim_eta)
        self.eta = ddim_eta

    def _make_schedule(self,  ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps, num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose) # DT
        ddim_timesteps_ = torch.from_numpy(self.ddim_timesteps.astype(np.int64)) # DT

        alphas_cumprod = self.model.alphas_cumprod # T
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        self.ddim_alphas = alphas_cumprod[ddim_timesteps_].double() # DT
        self.ddim_alphas_prev = torch.cat([alphas_cumprod[0:1], alphas_cumprod[ddim_timesteps_[:-1]]], 0) # DT
        self.ddim_sigmas = ddim_eta * torch.sqrt((1 - self.ddim_alphas_prev) / (1 - self.ddim_alphas) * (1 - self.ddim_alphas / self.ddim_alphas_prev))

        self.ddim_alphas_raw = self.model.alphas[ddim_timesteps_].float() # DT
        self.ddim_sigmas = self.ddim_sigmas.float()
        self.ddim_alphas = self.ddim_alphas.float()
        self.ddim_alphas_prev = self.ddim_alphas_prev.float()
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1. - self.ddim_alphas).float()


    @torch.no_grad()
    def denoise_apply_impl(self, x_target_noisy, index, noise_pred, is_step0=False):
        """
        @param x_target_noisy: B,N,4,H,W
        @param index:          index
        @param noise_pred:     B,N,4,H,W
        @param is_step0:       bool
        @return:
        """
        device = x_target_noisy.device
        B,N,_,H,W = x_target_noisy.shape

        # apply noise
        a_t = self.ddim_alphas[index].to(device).float().view(1,1,1,1,1)
        a_prev = self.ddim_alphas_prev[index].to(device).float().view(1,1,1,1,1)
        sqrt_one_minus_at = self.ddim_sqrt_one_minus_alphas[index].to(device).float().view(1,1,1,1,1)
        sigma_t = self.ddim_sigmas[index].to(device).float().view(1,1,1,1,1)

        pred_x0 = (x_target_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt()
        dir_xt = torch.clamp(1. - a_prev - sigma_t**2, min=1e-7).sqrt() * noise_pred
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt
        if not is_step0:
            noise = sigma_t * torch.randn_like(x_target_noisy)
            x_prev = x_prev + noise
        return x_prev

    @torch.no_grad()
    def denoise_apply(self, x_target_noisy, input_info, clip_embed, time_steps, index, unconditional_scale, batch_view_num=1, is_step0=False):
        """
        @param x_target_noisy:   B,N,4,H,W
        @param input_info:
        @param clip_embed:       B,M,768
        @param time_steps:       B,
        @param index:            int
        @param unconditional_scale:
        @param batch_view_num:   int
        @param is_step0:         bool
        @return:
        """
        x_input, dp = input_info['x'], input_info['dp']
        B, _, C, H, W = x_target_noisy.shape
        N = self.model.view_num

        # construct source data
        v_embed = self.model.get_viewpoint_embedding(dp) # B,N,v_dim
        
        e_t = []
        target_indices = torch.arange(N) # N
        for ni in range(0, N, batch_view_num):
            x_target_noisy_ = x_target_noisy[:, ni:ni + batch_view_num]
            VN = x_target_noisy_.shape[1]
            x_target_noisy_ = x_target_noisy_.reshape(B*VN,C,H,W)

            time_steps_ = repeat_to_batch(time_steps, B, VN)
            target_indices_ = target_indices[ni:ni+batch_view_num].unsqueeze(0).repeat(B,1)
            
            TN = target_indices_.shape[1]
            clip_embed_ = clip_embed.unsqueeze(1).repeat(1,TN,1,1).view(B*TN,clip_embed.shape[1],768)
            v_embed_ = v_embed[torch.arange(B)[:,None], target_indices_].view(B*TN, self.model.viewpoint_dim) # B*TN,v_dim
            clip_embed_ = self.model.cc_projection(torch.cat([clip_embed_, v_embed_.unsqueeze(1).repeat(1,clip_embed.shape[1],1)], -1))  # B*TN,1,768
    
            x_input_ = x_input.unsqueeze(1).repeat(1, TN, 1, 1, 1).view(B * TN, x_input.shape[1], H, W)
            x_concat_ = x_input_
            
            if unconditional_scale!=1.0:
                noise = self.model.model.predict_with_unconditional_scale(x_target_noisy_, time_steps_, clip_embed_, x_concat_, unconditional_scale)
            else:
                noise = self.model.model(x_target_noisy_, time_steps_, clip_embed_, x_concat_, is_train=False)
            e_t.append(noise.view(B,VN,4,H,W))
            
        if self.model.normal_predict:
            for ni in range(0, N, batch_view_num):
                x_target_noisy_ = x_target_noisy[:, ni + N:ni + N + batch_view_num]
                VN = x_target_noisy_.shape[1]
                x_target_noisy_ = x_target_noisy_.reshape(B*VN,C,H,W)
    
                time_steps_ = repeat_to_batch(time_steps, B, VN)
                target_indices_ = target_indices[ni:ni+batch_view_num].unsqueeze(0).repeat(B,1)
                
                TN = target_indices_.shape[1]
                clip_embed_ = clip_embed.unsqueeze(1).repeat(1,TN,1,1).view(B*TN,clip_embed.shape[1],768)
                v_embed_ = v_embed[torch.arange(B)[:,None], target_indices_+N].view(B*TN, self.model.viewpoint_dim) # B*TN,v_dim
                clip_embed_ = self.model.cc_projection(torch.cat([clip_embed_, v_embed_.unsqueeze(1).repeat(1,clip_embed.shape[1],1)], -1))  # B*TN,1,768
        
                x_input_ = x_input.unsqueeze(1).repeat(1, TN, 1, 1, 1).view(B * TN, x_input.shape[1], H, W)
                x_concat_ = x_input_
                
                if unconditional_scale!=1.0:
                    noise = self.model.model.predict_with_unconditional_scale(x_target_noisy_, time_steps_, clip_embed_, x_concat_, unconditional_scale)
                else:
                    noise = self.model.model(x_target_noisy_, time_steps_, clip_embed_, x_concat_, is_train=False)
                e_t.append(noise.view(B,VN,4,H,W))
        
        e_t = torch.cat(e_t, 1)
        x_prev = self.denoise_apply_impl(x_target_noisy, index, e_t, is_step0)
        return x_prev

    @torch.no_grad()
    def sample(self, input_info, clip_embed, unconditional_scale=1.0, log_every_t=50, batch_view_num=1):
        """
        @param input_info:      x, elevation
        @param clip_embed:      B,M,768
        @param unconditional_scale:
        @param log_every_t:
        @param batch_view_num:
        @return:
        """
        print(f"unconditional scale {unconditional_scale:.1f}")
        C, H, W = 4, self.latent_size, self.latent_size
        B = clip_embed.shape[0]
        N = self.model.view_num
        device = self.model.device
        if self.model.normal_predict:
            x_target_noisy = torch.randn([B, 2*N, C, H, W], device=device)
        else:
            x_target_noisy = torch.randn([B, N, C, H, W], device=device)

        timesteps = self.ddim_timesteps
        intermediates = {'x_inter': []}
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1 # index in ddim state
            time_steps = torch.full((B,), step, device=device, dtype=torch.long)
            x_target_noisy = self.denoise_apply(x_target_noisy, input_info, clip_embed, time_steps, index, unconditional_scale, batch_view_num=batch_view_num, is_step0=index==0)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(x_target_noisy)

        return x_target_noisy, intermediates