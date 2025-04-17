import os

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from third_parties.lpips import LPIPS

from core.train import create_lr_updater
from core.data import create_dataloader
from core.utils.network_util import set_requires_grad, get_gaussian_kernel
from core.utils.train_util import cpu_data_to_gpu, Timer
from core.utils.image_util import tile_images, to_8b_image
import random
from configs import cfg
from core.train.trainers.human_nerf.event_loss_helpers import get_event_rgb
from core.utils.train_util import event_loss_call, event_loss_call_uncert

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
img2mse_uncert_alpha = lambda x, y, uncert: torch.mean((1 / (2*(uncert+1e-9).unsqueeze(-1))) *((x - y) ** 2))
img2mse_weight = lambda x, y, weight: torch.mean(weight.unsqueeze(-1) *((x - y) ** 2))

to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height', 'next_target_rgbs']

SCALE = 2.2 
THR = 0.5 
EPS = 1e-5 
RATIO = SCALE/THR
UNCERT = 1

def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    # patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    patch_imgs = targets.clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

    return patch_imgs


def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Trainer(object):
    def __init__(self, network, optimizer):
        print('\n********** Init Trainer ***********')
        setup_seed(20)
        network = network.cuda().deploy_mlps_to_secondary_gpus()
        self.network = network

        self.optimizer = optimizer
        self.update_lr = create_lr_updater()

        if cfg.resume and Trainer.ckpt_exists(cfg.load_net):
            self.load_ckpt(f'{cfg.load_net}')
        else:
            self.iter = 0
            self.save_ckpt('init')
            self.iter = 1

        self.timer = Timer()

        self.lpips = LPIPS(net='vgg')
        set_requires_grad(self.lpips, requires_grad=False)
        self.lpips = nn.DataParallel(self.lpips).cuda()

        print("Load Progress Dataset ...")
        self.prog_dataloader = create_dataloader(data_type='progress')

        self.gaussian_blur = get_gaussian_kernel(9, 5).cuda()
        print('************************************')
        print(f' -- UNCERT: {UNCERT}')

    @staticmethod
    def get_ckpt_path(name):
        return os.path.join(cfg.logdir, f'{name}.tar')

    @staticmethod
    def ckpt_exists(name):
        return os.path.exists(Trainer.get_ckpt_path(name))

    ######################################################3
    ## Training 

    def get_img_rebuild_loss(self, loss_names, rgb, uncert_unpacked, sub_rgb, sub_uncert_unpacked, alpha_unpacked, all_sub_alpha_unpacked, target, event_target, alpha_target):
        losses = {}

        all_rgb = torch.stack([rgb, *sub_rgb[:-1]], dim=0)
        blur_rgb = torch.mean(all_rgb, dim = 0)
        event_rgb = sub_rgb[-1]

        all_uncert = torch.stack([uncert_unpacked, *sub_uncert_unpacked[:-1]], dim=0)
        blur_num = all_rgb.shape[0]

        blur_uncert = torch.max(all_uncert, dim = 0).values
        uncert_unpacked_copy = uncert_unpacked.clone()
        uncert_unpacked_copy[uncert_unpacked_copy > UNCERT] = 1e9
        blur_uncert_copy = blur_uncert.clone()
        blur_uncert_copy[blur_uncert_copy > UNCERT] = 1e9


        all_alpha = torch.stack([alpha_unpacked, *all_sub_alpha_unpacked], dim=0)
        blur_alpha = torch.max(all_alpha, dim = 0).values

        
        if "blur_mse_uncert" in loss_names:
            losses["blur_mse_uncert"] = img2mse_uncert_alpha(blur_rgb, target, blur_uncert_copy)
        
        if "blur_lpips_uncert" in loss_names:
            lpips_loss = self.lpips(scale_for_lpips(blur_rgb.permute(0, 3, 1, 2)),
                                    scale_for_lpips(target.permute(0, 3, 1, 2)),
                                    blur_uncert_copy.unsqueeze(1))
            losses["blur_lpips_uncert"] = torch.mean(lpips_loss)
        
        if "event_mse_uncert" in loss_names:
            rgb_uncert = uncert_unpacked + 1e-9
            event_uncert = sub_uncert_unpacked[-1] + 1e-9
            losses["event_mse_uncert"] = event_loss_call_uncert([rgb.reshape(-1, 3), event_rgb.reshape(-1, 3)], event_target.reshape(-1, 1), "rgb", [rgb_uncert.reshape(-1, 1), event_uncert.reshape(-1, 1)])

        if "alpha_mse_uncert" in loss_names:
            losses["alpha_mse_uncert"] = img2mse_uncert_alpha(blur_alpha.unsqueeze(-1), alpha_target.unsqueeze(-1), blur_uncert_copy) 

        return losses

    def get_loss(self, net_output, all_sub_net_output,
                 patch_masks=None, bgcolor=None, targets=None, event_target=None, alpha_target=None, div_indices=None, target_rgbs=None, target_events=None, dst_Rs=None, dst_Ts=None, use_bg=False):

        lossweights = cfg.train.lossweights.copy()
        if not use_bg:
            lossweights.pop('event_mse', 'without event_mse')
            lossweights.pop('event_mse_uncert', 'without event_mse_uncert')
        loss_names = list(lossweights.keys())

        rgb = net_output['rgb']
        all_sub_rgb = [sub_net_output['rgb'] for sub_net_output in all_sub_net_output]

        alpha = net_output['acc']
        all_sub_alpha = [sub_net_output['acc'] for sub_net_output in all_sub_net_output]

        velocity = net_output['velocity'].detach()
        uncert = torch.exp(velocity) - 1 + 0.01

        all_sub_velocity = [sub_net_output['velocity'].detach() for sub_net_output in all_sub_net_output]
        all_sub_uncert = [torch.exp(sub_velocity) - 1 + 0.01 for sub_velocity in all_sub_velocity]

        if np.random.rand() < 0.01 or uncert.min() < 0:
            print(f"uncert min max mean: {uncert.min().item()} {uncert.max().item()} {uncert.mean().item()}")
        
        if cfg.train.ray_shoot_mode == 'patch':

            rgb_unpacked = _unpack_imgs(rgb, patch_masks, bgcolor,
                                        targets, div_indices)
            uncert_unpacked = _unpack_imgs(uncert, patch_masks, bgcolor,
                                            torch.ones_like(targets[...,0]), div_indices) 
            all_sub_rgb_unpacked = [_unpack_imgs(sub_rgb, patch_masks, bgcolor,
                                                    targets, div_indices) for sub_rgb in all_sub_rgb]
            all_sub_uncert_unpacked = [_unpack_imgs(sub_uncert, patch_masks, bgcolor,
                                                        torch.ones_like(targets[...,0]), div_indices) for sub_uncert in all_sub_uncert]
            alpha_unpacked = _unpack_imgs(alpha, patch_masks, bgcolor,
                                        alpha_target, div_indices)
            all_sub_alpha_unpacked = [_unpack_imgs(sub_alpha, patch_masks, bgcolor,
                                                    alpha_target, div_indices) for sub_alpha in all_sub_alpha]
            losses = self.get_img_rebuild_loss(
                            loss_names, 
                            rgb_unpacked, 
                            uncert_unpacked,
                            all_sub_rgb_unpacked,
                            all_sub_uncert_unpacked,
                            alpha_unpacked,
                            all_sub_alpha_unpacked,
                            targets,
                            event_target,
                            alpha_target)
        
        else:
            losses = self.get_img_rebuild_loss(
                            loss_names, 
                            rgb, 
                            uncert,
                            all_sub_rgb,
                            all_sub_uncert,
                            target_rgbs,
                            target_events)
        
        if "pose_reg" in loss_names:
            out_dst_Rs = net_output['dst_Rs']
            out_dst_Ts = net_output['dst_Ts']
            losses["pose_reg"] = img2mse(out_dst_Rs, dst_Rs) + img2mse(out_dst_Ts, dst_Ts)
        
        if "sparsity_reg" in loss_names:
            acc = net_output['acc']
            all_sub_acc = [sub_net_output['acc'] for sub_net_output in all_sub_net_output]
            all_acc = torch.stack([acc, *all_sub_acc], dim=0)
            HARD_SURFACE_OFFSET = 0.31326165795326233
            losses["sparsity_reg"] = torch.mean(-torch.log(torch.exp(-torch.abs(all_acc)) + torch.exp(-torch.abs(1-all_acc))
            ) + HARD_SURFACE_OFFSET)


        train_losses = [
            weight * losses[k] for k, weight in lossweights.items()
        ]

        return sum(train_losses), \
               {loss_names[i]: train_losses[i] for i in range(len(loss_names))}

    def train_begin(self, train_dataloader):
        assert train_dataloader.batch_size == 1

        self.network.train()
        cfg.perturb = cfg.train.perturb

    def train_end(self):
        pass

    def train(self, epoch, train_dataloader):
        self.train_begin(train_dataloader=train_dataloader)
        reload_times = 0
        self.timer.begin()
        for batch_idx, (batch, sub_batch_list) in enumerate(train_dataloader):
            if self.iter > cfg.train.maxiter:
                break

            self.optimizer.zero_grad()

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            net_output = self.network(**data)

            all_sub_net_output = []
            for sub_batch in sub_batch_list:
                for k, v in sub_batch.items():
                    sub_batch[k] = v[0]

                sub_batch['iter_val'] = torch.full((1,), self.iter)
                sub_data = cpu_data_to_gpu(
                    sub_batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
                sub_net_output = self.network(**sub_data)
                all_sub_net_output.append(sub_net_output)
            
            train_loss, loss_dict = self.get_loss(
                net_output=net_output,
                all_sub_net_output=all_sub_net_output,
                patch_masks=data['patch_masks'] if 'patch_masks' in data else None,
                bgcolor=data['bgcolor'] / 255. if 'bgcolor' in data else None,
                targets=data['target_patches'] if 'target_patches' in data else None,
                event_target=data['target_patches_event'] if 'target_patches_event' in data else None,
                alpha_target=data['target_patches_alpha'] if 'target_patches_alpha' in data else None,
                div_indices=data['patch_div_indices'] if 'patch_div_indices' in data else None,
                target_rgbs=data['target_rgbs'] if 'target_rgbs' in data else None,
                target_events=data['target_events'] if 'target_events' in data else None,
                dst_Rs=data['dst_Rs'] if 'dst_Rs' in data else None,
                dst_Ts=data['dst_Ts'] if 'dst_Ts' in data else None,
                use_bg="uv" in data)

            train_loss.backward()
            self.optimizer.step()

            if self.iter % cfg.train.log_interval == 0:
                loss_str = f"Loss: {train_loss.item():.4f} ["
                for k, v in loss_dict.items():
                    loss_str += f"{k}: {v.item():.4f} "
                loss_str += "]"

                log_str = 'Epoch: {} [Iter {}, {}/{} ({:.0f}%), {}] {}'
                log_str = log_str.format(
                    epoch, self.iter,
                    batch_idx * cfg.train.batch_size, len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), 
                    self.timer.log(),
                    loss_str)
                print(log_str)

            is_reload_model = False
            if self.iter in [100, 300, 1000, 2500] or \
                self.iter % cfg.progress.dump_interval == 0:
                is_reload_model = self.progress()

            if torch.isnan(train_loss):
                print("Produce non loss; reload the latest model.")            
                self.load_ckpt('latest')
                reload_times += 1
                is_reload_model = True
                print(f"Reloaded for {reload_times} times!")

            if not is_reload_model:
                if self.iter % cfg.train.save_checkpt_interval == 0:
                    self.save_ckpt('latest')
                    reload_times = 0

                if cfg.save_all:
                    if self.iter % cfg.train.save_model_interval == 0:
                        self.save_ckpt(f'iter_{self.iter}')

                self.update_lr(self.optimizer, self.iter)

                self.iter += 1
    
    def finalize(self):
        self.save_ckpt('latest')

    ######################################################3
    ## Progress

    def progress_begin(self):
        self.network.eval()
        cfg.perturb = 0.

    def progress_end(self):
        self.network.train()
        cfg.perturb = cfg.train.perturb

    def progress(self):
        self.progress_begin()

        print('Evaluate Progress Images ...')

        images = []
        is_empty_img = False
        for _, (batch, sub_batch_list) in enumerate(tqdm(self.prog_dataloader)):

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            rendered = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
            velocity_map = np.full(
                        (height * width, 1), 0., 
                        dtype='float32')
            velocity_map_mask = np.full(
                        (height * width, 1), 1., 
                        dtype='float32')


            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                net_output = self.network(**data)


            rgb = net_output['rgb'].data.to("cpu").numpy()
            velocity = net_output['velocity']
            velocity = torch.exp(velocity) - 1 + 0.01 # zju_mocap
            target_rgbs = batch['target_rgbs']
            if len(cfg.train.lossweights.keys()) == 1 and 'event_mse' in cfg.train.lossweights.keys():
                rendered[ray_mask,:] = rgb.mean(axis=-1, keepdims=True)
            else:
                rendered[ray_mask,:] = rgb
            truth[ray_mask] = target_rgbs
            velocity_map[ray_mask, 0] = (velocity / torch.max(velocity)).data.cpu().numpy()
            velocity[velocity > UNCERT] = 1e9
            velocity_map_mask[ray_mask, 0] = (velocity / torch.max(velocity)).data.cpu().numpy()            
            truth_event = np.full(
                            (height * width, 3), np.array([0, 0, 0])/255., 
                            dtype='float32')
            truth_event[ray_mask] = batch['target_events'].numpy()
            truth_event_rgb = get_event_rgb(truth_event.reshape((height, width, -1)))

            truth = to_8b_image(truth.reshape((height, width, -1)))
            rendered = to_8b_image(rendered.reshape((height, width, -1)))
            velocity_map = to_8b_image(velocity_map.reshape((height, width, -1)))
            velocity_map_mask = to_8b_image(velocity_map_mask.reshape((height, width, -1)))
            velocity_map = velocity_map.repeat(3, axis=-1)
            velocity_map_mask = velocity_map_mask.repeat(3, axis=-1)
            images.append(np.concatenate([rendered, truth, velocity_map, velocity_map_mask, truth_event_rgb], axis=1))

             # check if we create empty images (only at the begining of training)
            if self.iter <= 5000 and \
                np.var(rendered.reshape(-1,3)[ray_mask], axis=0).mean() < 0.01: # np.allclose(rendered, np.array(cfg.bgcolor), atol=5.):
                is_empty_img = True
                break

        tiled_image = tile_images(images)
        
        Image.fromarray(tiled_image).save(
            os.path.join(cfg.logdir, "prog_{:06}.jpg".format(self.iter)))

        if is_empty_img:
            print("Produce empty images; reload the init model.")
            self.load_ckpt('init')
            
        self.progress_end()

        return is_empty_img


    ######################################################3
    ## Utils

    def save_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Save checkpoint to {path} ...")

        torch.save({
            'iter': self.iter,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Load checkpoint from {path} ...")
        
        ckpt = torch.load(path, map_location='cuda:0')
        self.iter = ckpt['iter'] + 1

        self.network.load_state_dict(ckpt['network'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer'])
