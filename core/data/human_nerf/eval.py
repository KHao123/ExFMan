import os
import pickle

import numpy as np
import cv2
import torch
import torch.utils.data

from core.utils.image_util import load_image
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from core.utils.file_util import list_files, split_path
from core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox

from configs import cfg

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            bgcolor=None,
            gama=0,
            ray_shoot_mode='image',
            skip=1,
            data_type='train',
            **_):

        self.data_type = data_type
        print('[Data Type]', self.data_type)
        print('[Dataset Path]', dataset_path) 

        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, 'images')
        self.avg_beta = None
        self.canonical_joints, self.canonical_bbox = \
            self.load_canonical_joints()

        if 'motion_weights_priors' in keyfilter:
            self.motion_weights_priors = \
                approx_gaussian_bone_volumes(
                    self.canonical_joints,   
                    self.canonical_bbox['min_xyz'],
                    self.canonical_bbox['max_xyz'],
                    grid_size=cfg.mweight_volume.volume_size).astype('float32')


        self.cameras = self.load_train_cameras()
        self.mesh_infos = self.load_train_mesh_infos()

        # zju-mocap
        framelist = self.load_train_frames()
        # real-world
        # framelist = list(self.mesh_infos.keys())
        self.framelist = framelist[::skip]
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]
        print(f' -- Total Frames: {self.get_total_frames()}')

        self.keyfilter = keyfilter
        print(f' -- Key Filter: {self.keyfilter}')
        self.bgcolor = bgcolor
        print(f' -- Background Color: {self.bgcolor}')

        self.gama = gama
        print(f' -- Use Background: {self.gama}')

        self.ray_shoot_mode = ray_shoot_mode

        # events
        event_dir = os.path.join(dataset_path, "event_info.pkl")
        with open(event_dir, 'rb') as fs:
            self.event_info = pickle.load(fs)
        self.tE = torch.tensor(self.event_info['tE'])
        self.xE = torch.tensor(self.event_info['xE'])
        self.yE = torch.tensor(self.event_info['yE'])
        self.pE = torch.tensor(self.event_info['pE'])
        # self.tF = torch.tensor(self.event_info['tF'])
        # assert len(self.tF) == len(framelist)

        # real-world
        # self.tF = torch.tensor(self.event_info['tF'])
        # self.tE = self.tE - self.tF[0]
        # self.tF = self.tF - self.tF[0]
        # tF_interpolation = self.tF.numpy()
        # print(f' -- Max tF: {tF_interpolation[-1]}')
        # self.per_ts = (tF_interpolation[1:] - tF_interpolation[:-1]).mean()
        # print(f' -- Per ts: {self.per_ts}')
        # if len(self.tF) != len(framelist):
        #     print(' -- Warning: tF and framelist have different length, some frame unavailable')
        #     indx = [int(file_name[-4:]) for file_name in framelist] #这里要保证name是从0开始
        #     tF_interpolation = tF_interpolation[indx]
        #     self.name2idx = {int(name[-4:]): idx for idx, name in enumerate(framelist)}
        # else:
        #     self.name2idx = {int(name[-4:]): idx for idx, name in enumerate(framelist)}
        # self.tF_interpolation = tF_interpolation

        # zju-mocap
        # camera parameters are static for some dataset
        tF_dict = self.event_info['tF']
        train_framelist = self.event_info['train_list']
        tF_interpolation = [tF_dict[frame_name] for frame_name in train_framelist]
        tF_interpolation = np.array(tF_interpolation)
        print(f' -- Length tF: {len(tF_interpolation)}')
        print(f' -- Max tF: {tF_interpolation[-1]}')
        self.per_ts = tF_interpolation[1] - tF_interpolation[0]
        print(f' -- Per ts: {self.per_ts}')
        self.per_name = int(train_framelist[1][-4:]) - int(train_framelist[0][-4:])
        print(f' -- Per name: {self.per_name}')
        self.tF_interpolation = tF_interpolation

    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        # load betas
        beta_path = os.path.join(self.dataset_path, 'avg_beta.pkl')
        with open(beta_path, 'rb') as f:
            self.avg_beta = pickle.load(f)
        

        return canonical_joints, canonical_bbox

    def load_train_cameras(self):
        cameras = None
        with open(os.path.join(self.dataset_path, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return cameras

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self):
        mesh_infos = None
        with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:   
            mesh_infos = pickle.load(f)

        for frame_name in mesh_infos.keys():
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox

        return mesh_infos

    def load_train_frames(self):
        img_paths = list_files(os.path.join(self.dataset_path, 'images'),
                               exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]
    
    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32')
        }

    @staticmethod
    def select_rays(select_inds, *data_list):
        return [data[select_inds] for data in data_list]
    
    def get_patch_ray_indices(
            self, 
            N_patch, 
            ray_mask, 
            subject_mask, 
            bbox_mask,
            patch_size, 
            H, W):
        assert subject_mask.dtype == np.bool
        assert bbox_mask.dtype == np.bool

        bbox_exclude_subject_mask = np.bitwise_and(
            bbox_mask,
            np.bitwise_not(subject_mask)
        )

        list_ray_indices = []
        list_mask = []
        list_xy_min = []
        list_xy_max = []

        total_rays = 0
        patch_div_indices = [total_rays]
        for _ in range(N_patch):
            # let p = cfg.patch.sample_subject_ratio
            # prob p: we sample on subject area
            # prob (1-p): we sample on non-subject area but still in bbox
            if np.random.rand(1)[0] < cfg.patch.sample_subject_ratio:
                candidate_mask = subject_mask 
            else:
                candidate_mask = bbox_exclude_subject_mask 

            ray_indices, mask, xy_min, xy_max = \
                self._get_patch_ray_indices(ray_mask, candidate_mask, 
                                            patch_size, H, W)

            assert len(ray_indices.shape) == 1
            total_rays += len(ray_indices)

            list_ray_indices.append(ray_indices)
            list_mask.append(mask)
            list_xy_min.append(xy_min)
            list_xy_max.append(xy_max)
            
            patch_div_indices.append(total_rays)

        select_inds = np.concatenate(list_ray_indices, axis=0)
        patch_info = {
            'mask': np.stack(list_mask, axis=0),
            'xy_min': np.stack(list_xy_min, axis=0),
            'xy_max': np.stack(list_xy_max, axis=0)
        }
        patch_div_indices = np.array(patch_div_indices)

        return select_inds, patch_info, patch_div_indices


    def _get_patch_ray_indices(
            self, 
            ray_mask, 
            candidate_mask, 
            patch_size, 
            H, W):

        assert len(ray_mask.shape) == 1
        assert ray_mask.dtype == np.bool
        assert candidate_mask.dtype == np.bool

        valid_ys, valid_xs = np.where(candidate_mask)

        # determine patch center
        select_idx = np.random.choice(valid_ys.shape[0], 
                                      size=[1], replace=False)[0]
        center_x = valid_xs[select_idx]
        center_y = valid_ys[select_idx]

        # determine patch boundary
        half_patch_size = patch_size // 2
        x_min = np.clip(a=center_x-half_patch_size, 
                        a_min=0, 
                        a_max=W-patch_size)
        x_max = x_min + patch_size
        y_min = np.clip(a=center_y-half_patch_size,
                        a_min=0,
                        a_max=H-patch_size)
        y_max = y_min + patch_size

        sel_ray_mask = np.zeros_like(candidate_mask)
        sel_ray_mask[y_min:y_max, x_min:x_max] = True

        #####################################################
        ## Below we determine the selected ray indices
        ## and patch valid mask

        sel_ray_mask = sel_ray_mask.reshape(-1)
        inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
        select_masked_inds = np.where(inter_mask)

        masked_indices = np.cumsum(ray_mask) - 1
        select_inds = masked_indices[select_masked_inds]
        
        inter_mask = inter_mask.reshape(H, W)

        return select_inds, \
                inter_mask[y_min:y_max, x_min:x_max], \
                np.array([x_min, y_min]), np.array([x_max, y_max])
    
    def load_image(self, frame_name, bg_color):
        imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.dataset_path, 
                                'masks', 
                                '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))
        
        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        alpha_mask = (alpha_mask > 0.2).astype('float32')
        img = orig_img
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask


    def get_total_frames(self):
        return len(self.framelist)

    def sample_patch_rays(self, img, event, H, W,
                          subject_mask, bbox_mask, ray_mask,
                          rays_o, rays_d, ray_img, near, far):

        select_inds, patch_info, patch_div_indices = \
            self.get_patch_ray_indices(
                N_patch=cfg.patch.N_patches, 
                ray_mask=ray_mask, 
                subject_mask=subject_mask, 
                bbox_mask=bbox_mask,
                patch_size=cfg.patch.size, 
                H=H, W=W)

        rays_o, rays_d, ray_img, near, far = self.select_rays(
            select_inds, rays_o, rays_d, ray_img, near, far)
        
        targets = []
        targets_event = []
        for i in range(cfg.patch.N_patches):
            x_min, y_min = patch_info['xy_min'][i] 
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(img[y_min:y_max, x_min:x_max])
            targets_event.append(event[y_min:y_max, x_min:x_max])
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)
        target_patches_event = np.stack(targets_event, axis=0) # (N_patches, P, P, 3)

        patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

        return rays_o, rays_d, ray_img, near, far, \
                target_patches, target_patches_event, patch_masks, patch_div_indices, select_inds

    def __len__(self):
        return self.get_total_frames()

    def __getitem__(self, idx):
        frame_name = self.framelist[idx]
        results = {
            'frame_name': frame_name
        }
        # real-world
        # ts = self.tF_interpolation[self.name2idx[int(frame_name[-4:])]]
        # zju-mocap
        ts = self.per_ts/self.per_name * (int(frame_name[-4:]))
        results.update({
            'ts': 16 * np.pi * np.array(ts).reshape(-1,1).astype('float32')/self.tF_interpolation[-1].astype('float32')
        })
        delta_ts = self.per_ts / 10
        next_ts = ts + delta_ts  
        results.update({
            'next_ts': 16 * np.pi * np.array(next_ts).reshape(-1,1).astype('float32')/self.tF_interpolation[-1].astype('float32')
        })

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        img, alpha = self.load_image(frame_name, bgcolor)
        use_bg = np.random.rand(1)[0] < self.gama
        if not use_bg:
            img = alpha * img + (1.0 - alpha) * bgcolor[None, None, :]

        img = (img / 255.).astype('float32')

        H, W = img.shape[0:2]

        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']

        assert frame_name in self.cameras
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2] *= cfg.resize_img_scale

        E = self.cameras[frame_name]['extrinsics']
        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]

        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        ray_img = img.reshape(-1, 3) 
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')
        
        if use_bg:
            uv = np.mgrid[:H, :W]
            uv = uv.transpose(1, 2, 0).astype(np.float32)
            uv = uv.reshape(-1, 2) / np.array([H, W])[None, :]
            uv = uv[ray_mask]

        if self.ray_shoot_mode == 'image':
            if self.data_type == 'train':
                sample_pix = 2048
                pnum = rays_o.shape[0]
                select_inds=torch.rand(pnum)<float(sample_pix)/float(pnum)
                rays_o, rays_d, ray_img, near, far, ray_event, next_ray_img = self.select_rays(
                select_inds, rays_o, rays_d, ray_img, near, far, ray_event, next_ray_img)
                if use_bg:
                    uv = uv[select_inds]
                all_sub_rays_o = [sub_rays_o[select_inds] for sub_rays_o in all_sub_rays_o]
                all_sub_rays_d = [sub_rays_d[select_inds] for sub_rays_d in all_sub_rays_d]
            else:
                pass
        elif self.ray_shoot_mode == 'patch':
            rays_o, rays_d, ray_img, near, far, \
            target_patches, target_patches_event, patch_masks, patch_div_indices, select_inds = \
                self.sample_patch_rays(img=img, event=img, H=H, W=W,
                                       subject_mask=alpha[:, :, 0] > 0.,
                                       bbox_mask=ray_mask.reshape(H, W),
                                       ray_mask=ray_mask,
                                       rays_o=rays_o, 
                                       rays_d=rays_d, 
                                       ray_img=ray_img, 
                                       near=near, 
                                       far=far)
            if use_bg:
                uv = uv[select_inds]
        else:
            assert False, f"Ivalid Ray Shoot Mode: {self.ray_shoot_mode}"
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'next_rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor})
            if use_bg:
                results['uv'] = uv

            if self.ray_shoot_mode == 'patch':
                results.update({
                    'patch_div_indices': patch_div_indices,
                    'patch_masks': patch_masks,
                    'target_patches': target_patches})
            elif self.ray_shoot_mode == 'image':
                results['target_rgbs'] = ray_img

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints
                )
            cnl_gtfms = get_canonical_global_tfms(
                            self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms
            })

            # next ray
            next_dst_Rs, next_dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints
                )
            next_cnl_gtfms = get_canonical_global_tfms(
                            self.canonical_joints)
            results.update({
                'next_dst_Rs': next_dst_Rs,
                'next_dst_Ts': next_dst_Ts,
                'next_cnl_gtfms': next_cnl_gtfms
            })

        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = self.motion_weights_priors.copy()

        # get the bounding box of canonical volume
        if 'cnl_bbox' in self.keyfilter:
            min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
            max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
            })

            # next ray
            next_dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'next_dst_posevec': next_dst_posevec_69,
            })

        return results
