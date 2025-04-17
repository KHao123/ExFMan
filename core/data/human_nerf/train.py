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
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class Slerp_poses():
    def __init__(self, tF, poses):
        self.slerp = []
        for i in range(24):
            pose = [p[3*i:3*i+3] for p in poses]
            self.slerp.append(Slerp(tF,  R.from_rotvec(pose)))

    def __call__(self, t):
        poses = np.zeros(72).astype('float32')
        for i in range(24):
            poses[3*i:3*i+3] = self.slerp[i](t).as_rotvec()
        return poses

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            bgcolor=None,
            gama=1,
            ray_shoot_mode='image',
            skip=1,
            data_type='train',
            use_blur=False,
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

        framelist = self.load_train_frames()
        self.framelist = framelist[::skip] # progress dataset
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
        # sort by time
        sort_idx = self.tE.argsort()
        self.tE = self.tE[sort_idx]
        self.xE = self.xE[sort_idx]
        self.yE = self.yE[sort_idx]
        self.pE = self.pE[sort_idx]

        tF_dict = self.event_info['tF']
        tF_interpolation = [tF_dict[frame_name] for frame_name in framelist]
        tF_interpolation = np.array(tF_interpolation)
        print(f' -- Max tF: {tF_interpolation[-1]}')
        self.per_ts = tF_interpolation[1] - tF_interpolation[0]
        print(f' -- Per ts: {self.per_ts}')
        self.per_name = int(framelist[1][-4:]) - int(framelist[0][-4:])
        print(f' -- Per name: {self.per_name}')

        self.tF_interpolation = tF_interpolation
        kind = 'linear'
        intrinsics = [self.cameras[frame_name]['intrinsics'][:3, :3] for frame_name in framelist]
        trans = [self.cameras[frame_name]['extrinsics'][:3, 3] for frame_name in framelist]
        rot = [self.cameras[frame_name]['extrinsics'][:3, :3] for frame_name in framelist]
        self.cameras_interpolator = {
            'intrinsics': interp1d(x=tF_interpolation, y=intrinsics, axis=0, kind=kind, bounds_error=True), #### 内参是否线性插值
            'trans': interp1d(x=tF_interpolation, y=trans, axis=0, kind=kind, bounds_error=True),
            'rot': Slerp(tF_interpolation, R.from_matrix(rot))
        }
        poses = [self.mesh_infos[frame_name]['poses'].astype('float32') for frame_name in framelist]
        dst_tpose_joints = [self.mesh_infos[frame_name]['tpose_joints'].astype('float32') for frame_name in framelist]
        bbox_max = [self.mesh_infos[frame_name]['bbox']['max_xyz'].copy() for frame_name in framelist]
        bbox_min = [self.mesh_infos[frame_name]['bbox']['min_xyz'].copy() for frame_name in framelist]
        Rh = [self.mesh_infos[frame_name]['Rh'].astype('float32') for frame_name in framelist]
        Th = [self.mesh_infos[frame_name]['Th'].astype('float32') for frame_name in framelist]
        self.mesh_infos_interpolator = {
            'poses': Slerp_poses(tF_interpolation, poses),
            'tpose_joints': interp1d(x=tF_interpolation, y=dst_tpose_joints, axis=0, kind=kind, bounds_error=True),
            'bbox_max': interp1d(x=tF_interpolation, y=bbox_max, axis=0, kind=kind, bounds_error=True),
            'bbox_min': interp1d(x=tF_interpolation, y=bbox_min, axis=0, kind=kind, bounds_error=True),
            'Rh': Slerp(tF_interpolation, R.from_rotvec(Rh)),
            'Th': interp1d(x=tF_interpolation, y=Th, axis=0, kind=kind, bounds_error=True),
        }

        self.use_blur = use_blur
        print(f' -- Use Blur: {self.use_blur}')

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
    
    def query_dst_skeleton_sub(self, ts):
        return {
            'poses': self.mesh_infos_interpolator['poses'](ts),
            'dst_tpose_joints': \
                self.mesh_infos_interpolator['tpose_joints'](ts),
            'bbox': {'min_xyz': self.mesh_infos_interpolator['bbox_min'](ts), 'max_xyz': self.mesh_infos_interpolator['bbox_max'](ts)},
            'Rh': self.mesh_infos_interpolator['Rh'](ts).as_rotvec(),
            'Th': self.mesh_infos_interpolator['Th'](ts)
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
        alpha_mask = (alpha_mask > 0.).astype('float32')
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
        targets_alpha = []
        for i in range(cfg.patch.N_patches):
            x_min, y_min = patch_info['xy_min'][i] 
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(img[y_min:y_max, x_min:x_max])
            targets_event.append(event[y_min:y_max, x_min:x_max])
            targets_alpha.append(subject_mask[y_min:y_max, x_min:x_max])
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)
        target_patches_event = np.stack(targets_event, axis=0) # (N_patches, P, P, 3)
        target_patches_alpha = np.stack(targets_alpha, axis=0) # (N_patches, P, P)

        patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

        return rays_o, rays_d, ray_img, near, far, \
                target_patches, target_patches_event, target_patches_alpha, patch_masks, patch_div_indices, select_inds

    def __len__(self):
        return self.get_total_frames()

    def __getitem__(self, idx):
        # remove the last frame
        if idx == len(self.framelist) - 1:
            idx -= 1
        if idx == 0:
            idx += 1
        frame_name = self.framelist[idx]
        results = {
            'frame_name': frame_name
        }
        ts = self.per_ts/self.per_name * (int(frame_name[-4:]))
        results.update({
            'ts': 16 * np.pi * np.array(ts).reshape(-1,1).astype('float32')/self.tF_interpolation[-1].astype('float32') 
        })

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        img, alpha = self.load_image(frame_name, bgcolor)
        next_img = np.zeros_like(img)
        use_bg = np.random.rand(1)[0] < self.gama 
        if not use_bg:
            img = alpha * img + (1.0 - alpha) * bgcolor[None, None, :]
            next_img = alpha * next_img + (1.0 - alpha) * bgcolor[None, None, :]

        img = (img / 255.).astype('float32')
        next_img = (next_img / 255.).astype('float32')

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
        next_ray_img = next_img.reshape(-1, 3)
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]
        next_ray_img = next_ray_img[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')

        # next rays
        delta_ts = self.per_ts / self.per_name
        next_ts = ts + delta_ts * int(self.per_name/2 -1)
        results.update({
            'next_ts': 16 * np.pi * np.array(next_ts).reshape(-1,1).astype('float32')/self.tF_interpolation[-1].astype('float32')
        })

        next_frame_name = f'frame_{int(int(frame_name[-4:])+(next_ts-ts)//((self.per_ts / self.per_name))):06d}'
        next_dst_skel_info = self.query_dst_skeleton(next_frame_name)
        next_dst_bbox = next_dst_skel_info['bbox']
        next_dst_poses = next_dst_skel_info['poses']
        next_dst_tpose_joints = next_dst_skel_info['dst_tpose_joints']
       
        assert next_frame_name in self.cameras
        K = self.cameras[next_frame_name]['intrinsics'][:3, :3].copy()
        K[:2] *= cfg.resize_img_scale
        E = self.cameras[next_frame_name]['extrinsics']
        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=next_dst_skel_info['Rh'],
                Th=next_dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]
        next_rays_o, next_rays_d = get_rays_from_KRT(H, W, K, R, T)
        next_rays_o = next_rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        next_rays_d = next_rays_d.reshape(-1, 3)
        next_rays_o = next_rays_o[ray_mask]
        next_rays_d = next_rays_d[ray_mask]

        # sample ray for sub-frame
        # blur_num =  self.per_name - 1 # 5 11
        blur_num = self.per_name - 1 if self.use_blur else 1
        # blur_num = 3
        ts_start = ts - (self.per_ts / self.per_name) * (blur_num // 2)
        ts_end = ts + (self.per_ts / self.per_name) * (blur_num // 2)
        all_sub_rays_o = []
        all_sub_rays_d = []
        all_sub_dst_poses = []
        all_sub_dst_tpose_joints = []
        all_sub_ts = []
        all_next_sub_rays_o = []
        all_next_sub_rays_d = []
        all_next_sub_dst_poses = []
        all_next_sub_dst_tpose_joints = []
        all_next_sub_ts = []

        all_ts = np.linspace(ts_start, ts_end, blur_num)
       
        event_ts = ts + (self.per_ts / self.per_name) * np.random.randint(-(int(self.per_name/2 - 1)), (int(self.per_name/2 - 1))+1) 
        all_ts = np.append(all_ts, event_ts)
        for sub_idx in range(all_ts.shape[0]):
            sub_ts = all_ts[sub_idx]
            if sub_idx == blur_num // 2:
                continue
            
            sub_frame_name = f'frame_{int(int(frame_name[-4:])+(sub_ts-ts)//((self.per_ts / self.per_name))):06d}'
            sub_dst_skel_info = self.query_dst_skeleton(sub_frame_name)
            sub_dst_bbox = sub_dst_skel_info['bbox']
            sub_dst_poses = sub_dst_skel_info['poses']
            sub_dst_tpose_joints = sub_dst_skel_info['dst_tpose_joints']            
            
            assert sub_frame_name in self.cameras
            K = self.cameras[sub_frame_name]['intrinsics'][:3, :3].copy()
            K[:2] *= cfg.resize_img_scale
            E = self.cameras[sub_frame_name]['extrinsics']
            ###
            E = apply_global_tfm_to_camera(
                    E=E, 
                    Rh=sub_dst_skel_info['Rh'],
                    Th=sub_dst_skel_info['Th'])
            R = E[:3, :3]
            T = E[:3, 3]
            sub_rays_o, sub_rays_d = get_rays_from_KRT(H, W, K, R, T)
            sub_rays_o = sub_rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
            sub_rays_d = sub_rays_d.reshape(-1, 3)
            sub_rays_o = sub_rays_o[ray_mask]
            sub_rays_d = sub_rays_d[ray_mask]
            all_sub_rays_o.append(sub_rays_o)
            all_sub_rays_d.append(sub_rays_d)
            all_sub_dst_poses.append(sub_dst_poses)
            all_sub_dst_tpose_joints.append(sub_dst_tpose_joints)
            all_sub_ts.append(sub_ts)

            next_sub_ts = sub_ts + delta_ts * int(self.per_name/2 -1)

            next_sub_frame_name = f'frame_{int(int(frame_name[-4:])+(next_sub_ts-ts)//((self.per_ts / self.per_name))):06d}'
            next_sub_dst_skel_info = self.query_dst_skeleton(next_sub_frame_name)
            next_sub_dst_bbox = next_sub_dst_skel_info['bbox']
            next_sub_dst_poses = next_sub_dst_skel_info['poses']
            next_sub_dst_tpose_joints = next_sub_dst_skel_info['dst_tpose_joints']

            assert next_sub_frame_name in self.cameras
            K = self.cameras[next_sub_frame_name]['intrinsics'][:3, :3].copy()
            K[:2] *= cfg.resize_img_scale
            E = self.cameras[next_sub_frame_name]['extrinsics']
            ###
            E = apply_global_tfm_to_camera(
                    E=E,
                    Rh=next_sub_dst_skel_info['Rh'],
                    Th=next_sub_dst_skel_info['Th'])
            R = E[:3, :3]
            T = E[:3, 3]
            next_sub_rays_o, next_sub_rays_d = get_rays_from_KRT(H, W, K, R, T)
            next_sub_rays_o = next_sub_rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
            next_sub_rays_d = next_sub_rays_d.reshape(-1, 3)
            next_sub_rays_o = next_sub_rays_o[ray_mask]
            next_sub_rays_d = next_sub_rays_d[ray_mask]
            all_next_sub_rays_o.append(next_sub_rays_o)
            all_next_sub_rays_d.append(next_sub_rays_d)
            all_next_sub_dst_poses.append(next_sub_dst_poses)
            all_next_sub_dst_tpose_joints.append(next_sub_dst_tpose_joints)
            all_next_sub_ts.append(next_sub_ts)
        assert len(all_sub_rays_o) == blur_num

        # event frame
        reverse = 0
        if event_ts < ts:
            reverse = 1
            ts, event_ts = event_ts, ts
        tStart = ts
        tStop = event_ts
        event_target = torch.zeros((int(H/cfg.resize_img_scale), int(W/cfg.resize_img_scale))).long()

        # search sorted list
        sliceIdx_start, sliceIdx_stop = torch.searchsorted(self.tE, torch.tensor([tStart, tStop]))
        if sliceIdx_start < sliceIdx_stop:
            xSlice = self.xE[sliceIdx_start:sliceIdx_stop+1]
            ySlice = self.yE[sliceIdx_start:sliceIdx_stop+1]
            pSlice = self.pE[sliceIdx_start:sliceIdx_stop+1]
            index = ySlice * int(W/cfg.resize_img_scale) + xSlice
            event_target.put_(index=index, source=pSlice, accumulate=True)

        event_target = event_target.float().numpy()
        if reverse == 1:
            event_target = -event_target
        if cfg.resize_img_scale != 1.:
            event_target = cv2.resize(event_target, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_NEAREST) 
        ray_event = event_target.reshape(-1, 1)
        ray_event = ray_event[ray_mask]

        
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
            target_patches, target_patches_event, target_patches_alpha, patch_masks, patch_div_indices, select_inds = \
                self.sample_patch_rays(img=img, event=event_target, H=H, W=W,
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
            ray_event = ray_event[select_inds]
            all_sub_rays_o = [sub_rays_o[select_inds] for sub_rays_o in all_sub_rays_o]
            all_sub_rays_d = [sub_rays_d[select_inds] for sub_rays_d in all_sub_rays_d]
            next_rays_o = next_rays_o[select_inds]
            next_rays_d = next_rays_d[select_inds]
            all_next_sub_rays_o = [next_sub_rays_o[select_inds] for next_sub_rays_o in all_next_sub_rays_o]
            all_next_sub_rays_d = [next_sub_rays_d[select_inds] for next_sub_rays_d in all_next_sub_rays_d]
        else:
            assert False, f"Ivalid Ray Shoot Mode: {self.ray_shoot_mode}"
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 
        next_batch_rays = np.stack([next_rays_o, next_rays_d], axis=0)

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'next_rays': next_batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor})
            if use_bg:
                results['uv'] = uv

            if self.ray_shoot_mode == 'patch':
                results.update({
                    'patch_div_indices': patch_div_indices,
                    'patch_masks': patch_masks,
                    'target_patches': target_patches,
                    'target_patches_event': target_patches_event,
                    'target_patches_alpha': target_patches_alpha.astype('float32')})
            elif self.ray_shoot_mode == 'image':
                results['target_rgbs'] = ray_img
                results['target_events'] = ray_event
                results['next_target_rgbs'] = next_ray_img

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
                    next_dst_poses, next_dst_tpose_joints
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
            next_dst_posevec_69 = next_dst_poses[3:] + 1e-2
            results.update({
                'next_dst_posevec': next_dst_posevec_69,
            })
        
        # get results for sub-frame
        all_sub_results = []
        for sub_idx in range(len(all_sub_ts)):
            sub_results = {
                'frame_name': frame_name,
                'ts': 16 * np.pi * np.array(all_sub_ts[sub_idx]).reshape(-1,1).astype('float32')/self.tF_interpolation[-1].astype('float32'), 
                'next_ts': 16 * np.pi * np.array(all_next_sub_ts[sub_idx]).reshape(-1,1).astype('float32')/self.tF_interpolation[-1].astype('float32'), 
            }
            sub_batch_rays = np.stack([all_sub_rays_o[sub_idx], all_sub_rays_d[sub_idx]], axis=0)
            next_sub_batch_rays = np.stack([all_next_sub_rays_o[sub_idx], all_next_sub_rays_d[sub_idx]], axis=0)
            if 'rays' in self.keyfilter:
                sub_results.update({
                    'img_width': W,
                    'img_height': H,
                    'ray_mask': ray_mask,
                    'rays': sub_batch_rays,
                    'next_rays': next_sub_batch_rays,
                    'near': near,
                    'far': far,
                    'bgcolor': bgcolor,
                    })
                if use_bg:
                    sub_results['uv'] = uv
            if 'motion_bases' in self.keyfilter:
                sub_dst_Rs, sub_dst_Ts = body_pose_to_body_RTs(
                        all_sub_dst_poses[sub_idx], all_sub_dst_tpose_joints[sub_idx]
                    )
                sub_cnl_gtfms = get_canonical_global_tfms(
                                self.canonical_joints)
                sub_results.update({
                    'dst_Rs': sub_dst_Rs,
                    'dst_Ts': sub_dst_Ts,
                    'cnl_gtfms': sub_cnl_gtfms
                })
                # next ray
                next_sub_dst_Rs, next_sub_dst_Ts = body_pose_to_body_RTs(
                        all_next_sub_dst_poses[sub_idx], all_next_sub_dst_tpose_joints[sub_idx]
                    )
                next_sub_cnl_gtfms = get_canonical_global_tfms(
                                self.canonical_joints)
                sub_results.update({
                    'next_dst_Rs': next_sub_dst_Rs,
                    'next_dst_Ts': next_sub_dst_Ts,
                    'next_cnl_gtfms': next_sub_cnl_gtfms
                })

            if 'motion_weights_priors' in self.keyfilter:
                sub_results['motion_weights_priors'] = self.motion_weights_priors.copy()
            if 'cnl_bbox' in self.keyfilter:
                min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
                max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
                sub_results.update({
                    'cnl_bbox_min_xyz': min_xyz,
                    'cnl_bbox_max_xyz': max_xyz,
                    'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
                })
                assert np.all(sub_results['cnl_bbox_scale_xyz'] >= 0)
            if 'dst_posevec_69' in self.keyfilter:
                # 1. ignore global orientation
                # 2. add a small value to avoid all zeros
                sub_dst_posevec_69 = all_sub_dst_poses[sub_idx][3:] + 1e-2
                sub_results.update({
                    'dst_posevec': sub_dst_posevec_69,
                })
                # next ray
                next_sub_dst_posevec_69 = all_next_sub_dst_poses[sub_idx][3:] + 1e-2
                sub_results.update({
                    'next_dst_posevec': next_sub_dst_posevec_69,
                })

            all_sub_results.append(sub_results)

        return results, all_sub_results
