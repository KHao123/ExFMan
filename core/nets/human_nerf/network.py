import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import MotionBasisComputer
from core.nets.human_nerf.component_factory import \
    load_positional_embedder, \
    load_canonical_mlp, \
    load_mweight_vol_decoder, \
    load_pose_decoder, \
    load_non_rigid_motion_mlp

from configs import cfg
from core.nets.human_nerf.mweight_vol_decoders.deconv_vol_decoder import BackGroundDecoder
from third_parties.smpl.smpl_numpy import SMPL
import trimesh
import numpy as np
from pykeops.torch import LazyTensor
import random

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(
                                        total_bones=cfg.total_bones)

        # motion weight volume
        self.mweight_vol_decoder = load_mweight_vol_decoder(cfg.mweight_volume.module)(
            embedding_size=cfg.mweight_volume.embedding_size,
            volume_size=cfg.mweight_volume.volume_size,
            total_bones=cfg.total_bones
        )

        # non-rigid motion st positional encoding
        self.get_non_rigid_embedder = \
            load_positional_embedder(cfg.non_rigid_embedder.module)

        # non-rigid motion MLP
        _, non_rigid_pos_embed_size = \
            self.get_non_rigid_embedder(cfg.non_rigid_motion_mlp.multires, 
                                        cfg.non_rigid_motion_mlp.i_embed)
        self.non_rigid_mlp = \
            load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(
                pos_embed_size=non_rigid_pos_embed_size,
                condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
                mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
                mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
                skips=cfg.non_rigid_motion_mlp.skips)
        self.non_rigid_mlp = \
            nn.DataParallel(
                self.non_rigid_mlp,
                device_ids=cfg.secondary_gpus,
                output_device=cfg.secondary_gpus[0])

        # canonical positional encoding
        get_embedder = load_positional_embedder(cfg.embedder.module)
        cnl_pos_embed_fn, cnl_pos_embed_size = \
            get_embedder(cfg.canonical_mlp.multires, 
                         cfg.canonical_mlp.i_embed)
        self.pos_embed_fn = cnl_pos_embed_fn

        # canonical mlp 
        skips = [4]
        self.cnl_mlp = \
            load_canonical_mlp(cfg.canonical_mlp.module)(
                input_ch=cnl_pos_embed_size, 
                mlp_depth=cfg.canonical_mlp.mlp_depth, 
                mlp_width=cfg.canonical_mlp.mlp_width,
                skips=skips)
        self.cnl_mlp = \
            nn.DataParallel(
                self.cnl_mlp,
                device_ids=cfg.secondary_gpus,
                output_device=cfg.primary_gpus[0])

        ts_embed_fn, ts_embed_size = \
            get_embedder(20,
                         0,
                         1)
        self.ts_embed_fn = ts_embed_fn
        # pose decoder MLP
        self.pose_decoder = \
            load_pose_decoder(cfg.pose_decoder.module)(
                embedding_size=cfg.pose_decoder.embedding_size + ts_embed_size,
                mlp_width=cfg.pose_decoder.mlp_width,
                mlp_depth=cfg.pose_decoder.mlp_depth)

        self.background_decoder = BackGroundDecoder()

    # surface vertice
    def load_smpl(self, avg_betas=None):
        betas = avg_betas['avg_beta'] if avg_betas is not None else np.zeros(10,)
        sex = avg_betas['sex'] if avg_betas is not None else 'neutral'
        body_model = SMPL(sex=sex, model_dir='./third_parties/smpl/models')
        
        verts, joints = body_model(np.zeros(72,), betas)
        base_mesh = trimesh.Trimesh(vertices=verts,#smpl_output.vertices[0].detach(),
                                    faces=body_model.faces,
                                    process=False,
                                    maintain_order=True)
        vertex_normals = base_mesh.vertex_normals
        self.point_base = torch.tensor(verts, dtype=torch.float32, requires_grad=False)
        self.normal_base = torch.tensor(vertex_normals, dtype=torch.float32, requires_grad=False)        
        # self.faces = torch.tensor(body_model.faces, dtype=torch.long, requires_grad=False)
        self.base_mesh = base_mesh.simplify_quadratic_decimation(int(body_model.faces.shape[0] / 5))

        # self.mesh = Meshes(verts=[torch.tensor(verts)], faces=[torch.tensor(body_model.faces)])

        
    def deploy_mlps_to_secondary_gpus(self):
        self.cnl_mlp = self.cnl_mlp.to(cfg.secondary_gpus[0])
        if self.non_rigid_mlp:
            self.non_rigid_mlp = self.non_rigid_mlp.to(cfg.secondary_gpus[0])

        return self


    def _query_mlp(
            self,
            pos_xyz,
            pos_embed_fn, 
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)

        result = self._apply_mlp_kernals(
                        pos_flat=pos_flat,
                        pos_embed_fn=pos_embed_fn,
                        non_rigid_mlp_input=non_rigid_mlp_input,
                        non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                        chunk=chunk)

        output = {}

        raws_flat = result['raws']
        output['raws'] = torch.reshape(
                            raws_flat, 
                            list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])

        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))


    def _apply_mlp_kernals(
            self, 
            pos_flat,
            pos_embed_fn,
            non_rigid_mlp_input,
            non_rigid_pos_embed_fn,
            chunk):
        raws = []

        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start

            xyz = pos_flat[start:end]

            if not cfg.ignore_non_rigid_motions:
                non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz)
                result = self.non_rigid_mlp(
                    pos_embed=non_rigid_embed_xyz,
                    pos_xyz=xyz,
                    condition_code=self._expand_input(non_rigid_mlp_input, total_elem)
                )
                xyz = result['xyz']

            xyz_embedded = pos_embed_fn(xyz)
            raws += [self.cnl_mlp(
                        pos_embed=xyz_embedded)]
            # raws += [self.cnl_mlp(
                        # xyz=xyz)]

        output = {}
        output['raws'] = torch.cat(raws, dim=0).to(cfg.primary_gpus[0])

        return output


    def _batchify_rays(self, rays_flat, next_rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], cfg.chunk):
            ret = self._render_rays(rays_flat[i:i+cfg.chunk], next_rays_flat[i:i+cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, uv, bgcolor=None, bgcolor_nn=None, velocity=None):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
        alpha = alpha * raw_mask[:, :, 0]

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
        
        uncert_map = nn.Softplus()(raw[..., 4]) + 0.01
        uncert_map = torch.sum(weights * weights * uncert_map, -1) 
        velocity_map = torch.sum(F.relu(weights) * velocity, -1)
        # velocity_map = torch.mean(torch.ones_like(weights).cuda() * velocity, -1)
        

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        if uv is not None:
            uv = uv * 2 - 1
            bgcolor_raw = F.grid_sample(input=bgcolor_nn[None, :, :, :], grid=uv[None, None, :, :], padding_mode='zeros', align_corners=True)
            bgcolor = torch.sigmoid(bgcolor_raw[0, :, 0, :].T[...,:3])
            bg_uncert = nn.Softplus()(bgcolor_raw[0, :, 0, :].T[...,3]) + 0.01
        else:
            bgcolor = bgcolor[None, :]/255.
            bg_uncert = 0.01
        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor
        uncert_map = uncert_map + (1.-acc_map) * bg_uncert
        

        return rgb_map, acc_map, weights, depth_map, uncert_map, velocity_map


    @staticmethod
    def _sample_motion_fields(
            pts,
            motion_scale_Rs, 
            motion_Ts, 
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            output_list):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] 

        weights_list = []
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            pos = (pos - cnl_bbox_min_xyz[None, :]) \
                            * cnl_bbox_scale_xyz[None, :] - 1.0 
            weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :], 
                                    grid=pos[None, None, None, :, :],           
                                    padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None] 
            weights_list.append(weights) 
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]

        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, 
                                                dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum

        x_skel = x_skel.reshape(orig_shape[:2]+[3])
        backwarp_motion_weights = \
            backwarp_motion_weights.reshape(orig_shape[:2]+[total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])

        results = {}
        
        if 'x_skel' in output_list: # [N_rays x N_samples, 3]
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list: # [N_rays x N_samples, 1]
            results['fg_likelihood_mask'] = fg_likelihood_mask
        
        return results


    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        if ray_batch.shape[-1] > 8:
            uv = ray_batch[...,8:10]
        else:
            uv = None
        return rays_o, rays_d, near, far, uv


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals


    def _render_rays(
            self, 
            ray_batch, 
            next_ray_batch,
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz,
            pos_embed_fn,
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input=None,
            bgcolor=None,
            bgcolor_nn=None,
            next_motion_scale_Rs=None,
            next_motion_Ts=None,
            ts=None,
            next_ts=None,
            **_):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far, uv = self._unpack_ray_batch(ray_batch)
        next_rays_o, next_rays_d, _, _, _ = self._unpack_ray_batch(next_ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far)
        if cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        next_pts = next_rays_o[...,None,:] + next_rays_d[...,None,:] * z_vals[...,:,None]
        mv_output = self._sample_motion_fields(
                            pts=pts,
                            motion_scale_Rs=motion_scale_Rs[0], 
                            motion_Ts=motion_Ts[0], 
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            output_list=['x_skel', 'fg_likelihood_mask'])
        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']

        next_mv_output = self._sample_motion_fields(
                            pts=next_pts,
                            motion_scale_Rs=next_motion_scale_Rs[0],
                            motion_Ts=next_motion_Ts[0],
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz,
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            output_list=['x_skel', 'fg_likelihood_mask'])
        # next_pts_mask = next_mv_output['fg_likelihood_mask']
        next_cnl_pts = next_mv_output['x_skel']

        velocity_xyz = (next_cnl_pts - cnl_pts) / (next_ts - ts).unsqueeze(-1).detach() 

        point_base = self.point_base.to(velocity_xyz.device)
        x_i = LazyTensor(cnl_pts.reshape(-1,3)[:,None,:]) 
        y_j = LazyTensor(point_base[None,:,:])  
        # We can now perform large-scale computations, without memory overflows:
        D_ij = ((x_i - y_j)**2).sum(dim=2)  
        idx_1 = D_ij.argKmin(1,dim=1)

        normal_xyz = self.normal_base.to(velocity_xyz.device)[idx_1].reshape(cnl_pts.shape)
        velocity_xyz = velocity_xyz - torch.sum(velocity_xyz * normal_xyz, dim=-1, keepdim=True) * normal_xyz


        velocity = torch.norm(velocity_xyz, dim=-1)

        query_result = self._query_mlp(
                                pos_xyz=cnl_pts,
                                non_rigid_mlp_input=non_rigid_mlp_input,
                                pos_embed_fn=pos_embed_fn,
                                non_rigid_pos_embed_fn=non_rigid_pos_embed_fn)
        raw = query_result['raws']
        
        rgb_map, acc_map, _, depth_map, uncert_map, velocity_map = \
            self._raw2outputs(raw, pts_mask, z_vals, rays_d, uv, bgcolor, bgcolor_nn, velocity)
        

        smpl_reg = torch.tensor(0.0, requires_grad=True).float().cuda()


        return {'rgb' : rgb_map,  
                'acc' : acc_map, 
                'depth': depth_map,
                'uncert': uncert_map,
                'velocity': velocity_map,
                'smpl_reg': smpl_reg.reshape(-1)}


    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(
                                        dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts


    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)

    
    def forward(self,
                rays, 
                dst_Rs, dst_Ts, cnl_gtfms,
                motion_weights_priors,
                dst_posevec=None,
                near=None, far=None,
                iter_val=1e7,
                uv=None,
                ts=None,
                next_ts=None,
                next_rays=None,
                next_dst_Rs=None, next_dst_Ts=None, next_cnl_gtfms=None,
                next_dst_posevec=None,
                **kwargs):

        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        dst_posevec=dst_posevec[None, ...]
        cnl_gtfms=cnl_gtfms[None, ...]
        next_dst_Rs=next_dst_Rs[None, ...]
        next_dst_Ts=next_dst_Ts[None, ...]
        next_dst_posevec=next_dst_posevec[None, ...]
        next_cnl_gtfms=next_cnl_gtfms[None, ...]
        motion_weights_priors=motion_weights_priors[None, ...]

        # correct body pose
        if iter_val >= cfg.pose_decoder.get('kick_in_iter', 0):
            ts_embedded = self.ts_embed_fn(ts)
            pose_out = self.pose_decoder(dst_posevec, ts_embedded)
            refined_Rs = pose_out['Rs']
            refined_Ts = pose_out.get('Ts', None)
            
            dst_Rs_no_root = dst_Rs[:, 1:, ...]
            dst_Rs_no_root = self._multiply_corrected_Rs(
                                        dst_Rs_no_root, 
                                        refined_Rs)
            dst_Rs = torch.cat(
                [dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

            if refined_Ts is not None:
                dst_Ts = dst_Ts + refined_Ts

            # next ts
            next_ts_embedded = self.ts_embed_fn(next_ts)
            next_pose_out = self.pose_decoder(next_dst_posevec, next_ts_embedded)
            # next_pose_out = self.pose_decoder(next_dst_posevec)
            next_refined_Rs = next_pose_out['Rs']
            next_refined_Ts = next_pose_out.get('Ts', None)
            
            next_dst_Rs_no_root = next_dst_Rs[:, 1:, ...]
            next_dst_Rs_no_root = self._multiply_corrected_Rs(
                                        next_dst_Rs_no_root,
                                        next_refined_Rs)
            next_dst_Rs = torch.cat(
                [next_dst_Rs[:, 0:1, ...], next_dst_Rs_no_root], dim=1)
            
            if next_refined_Ts is not None:
                next_dst_Ts = next_dst_Ts + next_refined_Ts

        non_rigid_pos_embed_fn, _ = \
            self.get_non_rigid_embedder(
                multires=cfg.non_rigid_motion_mlp.multires,                         
                is_identity=cfg.non_rigid_motion_mlp.i_embed,
                iter_val=iter_val,)

        if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter:
            # mask-out non_rigid_mlp_input 
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
            next_non_rigid_mlp_input = torch.zeros_like(next_dst_posevec) * next_dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec
            next_non_rigid_mlp_input = next_dst_posevec

        kwargs.update({
            "pos_embed_fn": self.pos_embed_fn,
            "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "non_rigid_mlp_input": non_rigid_mlp_input,
            "next_non_rigid_mlp_input": next_non_rigid_mlp_input,
        })

        motion_scale_Rs, motion_Ts = self._get_motion_base(
                                            dst_Rs=dst_Rs, 
                                            dst_Ts=dst_Ts, 
                                            cnl_gtfms=cnl_gtfms)
        next_motion_scale_Rs, next_motion_Ts = self._get_motion_base(
                                            dst_Rs=next_dst_Rs,
                                            dst_Ts=next_dst_Ts,
                                            cnl_gtfms=next_cnl_gtfms)
        motion_weights_vol = self.mweight_vol_decoder(
            motion_weights_priors=motion_weights_priors)
        motion_weights_vol=motion_weights_vol[0] # remove batch dimension

        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'ts': ts,
            'motion_weights_vol': motion_weights_vol,
            'bgcolor_nn': self.background_decoder()[0],
            'next_motion_scale_Rs': next_motion_scale_Rs,
            'next_motion_Ts': next_motion_Ts,
            'next_ts': next_ts,
        })

        rays_o, rays_d = rays
        rays_shape = rays_d.shape 

        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        if uv is not None:
            uv = torch.reshape(uv, [-1,2]).float()
            packed_ray_infos = torch.cat([rays_o, rays_d, near, far, uv], -1)
        else:
            packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        next_rays_o, next_rays_d = next_rays
        next_rays_o = torch.reshape(next_rays_o, [-1,3]).float()
        next_rays_d = torch.reshape(next_rays_d, [-1,3]).float()
        if uv is not None:
            next_uv = torch.reshape(uv, [-1,2]).float()
            next_packed_ray_infos = torch.cat([next_rays_o, next_rays_d, near, far, next_uv], -1)
        else:
            next_packed_ray_infos = torch.cat([next_rays_o, next_rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, next_packed_ray_infos, **kwargs)
        for k in all_ret:
            if k in ['smpl_reg']:
                continue
            k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_shape)

        all_ret.update({
            'dst_Rs': dst_Rs,
            'dst_Ts': dst_Ts
        })
        return all_ret

    def sample_points(self,
                pts, 
                dst_Rs, dst_Ts, cnl_gtfms,
                motion_weights_priors,
                dst_posevec=None,
                near=None, far=None,
                iter_val=1e7,
                uv=None,
                ts=None,
                **kwargs):

        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        dst_posevec=dst_posevec[None, ...]
        cnl_gtfms=cnl_gtfms[None, ...]
        next_dst_Rs=next_dst_Rs[None, ...]
        next_dst_Ts=next_dst_Ts[None, ...]
        next_dst_posevec=next_dst_posevec[None, ...]
        next_cnl_gtfms=next_cnl_gtfms[None, ...]
        motion_weights_priors=motion_weights_priors[None, ...]

        # correct body pose
        if iter_val >= cfg.pose_decoder.get('kick_in_iter', 0):
            ts_embedded = self.ts_embed_fn(ts)
            pose_out = self.pose_decoder(dst_posevec, ts_embedded)
            refined_Rs = pose_out['Rs']
            refined_Ts = pose_out.get('Ts', None)
            
            dst_Rs_no_root = dst_Rs[:, 1:, ...]
            dst_Rs_no_root = self._multiply_corrected_Rs(
                                        dst_Rs_no_root, 
                                        refined_Rs)
            dst_Rs = torch.cat(
                [dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

            if refined_Ts is not None:
                dst_Ts = dst_Ts + refined_Ts


        non_rigid_pos_embed_fn, _ = \
            self.get_non_rigid_embedder(
                multires=cfg.non_rigid_motion_mlp.multires,                         
                is_identity=cfg.non_rigid_motion_mlp.i_embed,
                iter_val=iter_val,)

        if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter:
            # mask-out non_rigid_mlp_input 
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
            next_non_rigid_mlp_input = torch.zeros_like(next_dst_posevec) * next_dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec
            next_non_rigid_mlp_input = next_dst_posevec


        kwargs.update({
            "pos_embed_fn": self.pos_embed_fn,
            "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "non_rigid_mlp_input": non_rigid_mlp_input,
            "next_non_rigid_mlp_input": next_non_rigid_mlp_input,
        })

        motion_scale_Rs, motion_Ts = self._get_motion_base(
                                            dst_Rs=dst_Rs, 
                                            dst_Ts=dst_Ts, 
                                            cnl_gtfms=cnl_gtfms)
        motion_weights_vol = self.mweight_vol_decoder(
            motion_weights_priors=motion_weights_priors)
        motion_weights_vol=motion_weights_vol[0] # remove batch dimension

        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'ts': ts,
            'motion_weights_vol': motion_weights_vol,
            'bgcolor_nn': self.background_decoder()[0],
        
        })
        cnl_bbox_min_xyz = kwargs['cnl_bbox_min_xyz']
        cnl_bbox_scale_xyz = kwargs['cnl_bbox_scale_xyz']
        pos_embed_fn = kwargs['pos_embed_fn']


        mv_output = self._sample_motion_fields(
                            pts=pts,
                            motion_scale_Rs=motion_scale_Rs[0], 
                            motion_Ts=motion_Ts[0], 
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            output_list=['x_skel', 'fg_likelihood_mask'])
        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']
        query_result = self._query_mlp(
                                pos_xyz=cnl_pts,
                                non_rigid_mlp_input=non_rigid_mlp_input,
                                pos_embed_fn=pos_embed_fn,
                                non_rigid_pos_embed_fn=non_rigid_pos_embed_fn)
        raw = query_result['raws']

        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)


        # alpha = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
        # alpha = alpha * pts_mask[:, :, 0]
        raw[..., :3] = torch.sigmoid(raw[..., :3])
        return raw