import math
from functools import partial

import torch
import torch.nn as nn
from flame.FLAME import FLAME
from pytorch3d.ops import knn_points
from model.point_cloud import PointCloud

from functorch import jacfwd, vmap

from model.geometry_network import GeometryNetwork
from model.deformer_network import ForwardDeformer
from model.texture_network import RenderingNetwork
from model.gaussian_network import GaussianNetwork
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.general import depth_to_normal
import numpy as np
import open3d as o3d


print_flushed = partial(print, flush=True)


class MonogaussianAvatar(nn.Module):
    def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background):
        super().__init__()
        self.FLAMEServer = FLAME('./flame/FLAME2020/generic_model.pkl', './flame/FLAME2020/landmark_embedding.npy',
                                 n_shape=100,
                                 n_exp=50,
                                 shape_params=shape_params,
                                 canonical_expression=canonical_expression,
                                 canonical_pose=canonical_pose).cuda()
        self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
            self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)
        self.prune_thresh = conf.get_float('prune_thresh', default=0.5)
        self.geometry_network = GeometryNetwork(**conf.get_config('geometry_network'))
        self.deformer_network = ForwardDeformer(FLAMEServer=self.FLAMEServer, **conf.get_config('deformer_network'))
        self.rendering_network = RenderingNetwork(**conf.get_config('rendering_network'))
        self.gaussian_deformer_network = GaussianNetwork(**conf.get_config('gaussian_network'))
        self.ghostbone = self.deformer_network.ghostbone
        if self.ghostbone:
            self.FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().cuda(), self.FLAMEServer.canonical_transformations], 1)
        self.pc = PointCloud(**conf.get_config('point_cloud')).cuda()
        n_points = self.pc.points.shape[0]
        self.img_res = img_res
        self.use_background = use_background
        if self.use_background:
            init_background = torch.zeros(img_res[0] * img_res[1], 3).float().cuda()
            self.background = nn.Parameter(init_background)
        else:
            self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()

        self.radius = 0.15 * (0.75 ** math.log2(n_points / 100))

        self.visible_points = torch.zeros(n_points).bool().cuda()

        self.scale_ac = torch.sigmoid
        self.rotations_ac = torch.nn.functional.normalize
        self.opacity_ac = torch.sigmoid
        self.color_ac = torch.sigmoid

    def _compute_canonical_normals_and_feature_vectors(self):
        geometry_output, scales, rotations, opacity = self.geometry_network(self.pc.points.detach())
        feature_rgb_vector = geometry_output
        feature_scale_vector = scales
        feature_rotation_vector = rotations
        feature_opacity_vector = opacity
        feature_vector = torch.concat([feature_rgb_vector, feature_rotation_vector, feature_scale_vector, feature_opacity_vector], dim=1)
        if not self.training:
            self._output['pnts_albedo'] = feature_rgb_vector
        return feature_vector

    def _render(self, world_view_transform, full_proj_transform, camera_center, tanfovx, tanfovy,
                bg_color, image_h, image_w, xyz, color, scales, rotations, opacity):

        screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        raster_settings = GaussianRasterizationSettings(
            image_height=image_h,
            image_width=image_w,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=3,
            campos=camera_center,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        render_image, radii,allmap = rasterizer(
            means3D=xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=color,
            opacities=opacity,
            scales=scales+self.radius,
            rotations=rotations,
            cov3D_precomp=None)
        
            # 附加的正则化
        render_alpha = allmap[1:2]

        # 获取法线图
        # 将法线从视图空间转换到世界空间
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1,2,0) @ (world_view_transform[:3,:3].T)).permute(2,0,1)
        
        # 获取中值深度图
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # 获取预期深度图
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        
        # 获取深度失真图
        render_dist = allmap[6:7]

        # 伪表面属性
        # surf_depth 是中值或预期深度，通过设置 depth_ratio 为 1 或 0
        # 对于有界的场景，使用中值深度，即 depth_ratio = 1; 
        # 对于无界场景，使用预期深度，即 depth_ratio = 0，以减少磁盘混叠。
        depth_ratio=0
        surf_depth = render_depth_expected * (1-depth_ratio) + (depth_ratio) * render_depth_median
        
        # 假设深度点形成 '表面' 并生成伪表面法线用于正则化。
        surf_normal = depth_to_normal(world_view_transform,full_proj_transform, surf_depth)
        surf_normal = surf_normal.permute(2,0,1)
        # 记得乘以 accum_alpha，因为 render_normal 是未归一化的。
        surf_normal = surf_normal * (render_alpha).detach()

        n_points = self.pc.points.shape[0]
        id = torch.arange(start=0, end=n_points, step=1).cuda()
        visible_points = id[opacity.reshape(-1) >= self.prune_thresh]
        visible_points = visible_points[visible_points != -1]
        
        # 更新返回字典
        rets = {
                'rend_alpha': render_alpha,
                'rend_normal': render_normal,
                'rend_dist': render_dist,
                'surf_depth': surf_depth,
                'surf_normal': surf_normal,
                'render': render_image, 
                'visible_points':visible_points,
        }
        return rets



    def forward(self, input):
        self._output = {}

        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        batch_size = flame_pose.shape[0]
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)

        if self.ghostbone:

            transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)

        # gaussian splatting
        world_view_transform = input["world_view_transform"].clone()
        full_proj_transform = input["full_proj_transform"].clone()
        camera_center = input["camera_center"].clone()
        tanfovx = input["tanfovx"]
        tanfovy = input["tanfovy"]
        bg_color = input["bg_color"].clone()


        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points

        feature_vector = self._compute_canonical_normals_and_feature_vectors()
        transformed_points, rgb_points, scale_vals, rotation_vals, opacity_vals = self.get_rbg_value_functorch(
                                                                                                               pnts_c=self.pc.points,
                                                                                                               feature_vectors=feature_vector,
                                                                                                               pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                               betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                               transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                               )

        shapedirs, posedirs, lbs_weights, pnts_c_flame,expression,pose_feature= self.deformer_network.query_weights(self.pc.points.detach(),expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1))
        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        scale = scale_vals.reshape(transformed_points.shape[0], -1, 2)
        rotation = rotation_vals.reshape(transformed_points.shape[0], -1, 4)
        opacity = opacity_vals.reshape(transformed_points.shape[0], -1, 1)
        offset = transformed_points.detach() - pnts_c_flame.detach()
        offset_scale, offset_rotation, offset_opacity, offset_color = self.gaussian_deformer_network(offset)
        scale = scale + offset_scale
        rotation = rotation + offset_rotation
        opacity = opacity + offset_opacity
        rgb_points = rgb_points.reshape(batch_size, n_points, 3)


        rgb_points = rgb_points + offset_color
        rgb_points = self.color_ac(rgb_points)
        scale = self.scale_ac(scale)
        #1024
        # scale = scale * 0.01
        #512
        scale = scale * 0.025
        rotation = self.rotations_ac(rotation)
        opacity = self.opacity_ac(opacity)

        rendering_list = []
        
        rend_distmaps = []
        rend_normalmaps = []
        surf_normalmaps = []
        
        for idx in range(transformed_points.shape[0]):
            world_view_transform_i = world_view_transform[idx]
            full_proj_transform_i = full_proj_transform[idx]
            camera_center_i = camera_center[idx]
            tanfovx_i = tanfovx[idx]
            tanfovy_i = tanfovy[idx]
            bg_color_i = bg_color[idx]
            image_h_i = self.img_res[0]
            image_w_i = self.img_res[1]
            # image_h_i = 1024
            # image_w_i = 1024
            xyz_i = transformed_points[idx]
            color_i = rgb_points[idx]
            scales_i = scale[idx]
            rotations_i = rotation[idx]
            opacity_i = opacity[idx]
            render_pkg = self._render(world_view_transform_i, full_proj_transform_i, camera_center_i, tanfovx_i, tanfovy_i,
                                 bg_color_i, image_h_i, image_w_i, xyz_i, color_i, scales_i, rotations_i, opacity_i)
            image = render_pkg['render']
            visible_points = render_pkg['visible_points']
            
            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            
            rend_distmaps.append(rend_dist.unsqueeze(0))
            rend_normalmaps.append(rend_normal.unsqueeze(0))
            surf_normalmaps.append(surf_normal.unsqueeze(0))
            
            if self.training:
                self.visible_points[visible_points] = True
            rendering_list.append(image.unsqueeze(0))
        rgb_values = torch.concat(rendering_list, dim=0).permute(0, 2, 3, 1)
        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)
        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)
        
        dist_values = torch.concat(rend_distmaps, dim=0).permute(0, 2, 3, 1)
        distmaps = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)
        
        rend_normal_values = torch.concat(rend_normalmaps, dim=0).permute(0, 2, 3, 1)
        normalmaps = rend_normal_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)
        
        surf_normal_values = torch.concat(surf_normalmaps, dim=0).permute(0, 2, 3, 1)
        surf_normal = surf_normal_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)
        
        
        # rgb_image = rgb_values.reshape(batch_size, 1024, 1024, 3)
        # training outputs
        output = {
            'img_res': self.img_res,
            'batch_size': batch_size,
            'rgb_image': rgb_image,
            'canonical_points': pnts_c_flame,
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights,
            'alpha':alpha,
            'normal':normal,
            'depth':depth,
            'depth_normal':depth_normal,
            'rend_dist':distmaps,
            'rend_normal':normalmaps,
            'surf_normal':surf_normal,
            
        }

        if not self.training:
            # # print("Running tsdf volume integration ...")
            # # print(f'voxel_size: {voxel_size}')
            # # print(f'sdf_trunc: {sdf_trunc}')
            # # print(f'depth_truc: {depth_trunc}')
            # voxel_size=0.003184356764539623
            # sdf_trunc=0.015921783822698116
            # depth_trunc=3.260781326888574
            
            # volume = o3d.pipelines.integration.ScalableTSDFVolume(
            #     voxel_length= voxel_size,
            #     sdf_trunc=sdf_trunc,
            #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            # )

            # rgb = rendering_list[0].squeeze(0).detach()
            # depth = surf_normalmaps[0].squeeze(0).detach()
            
            # # make open3d rgbd
            # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            #     o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
            #     o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
            #     depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
            #     depth_scale = 1.0
            # )
            # name='fuse'    
            # volume.integrate(rgbd, intrinsic=input['intrinsics'].squeeze(0), extrinsic=world_view_transform.cpu().numpy().astype(np.float64))
            # train_dir ='/home/2dgs-avatar'
            # mesh = volume.extract_triangle_mesh()
            # o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
            # print("mesh saved at {}".format(os.path.join(train_dir, name)))
            # # post-process the mesh and save, saving the largest N clusters
            # mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
            # o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
            # print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))
                
            
            output_testing = {
                'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
                'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
                'deformed_verts': verts.reshape(-1, 3),
                'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
            }
            if self.deformer_network.deform_c:
                output_testing['unconstrained_canonical_points'] = self.pc.points
            output.update(output_testing)
        output.update(self._output)

        return output


  
    def get_rbg_value_functorch(self, pnts_c, feature_vectors, pose_feature, betas, transformations):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)
        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])
        n_points = pnts_c.shape[0]
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature):
            pnts_c = pnts_c.unsqueeze(0)
            shapedirs, posedirs, lbs_weights, pnts_c_flame,betas,pose_feature = self.deformer_network.query_weights(pnts_c,betas,pose_feature)
            shapedirs = shapedirs.expand(batch_size, -1, -1)
            posedirs = posedirs.expand(batch_size, -1, -1)
            lbs_weights = lbs_weights.expand(batch_size, -1)
            pnts_c_flame = pnts_c_flame.expand(batch_size, -1)
            pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights)
            pnts_d = pnts_d.reshape(-1)
            return pnts_d, pnts_d

        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        rgb_vals = feature_vectors[:, 0:3]
        scale_vals = feature_vectors[:, 3:5]
        rotation_vals = feature_vectors[:, 5:9]
        opacity_vals = feature_vectors[:, 9:10]
        return pnts_d, rgb_vals, scale_vals, rotation_vals, opacity_vals
