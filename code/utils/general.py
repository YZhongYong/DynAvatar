import os
import torch

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def get_split_name(splits):
    # example splits: [MVI_1812, MVI_1813]
    # example output: MVI_1812+MVI_1813
    name = ''
    for s in splits:
        name += str(s)
        name += '+'
    assert len(name) > 1
    return name[:-1]


def depths_to_points(world_view_transform,full_proj_transform, depthmap):
    c2w = (world_view_transform.T).inverse()
    W, H = 512, 512
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(world_view_transform,full_proj_transform, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(world_view_transform,full_proj_transform, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output
