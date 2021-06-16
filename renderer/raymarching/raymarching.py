import torch
from torch.autograd import Function
import torch.nn as nn
from kornia import create_meshgrid

from . backend import _backend


def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    directions = torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, p=2, dim=-1, keepdim=True)

    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_o = rays_o.view(-1, 3).float()
    rays_d = rays_d.view(-1, 3).float()

    return rays_o, rays_d

######################################

class _raymarching(Function):
    @staticmethod
    def forward(ctx, volume, H, W, focal, c2w, near, far, xmin, xmax, ymin, ymax, zmin, zmax):
        
        # compute rays with pytorch
        directions = get_ray_directions(H, W, focal).to(c2w.device)
        rays_o, rays_d = get_rays(directions, c2w)

        # make sure these tensors are contiguous !!! (wasted 5h+, cry)
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        # rendering with cuda
        rgb = _backend.raymarching_cuda(volume, rays_o, rays_d, near, far, xmin, xmax, ymin, ymax, zmin, zmax)

        return rgb.view(H, W, 3)


raymarching = _raymarching.apply

class _raymarching_fastnerf(Function):
    @staticmethod
    def forward(ctx, volume, H, W, focal, c2w, near, far, xmin, xmax, ymin, ymax, zmin, zmax):
        
        # compute rays with pytorch
        directions = get_ray_directions(H, W, focal).to(c2w.device)
        rays_o, rays_d = get_rays(directions, c2w)

        # make sure these tensors are contiguous !!! (wasted 5h+, cry)
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        # rendering with cuda
        rgb = _backend.raymarching_fastnerf_cuda(volume, rays_o, rays_d, near, far, xmin, xmax, ymin, ymax, zmin, zmax)

        return rgb.view(H, W, 3)


raymarching_fastnerf = _raymarching_fastnerf.apply

class _raymarching_fastnerf_sparse(Function):
    @staticmethod
    def forward(ctx, inds, uvws, beta, H, W, focal, c2w, near, far, xmin, xmax, ymin, ymax, zmin, zmax):
        
        # compute rays with pytorch
        directions = get_ray_directions(H, W, focal).to(c2w.device)
        rays_o, rays_d = get_rays(directions, c2w)

        # make sure these tensors are contiguous !!! (wasted 5h+, cry)
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        # pre-compute inverse to avoid divisions in cuda kernel
        ixr = 1 / (xmax - xmin)
        iyr = 1 / (ymax - ymin)
        izr = 1 / (zmax - zmin)

        # if use spherical coordniate system (M, M, 3) or cartesian (M, M, M, 3)
        spherical = len(beta.shape) == 3

        # rendering with cuda
        rgb = _backend.raymarching_fastnerf_sparse_cuda(inds, uvws, beta, rays_o, rays_d, near, far, xmin, ixr, ymin, iyr, zmin, izr, spherical)

        return rgb.view(H, W, 3)


raymarching_fastnerf_sparse = _raymarching_fastnerf_sparse.apply

