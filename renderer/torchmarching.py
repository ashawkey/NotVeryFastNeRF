import torch
from torch.autograd import Function
import torch.nn as nn
import numpy as np
from kornia import create_meshgrid



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


def render_rays(inds, uvws, beta, rays_o, rays_d, near, far, xyz_min, xyz_max,
                N_samples=64,
                use_disp=False,
                chunk=1024*32,
                white_back=False,
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(xyz_, dir_, z_vals):
        """
        Helper function that performs model inference.

        Inputs:
            inds: [N, N, N]
            uvws: [nnz, 3*8+1]
            beta: [M, 2M, 8]

            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            

        Outputs:
            rgb_final: (N_rays, 3) the final rgb image
            depth_final: (N_rays) depth map
            weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz_.shape[1]
        
        xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)

        # xyz: world to cache_coord
        hwd = torch.tensor(inds.shape[:3]).to(xyz_.device)
        xyz_ = (xyz_ - xyz_min) / (xyz_max - xyz_min) * hwd
        xyz_ = torch.round(xyz_).long()

        # dir: normal to sphere to cache_coord
        dir_original = dir_
        dir_ = torch.stack([
            torch.atan2((dir_[:, 0]**2 + dir_[:, 1]**2).sqrt(), dir_[:, 2]),
            torch.atan2(dir_[:, 1], dir_[:, 0]),
        ], dim=1) # [N_rays, 2]

        dir_ = (dir_ + 2 * np.pi) % (2 * np.pi) # to [0, 2pi]

        dir_max = torch.FloatTensor(np.array([np.pi, 2 * np.pi])).to(dir_.device)
        m2m = torch.tensor(beta.shape[:2]).to(dir_.device)
        dir_ = dir_ / dir_max * m2m
        dir_ = torch.round(dir_).long() % m2m # kiui CHECK: still have oor...
        
        dir_ = torch.repeat_interleave(dir_, repeats=N_samples_, dim=0) # (N_rays*N_samples_, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        for start in range(0, B, chunk):
            end = min(start + chunk, B)
            #print('chunk', start, end)
            # query cache
            xyz_chunk = xyz_[start:end] # [N, 3]
            dir_chunk = dir_[start:end] # [N, 3]

            mask_xyz = (xyz_chunk < 0) + (xyz_chunk >= hwd) # [N, 3]
            mask_xyz = 1 - mask_xyz.sum(1).bool().float().unsqueeze(1) # [N, 1], valid == 1
            xyz_chunk *= mask_xyz.long() # out-of-range coords to 0

            #print(xyz_chunk.shape, xyz_chunk.min(), xyz_chunk.max())
            #print(dir_chunk.shape, dir_chunk.min(), dir_chunk.max())

            inds_chunk = inds[xyz_chunk[:, 0], xyz_chunk[:, 1], xyz_chunk[:, 2]] # [N], index in uvws, -1 means invalid
            uvws_chunk = uvws[inds_chunk] # [N, 25]
            beta_chunk = beta[dir_chunk[:, 0], dir_chunk[:, 1]] # [N, 8]

            uvw_chunk = torch.stack([uvws_chunk[:, 0:8], uvws_chunk[:, 8:16], uvws_chunk[:, 16:24]], dim=-1) # [N, 8, 3]
            rgb_chunk = (uvw_chunk * beta_chunk.unsqueeze(-1)).sum(1) # [N, 3]
            rgb_chunk = torch.clamp(rgb_chunk, 0, 1)
            
            out = torch.cat([rgb_chunk, uvws_chunk[:, 24:25]], dim=-1) # [N, 4]
            mask_inds = (inds_chunk != -1).float().unsqueeze(1)
            out *= mask_inds * mask_xyz

            out_chunks += [out]

        out = torch.cat(out_chunks, 0)
        
        rgbsigma = out.view(N_rays, N_samples_, 4)
        rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
        sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        # kiui: but dir_ is already normalized ?
        deltas = deltas * torch.norm(dir_original.unsqueeze(1), dim=-1)

        # compute alpha by the formula (3)
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas)) # (N_rays, N_samples_)
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights

    N_rays = rays_o.shape[0]

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays_o.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    xyz_coarse_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

    rgb_coarse, depth_coarse, weights_coarse = inference(xyz_coarse_sampled, rays_d, z_vals)


    return rgb_coarse # [N_rays, 3]


######################################
## naive sparse fastnerf rendering, torch impl., slower
######################################


def marching(inds, uvws, beta, H, W, focal, c2w, near, far, xmin, xmax, ymin, ymax, zmin, zmax):
    
    # compute rays with pytorch
    directions = get_ray_directions(H, W, focal).to(c2w.device)
    rays_o, rays_d = get_rays(directions, c2w)

    # make sure these tensors are contiguous !!! (wasted 5h+, cry)
    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()

    xyz_min = torch.FloatTensor(np.array([xmin, ymin, zmin])).to(inds.device)
    xyz_max = torch.FloatTensor(np.array([xmax, ymax, zmax])).to(inds.device)

    # rendering with pytorch
    rgb = render_rays(inds, uvws, beta, rays_o, rays_d, near, far, xyz_min, xyz_max,
                N_samples=128,
                use_disp=False,
                chunk=40960000 // 10,
                white_back=True)
    

    return rgb.view(H, W, 3)



