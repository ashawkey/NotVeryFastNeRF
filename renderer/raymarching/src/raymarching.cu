#include <cmath>
#include <cstdio>

#define PI 3.141592653589793

/////////////////////////////////////////////////////////
// nerf ver.
/////////////////////////////////////////////////////////


__global__ void raymarhcing_cuda_kernel(int H, int W, int D, int N, float near, float far, float xmin, float ixr, float ymin, float iyr, float zmin, float izr,
                                        const float* __restrict__ volume, // [H, W, D, 4]
                                        const float* __restrict__ rays_o, // [N, 3]
                                        const float* __restrict__ rays_d, // [N, 3]
                                        float* __restrict__ rgb // [N, 3]
                                        )
{
    // locate     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // in range
    if (tid < N) {
        // get ray 
        float o_x = rays_o[tid * 3];
        float o_y = rays_o[tid * 3 + 1];
        float o_z = rays_o[tid * 3 + 2];
        
        float d_x = rays_d[tid * 3];
        float d_y = rays_d[tid * 3 + 1];
        float d_z = rays_d[tid * 3 + 2];

        //printf("tid = %d, (%f, %f, %f) --> (%f, %f, %f)\n", tid, o_x, o_y, o_z, d_x, d_y, d_z);

        // march ray
        float r = 0, g = 0, b = 0;
        float transmittance = 1;
        float step_size = (far - near) / 64; // 64 steps (hard coded)
        
        for (float t = near; t < far; t += step_size) {
            // query nerf --> look up cache
            // xyz position --> matrix coordinate.
            int x = ((o_x + t * d_x) - xmin) * ixr * H;
            int y = ((o_y + t * d_y) - ymin) * iyr * W;
            int z = ((o_z + t * d_z) - zmin) * izr * D;
            // TODO: dir --> coordinate.
            
            // outside the volume, can be further accelerated by BVH ...
            if ((x < 0) || (y < 0) || (z < 0) || (x >= H) || (y >= W) || (z >= D)) continue;

            int idx = x * (W * D * 4) + y * (D * 4) + z * 4;
            
            float sigma = fmaxf(volume[idx + 3], 0); // only keep positive sigma
            float alpha = 1 - exp(- sigma * step_size);
            
            r += transmittance * alpha * volume[idx];
            g += transmittance * alpha * volume[idx + 1];
            b += transmittance * alpha * volume[idx + 2];
            
            transmittance *= (1 - alpha);
        }

        // write ray, no need to be atomic
        rgb[tid * 3] = r;
        rgb[tid * 3 + 1] = g;
        rgb[tid * 3 + 2] = b;
    }
}


void raymarching_cuda_wrapper(int H, int W, int D, int N, float near, float far, float xmin, float ixr, float ymin, float iyr, float zmin, float izr, 
                              const float* volume, const float* rays_o, const float* rays_d, float* rgb) 
{    
    // 1D launch settings (each thread process one ray)

    int block_size = 1024;
    int grid_size = ((N + block_size - 1) / block_size); 

    raymarhcing_cuda_kernel<<<grid_size, block_size>>>(H, W, D, N, near, far, xmin, ixr, ymin, iyr, zmin, izr, volume, rays_o, rays_d, rgb);
}

/////////////////////////////////////////////////////////
// dense fastnerf ver.
/////////////////////////////////////////////////////////

__global__ void raymarhcing_fastnerf_cuda_kernel(int H, int W, int D, int K, int N, float near, float far, float xmin, float ixr, float ymin, float iyr, float zmin, float izr,
                                        const float* __restrict__ volume, // [H, W, D, 4*K+1]
                                        const float* __restrict__ rays_o, // [N, 3]
                                        const float* __restrict__ rays_d, // [N, 3]
                                        float* __restrict__ rgb // [N, 3]
                                        )
{
    // locate     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int K_ = 4 * K + 1;
    
    // in range
    if (tid < N) {
        // get ray 
        float o_x = rays_o[tid * 3];
        float o_y = rays_o[tid * 3 + 1];
        float o_z = rays_o[tid * 3 + 2];
        
        float d_x = rays_d[tid * 3];
        float d_y = rays_d[tid * 3 + 1];
        float d_z = rays_d[tid * 3 + 2];

        //printf("tid = %d, (%f, %f, %f) --> (%f, %f, %f)\n", tid, o_x, o_y, o_z, d_x, d_y, d_z);
        
        // dir --> coordinate.
        // redundant direction representation (should be N^2 for <theta, phi>, but now N^3 for <dx, dy, dz>)
        int nx = (d_x + 1) * 0.5 * H;
        int ny = (d_y + 1) * 0.5 * W;
        int nz = (d_z + 1) * 0.5 * D;
        int n_idx = nx * (W * D * K_) + ny * (D * K_) + nz * K_;

        // march ray
        float r = 0, g = 0, b = 0;
        float transmittance = 1;
        float step_size = (far - near) / 256; // very necessary ... to both quality and speed :(
        
        for (float t = near; t < far; t += step_size) {
            // query nerf --> look up cache
            // xyz position --> matrix coordinate.
            int x = ((o_x + t * d_x) - xmin) * ixr * H;
            int y = ((o_y + t * d_y) - ymin) * iyr * W;
            int z = ((o_z + t * d_z) - zmin) * izr * D;

            if ((x < 0) || (y < 0) || (z < 0) || (x >= H) || (y >= W) || (z >= D)) continue;
            
            int idx = x * (W * D * K_) + y * (D * K_) + z * K_;
            
            float sigma = fmaxf(volume[idx + 4 * K], 0); // only keep positive sigma
            float alpha = 1 - exp(- sigma * step_size);
            
            float r_ = 0, g_ = 0, b_ = 0;
            #pragma unroll    
            for (int i = 0; i < K; i++) {
                r_ += volume[idx + i] * volume[n_idx + 3 * K + i];
                g_ += volume[idx + K + i] * volume[n_idx + 3 * K + i];
                b_ += volume[idx + 2 * K + i] * volume[n_idx + 3 * K + i];
            }

            r += transmittance * alpha * fminf(fmaxf(r_, 0), 1);
            g += transmittance * alpha * fminf(fmaxf(g_, 0), 1);
            b += transmittance * alpha * fminf(fmaxf(b_, 0), 1);
            
            transmittance *= (1 - alpha);
        }

        // write ray, no need to be atomic
        rgb[tid * 3] = r;
        rgb[tid * 3 + 1] = g;
        rgb[tid * 3 + 2] = b;
    }
}


void raymarching_fastnerf_cuda_wrapper(int H, int W, int D, int K, int N, float near, float far, float xmin, float ixr, float ymin, float iyr, float zmin, float izr, 
                              const float* volume, const float* rays_o, const float* rays_d, float* rgb) 
{    
    // 1D launch settings (each thread process one ray)

    int block_size = 512;
    int grid_size = ((N + block_size - 1) / block_size); 

    raymarhcing_fastnerf_cuda_kernel<<<grid_size, block_size>>>(H, W, D, K, N, near, far, xmin, ixr, ymin, iyr, zmin, izr, volume, rays_o, rays_d, rgb);
}


/////////////////////////////////////////////////////////
// sparse fastnerf ver.
/////////////////////////////////////////////////////////


__device__ inline float bilinear_interp(const float* __restrict__ data, int H, int W, int K, int offset, float x, float y) {
    // data: [H, W, K], K is channel
    // return: [1], bilinear interp value at (x, y, offset)
    int x0 = floor(x), y0 = floor(y);
    int x1 = x0 + 1, y1 = y0 + 1;

    int n_idx_000 = x0 * (W * K) + y0 * K;
    int n_idx_001 = x0 * (W * K) + y1 * K;
    int n_idx_010 = x1 * (W * K) + y0 * K;
    int n_idx_011 = x1 * (W * K) + y1 * K;

    float res = data[n_idx_000 + offset] * (x1 - x) * (y1 - y) +
                data[n_idx_001 + offset] * (x1 - x) * (y - y0) +
                data[n_idx_010 + offset] * (x - x0) * (y1 - y) +
                data[n_idx_011 + offset] * (x - x0) * (y - y0);
    
    return res;
}


__global__ void raymarhcing_fastnerf_sparse_cuda_kernel(int H, int W, int D, int M, int K, int N, float near, float far, float xmin, float ixr, float ymin, float iyr, float zmin, float izr,
                                        const int* __restrict__ inds, // [H, W, D]
                                        const float* __restrict__ uvws, // [nnz, 3K+1]
                                        const float* __restrict__ beta, // [M, M, K]
                                        const float* __restrict__ rays_o, // [N, 3]
                                        const float* __restrict__ rays_d, // [N, 3]
                                        float* __restrict__ rgb, // [N, 3]
                                        bool spherical
                                        )
{
    // locate     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        // get ray 
        float o_x = rays_o[tid * 3];
        float o_y = rays_o[tid * 3 + 1];
        float o_z = rays_o[tid * 3 + 2];
        
        float d_x = rays_d[tid * 3];
        float d_y = rays_d[tid * 3 + 1];
        float d_z = rays_d[tid * 3 + 2];
        
        // direction
        // spherical coordinate system.
        float theta_idx, phi_idx;
        int didx;
        if (spherical) {
            float theta = atan2(sqrt(d_x * d_x + d_y * d_y), d_z); // [0, PI)
            float phi = atan2(d_y, d_x); // [-PI, PI)

            theta_idx = theta / PI * M;
            phi_idx = (phi + PI) / (2 * PI) * M;
        }
        // cartesian coordinate system
        else {
            int nx = round((d_x + 1) * 0.5 * M);
            int ny = round((d_y + 1) * 0.5 * M);
            int nz = round((d_z + 1) * 0.5 * M);
            didx = nx * (M * M * K) + ny * (M * K) + nz * K;
        }
        
        
        // march ray
        float r = 0, g = 0, b = 0;
        float transmittance = 1;

        float near2 = near, far2 = far;

        // mimic raytracing to determine a fine bounds (very naive, should be enhanced...)
        float step_size_coarse = (far - near) / 128; 
        for (float t = near; t <= far; t += step_size_coarse) {
            int x = round(((o_x + t * d_x) - xmin) * ixr * H);
            int y = round(((o_y + t * d_y) - ymin) * iyr * W);
            int z = round(((o_z + t * d_z) - zmin) * izr * D);
            int idx = x * W * D + y * D + z;
        
            // out of range or empty
            if ((x < 0) || (y < 0) || (z < 0) || (x >= H) || (y >= W) || (z >= D) || (inds[idx] == -1)) {
                near2 = t;
            }
            else break;
        }
        for (float t = far; t >= near; t -= step_size_coarse) {
            int x = round(((o_x + t * d_x) - xmin) * ixr * H);
            int y = round(((o_y + t * d_y) - ymin) * iyr * W);
            int z = round(((o_z + t * d_z) - zmin) * izr * D);
            int idx = x * W * D + y * D + z;
        
            // out of range or empty
            if ((x < 0) || (y < 0) || (z < 0) || (x >= H) || (y >= W) || (z >= D) || (inds[idx] == -1)) {
                far2 = t;
            }
            else break;
        }
                        
        float step_size_fine = (far2 - near2) / 64; // necessary ... to both quality and speed :(
        float weight_sum = 0;
        for (float t = near2; t <= far2; t += step_size_fine) {
            // query nerf --> look up cache
            // xyz position --> matrix coordinate.
            int x = round(((o_x + t * d_x) - xmin) * ixr * H);
            int y = round(((o_y + t * d_y) - ymin) * iyr * W);
            int z = round(((o_z + t * d_z) - zmin) * izr * D);
            int idx = x * W * D + y * D + z;

            // out of range or empty
            if ((x < 0) || (y < 0) || (z < 0) || (x >= H) || (y >= W) || (z >= D) || (inds[idx] == -1)) continue;

            int iidx = inds[idx] * (3 * K + 1); // sparse idx in uvws
            
            float sigma = fmaxf(uvws[iidx + 3 * K], 0); // assert positive sigma
            float alpha = 1 - exp(- sigma * step_size_fine);
            
            float r_ = 0, g_ = 0, b_ = 0, beta_;
            #pragma unroll    
            for (int i = 0; i < K; i++) {
                if (spherical) beta_ = bilinear_interp(beta, M, M, K, i, theta_idx, phi_idx);
                else beta_ = beta[didx + i];
                r_ += uvws[iidx + i] * beta_;
                g_ += uvws[iidx + K + i] * beta_;
                b_ += uvws[iidx + 2 * K + i] * beta_;
            }
            
            float weight = transmittance * alpha;
            r += weight * fminf(fmaxf(r_, 0), 1);
            g += weight * fminf(fmaxf(g_, 0), 1);
            b += weight * fminf(fmaxf(b_, 0), 1);

            weight_sum += weight;
            transmittance *= (1 - alpha);

            // stop if transmittance is enough small
            if (transmittance < 0.0001) break;
        }

        // use white background (instead of black)
        r += (1 - weight_sum);
        g += (1 - weight_sum);
        b += (1 - weight_sum);

        // write ray, no need to be atomic
        rgb[tid * 3] = r;
        rgb[tid * 3 + 1] = g;
        rgb[tid * 3 + 2] = b;
    }
}


void raymarching_fastnerf_sparse_cuda_wrapper(int H, int W, int D, int M, int K, int N, float near, float far, float xmin, float ixr, float ymin, float iyr, float zmin, float izr, 
                              const int* inds, const float* uvws, const float* beta, const float* rays_o, const float* rays_d, float* rgb, bool spherical) 
{    
    // 1D launch settings (each thread process one ray)

    int block_size = 1024;
    int grid_size = ((N + block_size - 1) / block_size); 

    raymarhcing_fastnerf_sparse_cuda_kernel<<<grid_size, block_size>>>(H, W, D, M, K, N, near, far, xmin, ixr, ymin, iyr, zmin, izr, inds, uvws, beta, rays_o, rays_d, rgb, spherical);
}