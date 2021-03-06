#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include "hdf5.h"

#define PI 3.14159265359

#ifdef HIGH_PRECISION
#define H5_REAL H5T_NATIVE_DOUBLE
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif
#define FLAG "%lf"
typedef double real_t;
#else
#define H5_REAL H5T_NATIVE_FLOAT
#define FLAG "%f"
typedef float real_t;
#endif

typedef unsigned int uint32_t;

typedef struct Particle {
public:
    bool free;
    real_t x;
    real_t px,py,pz;
    real_t gamma;
    real_t realpart;
} Particle;

typedef struct Parameters {
public:
    real_t a0, tau, xmax, xmin, dx, lambda;
    real_t x_rise, x_uni, x_fall, x_end;
    real_t t_delay, t_interval;
    real_t n_plasma;
    real_t pi_cs, orb_en;     //photoionization cross-section and orbital energy
    uint32_t n_pulse;
    uint32_t cell_part_num;
    uint32_t total_part_num;
    uint32_t ngrid;
    real_t sim_len;
} Parameters;

inline void cudaAssert(cudaError_t status)
{
   if (status != cudaSuccess) 
   {
      fprintf(stderr,"Cuda assert failed: %s.\n", cudaGetErrorString(status));
      exit(status);
   }
}

cudaError_t advanceWithCuda(real_t *ey, real_t *bz, real_t *ex, 
                            real_t *jx, real_t *jy, real_t *np,
                            Particle *particles, Parameters para);

__global__ void initializeFieldsKernel(real_t *ey, real_t *bz, real_t *ex, Parameters para) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < para.ngrid) {
        ey[i] = 0.0;
        bz[i] = 0.0;
        for (int j = 0; j < para.n_pulse; j++) {
            ey[i] += para.a0 * exp ( - ( para.xmin + i * para.dx - para.xmin - 3 * para.tau - j * para.t_interval) * ( para.xmin + i * para.dx - para.xmin - 3 * para.tau - j * para.t_interval) / ( para.tau * para.tau ) ) * sin ( 2 * PI * ( para.xmin + i * para.dx ) );
            bz[i] += para.a0 * exp ( - ( para.xmin + i * para.dx + 0.75 * para.dx - para.xmin - 3 * para.tau - j * para.t_interval) * ( para.xmin + i * para.dx + 0.75 * para.dx - para.xmin - 3 * para.tau - j * para.t_interval) / ( para.tau * para.tau ) ) * sin ( 2 * PI * ( para.xmin + i * para.dx + 0.75 * para.dx ) );
        }
        ex[i] = 0.0;
    }
}

__global__ void initializeParticlesKernel(Particle *particles, Parameters para, curandState *devStates) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t ncell = (para.x_end - para.x_rise) / para.dx;
    if(i < para.total_part_num) {
        particles[i].free = false;
        //particles[i].x = para.x_rise + curand_uniform(&devStates[threadIdx.x]) * (para.x_end - para.x_rise);
        //particles[i].x = para.x_rise + (i / para.cell_part_num) * para.dx + (i % para.cell_part_num) * para.dx / para.cell_part_num;
        particles[i].x = para.x_rise + ((i*256)%ncell) * para.dx + uint32_t((i*256)/ncell) * para.dx / para.cell_part_num;
        particles[i].px = 0.0;
        particles[i].py = 0.0;
        particles[i].pz = 0.0;
        particles[i].gamma = 1.0;
        particles[i].realpart = particles[i].x >= para.x_uni ? \
                                (particles[i].x > para.x_fall ? (para.n_plasma * (para.x_end - particles[i].x) / (para.x_end - para.x_fall)) : para.n_plasma) : \
                                (para.n_plasma * (particles[i].x - para.x_rise) / (para.x_uni - para.x_rise));
        particles[i].realpart /= para.cell_part_num;
        particles[i].realpart = particles[i].realpart >= 0.0 ? particles[i].realpart : 0.0;
    }
}

__global__ void resetCurrentsKernel(real_t *jx, real_t *jy, real_t *np, Parameters para) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < para.ngrid) {
        jx[i] = 0.0;
        jy[i] = 0.0;
        np[i] = 0.0;
    }
}

__global__ void advanceExyKernel(real_t *ey, real_t *bz, real_t *ex, real_t *jx, real_t *jy, Parameters para) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if(i < para.ngrid-1) {
        ey[i] -= ( bz[i] - bz[i-1] ) * 0.5 + para.dx * PI * jy[i];
        ex[i] -= para.dx * PI * jx[i];
    }
}

__global__ void advanceBzKernel(real_t *ey, real_t *bz, Parameters para) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < para.ngrid-2) bz[i] -= ( ey[i+1] - ey[i] ) * 0.25;
}

__global__ void advanceParticlesKernel(real_t *ey, real_t *bz, real_t *ex, 
                                       real_t *jx, real_t *jy, real_t *np,
                                       Particle *part, Parameters para, curandState *devStates) {
    //advance particles
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        
    if(i < para.total_part_num && part[i].x > para.xmin + 2 * para.dx && part[i].x < para.xmax - 2 * para.dx && (!part[i].free)) {
        uint32_t ix = (uint32_t) ((part[i].x - para.xmin) / para.dx);
        real_t eyy = 0.5*(ey[ix+1]+ey[ix-1]-2*ey[ix]) * (part[i].x - para.xmin - ix * para.dx) * (part[i].x - para.xmin - ix * para.dx) / (para.dx*para.dx) + 0.5*(ey[ix + 1]-ey[ix-1]) * (part[i].x - para.xmin - ix * para.dx) / para.dx+ey[ix];
        //real_t intensity = 1.37e24*eyy*eyy/(para.lambda*para.lambda);
        //real_t en_l = 1.98373e-16 / para.lambda;
        //real_t flux = intensity / en_l;
        //real_t ph_p = flux * para.pi_cs * 1e-18 * para.dx * 0.5 * para.lambda * 1e-9 / 2.9979e8;
        //real_t ph_p = 11518.3658 * eyy * eyy * para.pi_cs * para.dx;
        real_t w_ph = 11518.3658 * eyy * eyy * para.pi_cs;
        real_t t_ph = - log(curand_uniform(&devStates[threadIdx.x]))/w_ph;
        if (t_ph <= para.dx && 1.37e24*eyy*eyy > part[i].realpart*1.1e9*1.98373e-16 / para.lambda) {
            part[i].free = true;
            atomicAdd(jy + ix, 6.927529399977104e-37*part[i].realpart*para.lambda/(para.dx*eyy));
            real_t theta = acos(1-2*curand_uniform(&devStates[threadIdx.x]));
            real_t phi = 2*PI*curand_uniform(&devStates[threadIdx.x]);
            part[i].gamma = (1239.8317388397/para.lambda - para.orb_en)/0.511e6+1;
            real_t p_k = sqrt(part[i].gamma * part[i].gamma - 1);
            part[i].px = p_k * sin(theta) * cos(phi);
            part[i].py = p_k * sin(theta) * sin(phi);
            part[i].pz = p_k * cos(theta);
        }
        
    }
    else if(i < para.total_part_num && part[i].x > para.xmin + 2 * para.dx && part[i].x < para.xmax - 2 * para.dx && part[i].free) {
        uint32_t ix = (uint32_t) ((part[i].x - para.xmin) / para.dx - 0.5);
        real_t exx = 0.5*(ex[ix+1]+ex[ix-1]-2*ex[ix]) * (part[i].x - para.xmin - ix * para.dx - 0.5 * para.dx) * (part[i].x - para.xmin - ix * para.dx - 0.5 * para.dx) / (para.dx*para.dx) + 0.5*(ex[ix + 1]-ex[ix-1]) * (part[i].x - para.xmin - ix * para.dx - 0.5 * para.dx) / para.dx+ex[ix];
        real_t bzz = 0.5*(bz[ix+1]+bz[ix-1]-2*bz[ix]) * (part[i].x - para.xmin - ix * para.dx - 0.5 * para.dx) * (part[i].x - para.xmin - ix * para.dx - 0.5 * para.dx) / (para.dx*para.dx) + 0.5*(bz[ix + 1]-bz[ix-1]) * (part[i].x - para.xmin - ix * para.dx - 0.5 * para.dx) / para.dx+bz[ix];
        ix = (uint32_t) ((part[i].x - para.xmin) / para.dx);
        real_t eyy = 0.5*(ey[ix+1]+ey[ix-1]-2*ey[ix]) * (part[i].x - para.xmin - ix * para.dx) * (part[i].x - para.xmin - ix * para.dx) / (para.dx*para.dx) + 0.5*(ey[ix + 1]-ey[ix-1]) * (part[i].x - para.xmin - ix * para.dx) / para.dx+ey[ix];
        real_t t, s, pxm, pym, pxp, pyp, pxq, pyq;
        pxm = part[i].px - (exx) * PI * para.dx / 2;
        pym = part[i].py - (eyy) * PI * para.dx / 2;
        t = - (bzz) * PI * para.dx / (2 * sqrt(1 + pxm * pxm + pym * pym + part[i].pz * part[i].pz));
        s = 2 * t / (1 + t * t);
        pxq = pxm + pym * t;
        pyq = pym - pxm * t;
        pyp = pym - pxq * s;
        pxp = pxm + pyq * s;
        part[i].px = pxp - (exx) * PI * para.dx / 2;
        part[i].py = pyp - (eyy) * PI * para.dx / 2;
        part[i].gamma = sqrt(1 + part[i].px * part[i].px + part[i].py * part[i].py + part[i].pz * part[i].pz);

        real_t xnew = part[i].x + part[i].px * para.dx / (2 * part[i].gamma);
        uint32_t ixnew = (uint32_t) ((xnew - para.xmin) / para.dx);

        for(int is = -1; is < 3; is++) {
            real_t Sx = 0.5 - abs(para.xmin + (ixnew + is) * para.dx - xnew) / (4 * para.dx); 
            atomicAdd(np + ixnew + is, part[i].realpart * Sx);
            atomicAdd(jy + ixnew + is, 0.0 - part[i].realpart * Sx * part[i].py / part[i].gamma);
        }
        ixnew = (uint32_t) ((xnew - para.xmin) / para.dx - 0.5);
        for(int is = -1; is < 3; is++) {
            real_t Sx = 0.5 - abs(para.xmin + (ixnew + is + 0.5) * para.dx - xnew) / (4 * para.dx); 
            atomicAdd(jx + ixnew + is, 0.0 - part[i].realpart * Sx * part[i].px / part[i].gamma);
        }
        /*
        atomicAdd(np + ixnew, part[i].realpart * (para.xmin + (ixnew + 1) * para.dx - xnew) / para.dx);
        atomicAdd(np + ixnew + 1, part[i].realpart * (xnew - para.xmin - ixnew * para.dx) / para.dx);
        if(ix == ixnew) {
            atomicAdd(jx + ix, 0.0 - part[i].realpart * (xnew - part[i].x) / (0.5 * para.dx));
            atomicAdd(jy + ix, 0.0 - part[i].realpart * part[i].py * (para.xmin + (ix + 1) * para.dx - (xnew + part[i].x) * 0.5) / (para.dx * part[i].gamma));
            atomicAdd(jy + ix + 1, 0.0 - part[i].realpart * part[i].py * ((xnew + part[i].x) * 0.5 - para.xmin - ix * para.dx) / (para.dx * part[i].gamma)); 
        }
        else if(ix > ixnew) {
            atomicAdd(jx + ix, 0.0 - part[i].realpart * (para.xmin + ix * para.dx - part[i].x) / (0.5 * para.dx));
            atomicAdd(jx + ixnew, 0.0 - part[i].realpart * (xnew - para.xmin - ix * para.dx) / (0.5 * para.dx));
            ixnew = (uint32_t) (((xnew + part[i].x) * 0.5 - para.xmin) / para.dx);
            atomicAdd(jy + ixnew, 0.0 - part[i].realpart * part[i].py * (para.xmin + (ixnew + 1) * para.dx - (xnew + part[i].x) * 0.5) / (para.dx * part[i].gamma));
            atomicAdd(jy + ixnew + 1, 0.0 - part[i].realpart * part[i].py * ((xnew + part[i].x) * 0.5 - para.xmin - ixnew * para.dx) / (para.dx * part[i].gamma));
        }
        else {
            atomicAdd(jx + ix, 0.0 - part[i].realpart * (para.xmin + ixnew * para.dx - part[i].x) / (0.5 * para.dx));
            atomicAdd(jx + ixnew, 0.0 - part[i].realpart * (xnew - para.xmin - ixnew * para.dx) / (0.5 * para.dx));
            ixnew = (uint32_t) (((xnew + part[i].x) * 0.5 - para.xmin) / para.dx);
            atomicAdd(jy + ixnew, 0.0 - part[i].realpart * part[i].py * (para.xmin + (ixnew + 1) * para.dx - (xnew + part[i].x) * 0.5) / (para.dx * part[i].gamma));
            atomicAdd(jy + ixnew + 1, 0.0 - part[i].realpart * part[i].py * ((xnew + part[i].x) * 0.5 - para.xmin - ixnew * para.dx) / (para.dx * part[i].gamma));
        }*/
        part[i].x = xnew;
    }
}

void saveData(uint32_t index, real_t *ey, real_t *bz, real_t *ex, real_t *np, real_t *jx, real_t *jy, Particle *particles, Parameters para) {
    real_t *pdata;
    
    pdata = (real_t *) malloc(para.total_part_num * 5 * sizeof(real_t));

    for(uint32_t i = 0; i < para.total_part_num; i++) {
        pdata[i*5+0] = particles[i].x;
        pdata[i*5+1] = particles[i].px;
        pdata[i*5+2] = particles[i].py;
        pdata[i*5+3] = particles[i].pz;
        pdata[i*5+4] = particles[i].gamma;
    }
    
    char filename[20], dataname[20];
    hid_t file_id, dataspace, dset_id;
    hsize_t dimsf[1];
    sprintf(filename, "data_%d.h5", index);
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    dimsf[0] = para.ngrid;
    dataspace = H5Screate_simple(1, dimsf, NULL);
    sprintf(dataname, "density");
    dset_id = H5Dcreate(file_id, dataname, H5_REAL, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, np);
    H5Dclose(dset_id);

    sprintf(dataname, "ey");
    dset_id = H5Dcreate(file_id, dataname, H5_REAL, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, ey);
    H5Dclose(dset_id);

    sprintf(dataname, "bz");
    dset_id = H5Dcreate(file_id, dataname, H5_REAL, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, bz);
    H5Dclose(dset_id);

    sprintf(dataname, "ex");
    dset_id = H5Dcreate(file_id, dataname, H5_REAL, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, ex);
    H5Dclose(dset_id);

    sprintf(dataname, "jx");
    dset_id = H5Dcreate(file_id, dataname, H5_REAL, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, jx);
    H5Dclose(dset_id);

    sprintf(dataname, "jy");
    dset_id = H5Dcreate(file_id, dataname, H5_REAL, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, jy);
    H5Dclose(dset_id);
    H5Sclose(dataspace);

    dimsf[0] = para.total_part_num*5;
    dataspace = H5Screate_simple(1, dimsf, NULL);
    sprintf(dataname, "particle");
    dset_id = H5Dcreate(file_id, dataname, H5_REAL, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, pdata);
    H5Dclose(dset_id);
    H5Sclose(dataspace);
    H5Fclose(file_id); 

    printf("Saved data, file index is %d.\n", index);
}

__global__ void setup_random_kernel(curandState *state)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init(1234, i, 0, &state[threadIdx.x]);
}

// Advance Ey and Bz using CUDA.
cudaError_t advanceWithCuda(real_t *ey, real_t *bz, real_t *ex,
                            real_t *jx, real_t *jy, real_t *np,
                            Particle *particles, Parameters para) {
    
    real_t *dev_ey, *dev_bz, *dev_ex;
    real_t *dev_jx, *dev_jy, *dev_np;
    uint32_t file_index;
    curandState *devStates;
    Particle *dev_particles;
    cudaDeviceProp deviceProp;
    cudaAssert( cudaGetDeviceProperties(&deviceProp, 0) );
    
    
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaAssert( cudaSetDevice(0) );
    
    // Allocate GPU buffers   .
    cudaAssert( cudaMalloc((void**)&dev_ey, para.ngrid * sizeof(real_t)) );
    
    cudaAssert( cudaMalloc((void**)&dev_bz, para.ngrid * sizeof(real_t)) );
    
    cudaAssert( cudaMalloc((void**)&dev_ex, para.ngrid * sizeof(real_t)) );
    
    cudaAssert( cudaMalloc((void**)&dev_jx, para.ngrid * sizeof(real_t)) );
    
    cudaAssert( cudaMalloc((void**)&dev_jy, para.ngrid * sizeof(real_t)) );
    
    cudaAssert( cudaMalloc((void**)&dev_np, para.ngrid * sizeof(real_t)) );
    
    cudaAssert( cudaMalloc((void**)&dev_particles, para.total_part_num * sizeof(Particle)) );

    cudaAssert( cudaMalloc((void**)&devStates, 256 * sizeof(curandState)) );
    
    setup_random_kernel<<<1, 256>>>(devStates);
    initializeFieldsKernel<<<(para.ngrid + 255)/256, 256>>>(dev_ey, dev_bz, dev_ex, para);
    cudaAssert( cudaGetLastError() );
    cudaAssert( cudaDeviceSynchronize() );
    
    initializeParticlesKernel<<<(para.total_part_num + 255)/256, 256>>>(dev_particles, para, devStates);
    cudaAssert( cudaGetLastError() );
    cudaAssert( cudaDeviceSynchronize() );

    cudaAssert( cudaMemcpy(ey, dev_ey, para.ngrid * sizeof(real_t), cudaMemcpyDeviceToHost) );
    cudaAssert( cudaMemcpy(bz, dev_bz, para.ngrid * sizeof(real_t), cudaMemcpyDeviceToHost) );
    cudaAssert( cudaMemcpy(ex, dev_ex, para.ngrid * sizeof(real_t), cudaMemcpyDeviceToHost) );
    cudaAssert( cudaMemcpy(np, dev_np, para.ngrid * sizeof(real_t), cudaMemcpyDeviceToHost) );
    cudaAssert( cudaMemcpy(jx, dev_jx, para.ngrid * sizeof(real_t), cudaMemcpyDeviceToHost) );
    cudaAssert( cudaMemcpy(jy, dev_jy, para.ngrid * sizeof(real_t), cudaMemcpyDeviceToHost) );
    cudaAssert( cudaMemcpy((void *)particles, (void *)dev_particles, para.total_part_num * sizeof(Particle), cudaMemcpyDeviceToHost) );
    
    file_index = 0;
    saveData(file_index, ey, bz, ex, np, jx, jy, particles, para);
    file_index++;
    
    for(real_t x = 0; x < para.sim_len; x += para.dx / 2) {
        // Launch a kernel on the GPU with one thread for each element.
        advanceBzKernel<<<(para.ngrid+254)/256, 256>>>(dev_ey, dev_bz, para);
        cudaAssert( cudaGetLastError() );
        cudaAssert( cudaDeviceSynchronize() );
        
        resetCurrentsKernel<<<(para.ngrid+255)/256, 256>>>(dev_jx, dev_jy, dev_np, para);
        cudaAssert( cudaGetLastError() );
        cudaAssert( cudaDeviceSynchronize() );
        
        advanceParticlesKernel<<<(para.total_part_num + 255)/256, 256>>>(dev_ey, dev_bz, dev_ex, dev_jx, dev_jy, dev_np, dev_particles, para, devStates);
        cudaAssert( cudaGetLastError() );
        cudaAssert( cudaDeviceSynchronize() );
        
        advanceBzKernel<<<(para.ngrid+254)/256, 256>>>(dev_ey, dev_bz, para);
        cudaAssert( cudaGetLastError() );
        cudaAssert( cudaDeviceSynchronize() );
        
        advanceExyKernel<<<(para.ngrid+254)/256, 256>>>(dev_ey, dev_bz, dev_ex, dev_jx, dev_jy, para);
        cudaAssert( cudaGetLastError() );
        cudaAssert( cudaDeviceSynchronize() );

        if (x >= file_index * 10 && x < file_index * 10 + para.dx / 2) {
            cudaAssert( cudaMemcpy(ey, dev_ey, para.ngrid * sizeof(real_t), cudaMemcpyDeviceToHost) );
            cudaAssert( cudaMemcpy(bz, dev_bz, para.ngrid * sizeof(real_t), cudaMemcpyDeviceToHost) );
            cudaAssert( cudaMemcpy(ex, dev_ex, para.ngrid * sizeof(real_t), cudaMemcpyDeviceToHost) );
            cudaAssert( cudaMemcpy(np, dev_np, para.ngrid * sizeof(real_t), cudaMemcpyDeviceToHost) );
            cudaAssert( cudaMemcpy(jx, dev_jx, para.ngrid * sizeof(real_t), cudaMemcpyDeviceToHost) );
            cudaAssert( cudaMemcpy(jy, dev_jy, para.ngrid * sizeof(real_t), cudaMemcpyDeviceToHost) );
            cudaAssert( cudaMemcpy((void *)particles, (void *)dev_particles, para.total_part_num * sizeof(Particle), cudaMemcpyDeviceToHost) );
            
            saveData(file_index, ey, bz, ex, np, jx, jy, particles, para);
            file_index++;
        }
    }
    cudaAssert( cudaFree(dev_ey) );
    cudaAssert( cudaFree(dev_bz) );
    cudaAssert( cudaFree(dev_ex) );
    cudaAssert( cudaFree(dev_jx) );
    cudaAssert( cudaFree(dev_jy) );
    cudaAssert( cudaFree(dev_np) );
    cudaAssert( cudaFree(dev_particles) );

    return cudaSuccess;
}

void getParameters(Parameters *para) {
    uint32_t ok=0;
    para->xmin = 0;
    while (ok == 0) {
        printf("Please input\nlaser a0 : ");
        scanf(FLAG, &para->a0);
        printf("number of laser pulses: ");
        scanf("%u", &para->n_pulse);
        printf("laser wavelength in nanometer: ");
        scanf(FLAG, &para->lambda);
        printf("laser duration in laser period : ");
        scanf(FLAG, &para->tau);
        printf("interval of laser pulses in laser period : ");
        scanf(FLAG, &para->t_interval);
        printf("box length in wavelength : ");
        scanf(FLAG, &para->xmax);
        printf("number of grids : ");
        scanf("%u", &para->ngrid);
        printf("simulation length in wavelength: ");
        scanf(FLAG, &para->sim_len);
        printf("plasma rises from : ");
        scanf(FLAG, &para->x_rise);
        printf("plasma is uniform from : ");
        scanf(FLAG, &para->x_uni);
        printf("plasma falls from : ");
        scanf(FLAG, &para->x_fall);
        printf("plasma ends from : ");
        scanf(FLAG, &para->x_end);
        printf("plasma uniform density in 10^18 per c.c. : ");
        scanf(FLAG, &para->n_plasma);
        printf("number of particles per cell : ");
        scanf("%u", &para->cell_part_num);
        printf("photoionization cross section in Mb : ");
        scanf(FLAG, &para->pi_cs);
        printf("orbital energy in eV : ");
        scanf(FLAG, &para->orb_en);
	    printf("Is setup OK ? 0 No 1 Yes : ");
	    scanf("%u", &ok);
    }
    para->dx = (para->xmax - para->xmin) / para->ngrid;
    para->total_part_num = para->cell_part_num * para->ngrid * (para->x_end - para->x_rise) / (para->xmax - para->xmin);
    para->n_plasma *= para->lambda * para->lambda / 1.1e9;
}

int main()
{
    Parameters para;
    getParameters(&para);

    real_t *ey, *bz, *ex;
    real_t *jx, *jy, *np;
    
    ey = (real_t *) malloc(para.ngrid * sizeof(real_t));
	bz = (real_t *) malloc(para.ngrid * sizeof(real_t));
    ex = (real_t *) malloc(para.ngrid * sizeof(real_t));
    jy = (real_t *) malloc(para.ngrid * sizeof(real_t));
	jx = (real_t *) malloc(para.ngrid * sizeof(real_t));
    np = (real_t *) malloc(para.ngrid * sizeof(real_t));

    Particle *particles;
    particles = (Particle *) malloc(para.total_part_num * sizeof(Particle));
    
    // Advance fields in parallel.
    cudaAssert( advanceWithCuda(ey, bz, ex, jx, jy, np, particles, para) );
	
    free(particles);
    free(ey);
    free(ex);
    free(bz);
    free(jy);
    free(jx);
    free(np);
// cudaDeviceReset must be called before exiting in order for profiling and
// tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaAssert( cudaDeviceReset() );
    
    printf("Simulation complete\n");

    return 0;
}
