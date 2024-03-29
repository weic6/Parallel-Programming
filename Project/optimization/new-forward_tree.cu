// optimization: reduction: tree; source: Lec 15; dynamically array alloc: https://stackoverflow.com/questions/5531247/allocating-shared-memory%5B/url%5D

#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 8

#define cudaCheckErr(stmt) { \
    cudaError_t error = stmt; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(-1); \
    } \
}


// implement your kernel code from Lecture 12
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil(W_out*1.0 / TILE_WIDTH);
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int b = blockIdx.z;
    int c = threadIdx.z; //0<=c<=C-1

    extern __shared__ float prod[]; //declare dynamically sized shared memory
    // #define prod_3d(i2, i1, i0) prod[(i2) * (TILE_WIDTH * C) + (i1) * (C) + i0]
    // #define prod_3d(i2, i1, i0) prod[(i2) * (W_out * C) + (i1) * (C) + i0]
    #define prod_3d(i2, i1, i0) prod[(i2) * (TILE_WIDTH * C) + (i1) * (C) + i0]

    if (h< H_out && w <W_out){
        //prepare array
        float temp = 0.0f;
        for (int p = 0; p < K; p++){ // loop over KxK filter
            for (int q = 0; q < K; q++){
                temp += in_4d(b, c, h*S+p, w*S+q) * mask_4d(m, c, p, q);
                // atomicAdd(&out_4d(b, m, h, w), in_4d(b, c, h*S+p, w*S+q) * mask_4d(m, c, p, q));
            }
        }
        prod_3d(threadIdx.y,threadIdx.x,c) = temp;
        // prod_3d(h,w,c) = temp;

        // reduction on prod_3d(threadIdx.y, threadIdx.x, _)
        for(int stride = 1; stride < C; stride *=2){
            __syncthreads();
            if((c % (2*stride)==0) && (c+stride < C)){
                prod_3d(threadIdx.y,threadIdx.x,c) += prod_3d(threadIdx.y,threadIdx.x,c+stride);
                // prod_3d(h,w,c) += prod_3d(h,w,c+stride);
            }
        }
        __syncthreads();

        //choose one thread to copy
        if(c == 0){
            // out_4d(b, m, h, w) = 0;
            // for(int i = 0; i < C; i++){
            //     out_4d(b, m, h, w) += prod_3d(threadIdx.y,threadIdx.x,i);
            // }   
            out_4d(b, m, h, w) = prod_3d(threadIdx.y,threadIdx.x,0);
            // out_4d(b, m, h, w) = prod_3d(h,w,0);
        }

    }


    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	//allocates memory and copies data from host to device (Note: the device pointers given to you in this function are double pointers). 
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    //allocates device memory
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int input_size = B * C * H * W;
    const int output_size = B * M * H_out * W_out;
    const int mask_size = C * M * K * K;

    // cudaCheckErr(cudaMalloc((void **)device_input_ptr, input_size* sizeof(float)));
    cudaMalloc((void **)device_input_ptr, input_size* sizeof(float));
    // cudaCheckErr(cudaMalloc((void **)device_output_ptr, output_size* sizeof(float))
    cudaMalloc((void **)device_output_ptr, output_size* sizeof(float));
    // cudaCheckErr(cudaMalloc((void **)device_mask_ptr, mask_size * sizeof(float)));
    cudaMalloc((void **)device_mask_ptr, mask_size * sizeof(float));

    //copies data from host to device
    // cudaCheckErr(cudaMemcpy(*device_input_ptr, host_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaMemcpy(*device_input_ptr, host_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaCheckErr(cudaMemcpy(*device_mask_ptr, host_mask, mask_size* sizeof(float), cudaMemcpyHostToDevice));
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size* sizeof(float), cudaMemcpyHostToDevice);
}

//computes kernel dimensions and invokes kernel.
__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    //可能存在不能整除，所以得用ceil
    int W_grid = ceil(W_out*1.0/TILE_WIDTH);
    int H_grid = ceil(H_out*1.0/TILE_WIDTH);
    int Y = W_grid * H_grid;
    dim3 DimGrid(M, Y, B);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, C);
    int a_size = TILE_WIDTH * TILE_WIDTH * C * sizeof(float); //dynamic shared memory allocation
    conv_forward_kernel<<<DimGrid, DimBlock, a_size>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);

}


// copies output back to host and free the device memory.
__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int output_size = B * M * H_out * W_out;
    // cudaCheckErr(cudaMemcpy(host_output, device_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaMemcpy(host_output, device_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    // cudaCheckErr(cudaFree(device_input));
    cudaFree(device_input);
    // cudaCheckErr(cudaFree(device_output));
    cudaFree(device_output);
    // cudaCheckErr(cudaFree(device_mask));
    cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
