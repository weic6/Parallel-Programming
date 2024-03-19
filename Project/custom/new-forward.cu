// optimization: FP16; source: CUDA ToolKit Documentation
//
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#include <cuda_fp16.h> 

#define TILE_WIDTH 8

// __half **half_device_input_ptr;
// __half **half_device_output_ptr;
// __half **half_device_mask_ptr;

__half *half_device_input;
__half *half_device_output;
__half *half_device_mask;

__half* half_host_input;
__half* half_host_output;
__half* half_host_mask;

// implement your kernel code from Lecture 12
__global__ void conv_forward_kernel(__half *output, const __half *input, const __half *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
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

    // int W_grid = W_out / TILE_WIDTH;
    int W_grid = ceil(W_out*1.0 / TILE_WIDTH);
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int b = blockIdx.z;
    __half acc = 0;


    if (h< H_out && w <W_out){
        for (int c = 0; c < C; c++) { // sum over all input channels
            for (int p = 0; p < K; p++) // loop over KxK filter
                for (int q = 0; q < K; q++)
                acc += in_4d(b, c, h*S+p, w*S+q) * mask_4d(m, c, p, q);
        }
        out_4d(b, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

//allocates memory and copies data from host to device (Note: the device pointers given to you in this function are double pointers). 
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    //allocates device memory
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int input_size = B * C * H * W;
    const int output_size = B * M * H_out * W_out;
    const int mask_size = C * M * K * K;

    half_host_input = (__half *)malloc(input_size*sizeof(__half));
    half_host_output =(__half *)malloc(output_size*sizeof(__half));//!!!
    half_host_mask = (__half *)malloc(mask_size*sizeof(__half));

    //convert float to __half on host
    for (int i = 0; i < input_size; ++i) half_host_input[i] = __float2half(host_input[i]);
    for (int i = 0; i < mask_size; ++i) half_host_mask[i] = __float2half(host_mask[i]);
    
    cudaMalloc((void **)&half_device_input, input_size* sizeof(__half));
    cudaMalloc((void **)&half_device_output, output_size* sizeof(__half));
    cudaMalloc((void **)&half_device_mask, mask_size * sizeof(__half));

    //copies data from host to device
    cudaMemcpy(half_device_input, half_host_input, input_size * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(half_device_mask, half_host_mask, mask_size * sizeof(__half), cudaMemcpyHostToDevice);
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
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);//each block deals with 1 tile
    conv_forward_kernel<<<DimGrid, DimBlock>>>(half_device_output, half_device_input, half_device_mask, B, M, C, H, W, K, S);
}


// copies output back to host and free the device memory.
__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int output_size = B * M * H_out * W_out;

    cudaMemcpy(half_host_output, half_device_output, output_size * sizeof(__half), cudaMemcpyDeviceToHost);

    //convert __half to float on host and copy result to host_output 
    for (int i = 0; i < output_size; ++i) host_output[i] = __half2float(half_host_output[i]);

    // Free device memory
    cudaFree(half_device_input);
    cudaFree(half_device_output);
    cudaFree(half_device_mask);

    // Free host memory
    free(half_host_input);
    free(half_host_output);
    free(half_host_mask);
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