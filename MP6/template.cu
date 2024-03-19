#include <wb.h>

#define BLOCK_SIZE 1024

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//Each thread handle 2 output
__global__ void accum_sum(float *arr, float *blockSums, int len) {
  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x;
  if(blockIdx.x >0){
    if(start + t < len){
      arr[start + t] += blockSums[blockIdx.x-1];
    }
    if(start + t + blockDim.x < len){
      arr[start + t + blockDim.x] += blockSums[blockIdx.x-1];
    }
  }
}
//Use Brent-Kung; Each thread handle 2 output
__global__ void scan(float *input, float *output, float *aux, int len) {
  __shared__ float T[2*BLOCK_SIZE];
  
  //copy data into shared mem
  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x;
  if(start + t < len){
    T[t] = input[start + t];
  }else{
    T[t] = 0.0f;//filling 0
  }
  if(start + t + blockDim.x < len){
    T[t+blockDim.x] = input[start + t+blockDim.x];
  }else{
    T[t+blockDim.x] = 0.0f;
  }

  //restuction step
  int stride = 1;
  while(stride < (2*BLOCK_SIZE)){
    __syncthreads();
    int index = (t+1)*stride*2-1;
    if(index < (2*BLOCK_SIZE) && (index-stride)>=0)
    {
      T[index]+=T[index-stride];
    }
    stride *=2;
  }
  
  //post scan step
  stride = BLOCK_SIZE/2;
  while(stride > 0){
    __syncthreads();
    int index = (t+1)*stride*2-1;
    if((index+stride)<(2*BLOCK_SIZE)){
      T[index+stride]+=T[index];
    }
    stride /=2;
  }

  //copy partial result to output in global mem
  __syncthreads();
  if(start + t < len){
    output[start + t] = T[t];
  }
  if(start + t+blockDim.x < len){
    output[start + t + blockDim.x] = T[t + blockDim.x];
  }

  //copy result to auxiliary array in global mem
  __syncthreads();
  if(aux){
    if(threadIdx.x == blockDim.x-1){
      aux[blockIdx.x] = T[(2*BLOCK_SIZE)-1];
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  
  int numAuxArrElement = ceil(numElements/(2.0*BLOCK_SIZE));
  float *deviceAux;
  cudaMalloc((void **)&deviceAux, numAuxArrElement * sizeof(float));

  wbTime_start(Compute, "Performing CUDA computation");

  // store block sum to auxiliary array and also save partial results to deviceOutput
  dim3 DimBlock1(BLOCK_SIZE, 1, 1);
  dim3 DimGrid1(ceil(numElements/(2.0*BLOCK_SIZE)), 1, 1);
  scan<<<DimGrid1, DimBlock1>>>(deviceInput,deviceOutput, deviceAux, numElements);
  cudaDeviceSynchronize();

  //scan block sums
  dim3 DimBlock2(ceil(numElements/(2.0*BLOCK_SIZE))/2, 1, 1);
  dim3 DimGrid2(1, 1, 1);
  scan<<<DimGrid2, DimBlock2>>>(deviceAux, deviceAux, NULL, numAuxArrElement);
  cudaDeviceSynchronize();

  //add scanned block sums to deviceOutput
  dim3 DimGrid3(ceil(numElements/(2.0*BLOCK_SIZE)), 1, 1);
  dim3 DimBlock3(BLOCK_SIZE, 1, 1);
  accum_sum<<<DimGrid3, DimBlock3>>>(deviceOutput, deviceAux, numElements);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * numElements, cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAux);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
