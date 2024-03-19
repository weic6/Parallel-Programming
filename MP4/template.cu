#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 4
//@@ Define constant memory for device kernel here
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int x_o = blockIdx.x * TILE_WIDTH + tx;
  int y_o = blockIdx.y * TILE_WIDTH + ty;
  int z_o = blockIdx.z * TILE_WIDTH + tz;
  int x_i = x_o - MASK_WIDTH/2;
  int y_i = y_o - MASK_WIDTH/2;
  int z_i = z_o - MASK_WIDTH/2;

  __shared__ float tile[TILE_WIDTH+ MASK_WIDTH-1][TILE_WIDTH+ MASK_WIDTH-1][TILE_WIDTH+ MASK_WIDTH-1];
  if ((x_i >=0) && (x_i < x_size ) && 
      (y_i >=0) && (y_i < y_size )&& 
      (z_i >=0) && (z_i < z_size )){
    tile[tx][ty][tz] = input[ x_i *y_size*z_size + y_i*z_size + z_i ];
  }else{
    tile[tx][ty][tz] = 0;
  }
  __syncthreads();
  
  float Pvalue = 0;
  if(tx<TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH){
    for(int i = 0; i < MASK_WIDTH; i++){
      for (int j = 0; j < MASK_WIDTH; j++){
        for (int k = 0; k < MASK_WIDTH; k++){
          Pvalue +=  Mc[i][j][k]*tile[tx+i][ty+j][tz+k];
        }
    }
    if(x_o < x_size && y_o < y_size && z_o < z_size){
      output[x_o * y_size * z_size + y_o * z_size + z_o] = Pvalue;
    }
    }
  }
}
int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **)&deviceInput, z_size * y_size * x_size * sizeof(float));
  cudaMalloc((void **)&deviceOutput, z_size * y_size * x_size * sizeof(float));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, z_size * y_size * x_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength*sizeof(float));

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(1.0*x_size / TILE_WIDTH), ceil(1.0* y_size / TILE_WIDTH), ceil(1.0*z_size / TILE_WIDTH));
  dim3 DimBlock(TILE_WIDTH + MASK_WIDTH-1, TILE_WIDTH + MASK_WIDTH-1, TILE_WIDTH + MASK_WIDTH-1);


  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput,z_size, y_size, x_size );

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutput +3 , deviceOutput, z_size*y_size*x_size * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;   
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
