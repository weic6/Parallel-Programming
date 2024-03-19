#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16

//float -> unsigned char
//!!! size = width * height * channels
__global__ void cast(unsigned char* output, float* intput, int size){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i<size){
    output[i] = (unsigned char) (255 * intput[i]);
  }
}

//rgb -> grayscale
//!!! size = width * height (L3 slide 13)
__global__ void convert(unsigned char * output,  unsigned char * input, int size) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i<size){
    unsigned char r = input[3 * i];
    unsigned char g = input[3 * i+1];
    unsigned char b = input[3 * i+2];
    output[i] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
  }
}

// Kernel 3: grayscale -> histogram
//!!! must have at least 256 threads in a block!!
__global__ void computeHistogram(int * output, unsigned char * input, long size){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

  if(threadIdx.x < HISTOGRAM_LENGTH){
    histo_private[threadIdx.x] = 0;
  }
  __syncthreads();

  // int stride = blockDim.x * gridDim.x;
  // while(i<size){
  //   int pos = input[i];
  //   atomicAdd(&(histo_private[pos]),1);
  //   i+=stride;
  // }
  if (i < size) {
    int pos = input[i];
    atomicAdd(&(histo_private[pos]), 1);
  }
  __syncthreads();

  if(threadIdx.x < HISTOGRAM_LENGTH){
    atomicAdd(&(output[threadIdx.x]), histo_private[threadIdx.x]);
  }
}

// Kernel 4: histogram -> cdf
//!!! only need 128 threads (histo size is only 256)
//!!! len = 256; size = width * height: pic's weight and height
__global__ void scan(float *output, int *input, int len, int size) {
  __shared__ float T[HISTOGRAM_LENGTH];
  
  //copy partial array to shared mem in each block
  int t = threadIdx.x;
  int start = 2*blockIdx.x*blockDim.x;
  if(start + t < len){
    T[t] = (input[start + t]*1.0)/size;
  }else{
    T[t] = 0.0f/size;
  }
  if(start + t+blockDim.x < len){
    T[t+blockDim.x] = (input[start + t+blockDim.x]*1.0)/size;
  }else{
    T[t+blockDim.x] = 0.0f/size;
  }

  //restuction step
  int stride = 1;
  while(stride < 2*128){
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2-1;
    if(index < 2*128 && (index-stride)>=0){
      T[index]+=T[index-stride];
    }
    stride *=2;
  }
  //post scan step
  stride = 128/2;
  while(stride > 0){
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2-1;
    if((index+stride)<2*128){
      T[index+stride]+=T[index];
    }
    stride /=2;
  }
  __syncthreads();

  //copy T to output
  if(start + t < len){
    output[start+t] = T[t];
  }
  if(start + t+blockDim.x < len){
    output[start+blockDim.x+t] = T[t+blockDim.x];
  }
}

// Kernel 5: correct color
__global__ void correctColor(unsigned char *output, float *input, int size) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i<size){
    float temp = 255 * (input[output[i]] - input[0]) / (1.0 - input[0]);
    output[i] = (unsigned char)min(max(temp,0.0),255.0);
  }
}

// Kernel 6: cast back to float
__global__ void castBack(float *output, unsigned char *input, int size) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i<size){
    output[i] = (float) (input[i]/255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceImageData;
  unsigned char* deviceGrayImage;
  int* deviceHistogram;
  float* deviceCdf;
  unsigned char* deviceUncharImage;
  const char *inputImageFile;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  hostInputImageData = wbImage_getData(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");
  hostOutputImageData = wbImage_getData(outputImage);

  int imagePixelLs = imageWidth * imageHeight;
  int imageTotalSize = imageWidth* imageHeight * imageChannels;

  cudaMalloc((void **)&deviceImageData, imageTotalSize * sizeof(float));
  cudaMalloc((void **)&deviceUncharImage, imageTotalSize * sizeof(unsigned char));
  cudaMalloc((void **)&deviceGrayImage, imagePixelLs * sizeof(unsigned char));
  cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(int));
  cudaMalloc((void **)&deviceCdf, HISTOGRAM_LENGTH * sizeof(float));

  cudaMemcpy(deviceImageData, hostInputImageData, imageTotalSize * sizeof(float), cudaMemcpyHostToDevice);

  //convert float to unsign char
  dim3 DimBlock1(BLOCK_SIZE, 1, 1);
  dim3 DimGrid1(ceil(imageTotalSize*1.0/BLOCK_SIZE), 1, 1);
  cast<<<DimGrid1, DimBlock1>>>(deviceUncharImage, deviceImageData, imageTotalSize);
  cudaDeviceSynchronize();

  //transform unsigned char to gray
  dim3 DimBlock2(BLOCK_SIZE, 1, 1);
  dim3 DimGrid2(ceil(imagePixelLs*1.0/BLOCK_SIZE), 1, 1);
  convert<<<DimGrid2, DimBlock2>>>(deviceGrayImage, deviceUncharImage, imagePixelLs);
  cudaDeviceSynchronize();

  //create histogram on gray
  dim3 DimBlock3(256, 1, 1);//must have at least 256 threads in a block!!
  dim3 DimGrid3(ceil(imagePixelLs*1.0/256), 1, 1);
  computeHistogram<<<DimGrid3, DimBlock3>>>(deviceHistogram, deviceGrayImage, imagePixelLs);
  cudaDeviceSynchronize();

  //create cdf on histogram (use scan)
  dim3 DimBlock4(128, 1, 1);//one block handle 2*128 = 256
  dim3 DimGrid4(1, 1, 1); 
  scan<<<DimGrid4, DimBlock4>>>(deviceCdf, deviceHistogram, HISTOGRAM_LENGTH, imagePixelLs);
  cudaDeviceSynchronize();

  //correct color
  dim3 DimBlock5(BLOCK_SIZE, 1, 1);
  dim3 DimGrid5(ceil(imageTotalSize*1.0/BLOCK_SIZE), 1, 1); //one block handle 2*128 = 256
  correctColor<<<DimGrid5, DimBlock5>>>(deviceUncharImage, deviceCdf, imageTotalSize);
  cudaDeviceSynchronize();

  //cast uchar to float
  dim3 DimBlock6(BLOCK_SIZE, 1, 1);
  dim3 DimGrid6(ceil(imageTotalSize*1.0/BLOCK_SIZE), 1, 1);
  castBack<<<DimGrid6, DimBlock6>>>(deviceImageData, deviceUncharImage, imageTotalSize);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceImageData, imageTotalSize * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  cudaFree(deviceImageData);
  cudaFree(deviceUncharImage);
  cudaFree(deviceGrayImage);
  cudaFree(deviceHistogram);
  cudaFree(deviceCdf);
  return 0;
}
