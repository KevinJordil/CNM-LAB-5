#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

#define BLOCK_SIZE 16

/* Cuda check error */
void cuda_check_error(cudaError err)
{
  if (err != cudaSuccess)
  {
    printf("CUDA error (%d): %s  \n", err, cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void convertToGray(uchar3 *input, unsigned char *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        output[y * width + x] = (input[y * width + x].x + input[y * width + x].y + input[y * width + x].z) / 3;
    }
}


int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Usage: " << argv[0] << " <input_image>" << endl;
        return 1;
    }

    Mat color_image;
    color_image = imread(argv[1], IMREAD_COLOR);
    if (color_image.empty())
    {
        cout << "Failed to load image file: " << argv[1] << endl;
        return 1;
    }

    int width = color_image.cols;
    int height = color_image.rows;
    Mat gray_image(height, width, CV_8UC1);


    uchar3 *d_color_image;
    unsigned char *d_gray_image;

    // Allocate memory on device
    cuda_check_error(cudaMalloc(&d_color_image, width * height * sizeof(uchar3)));
    cuda_check_error(cudaMalloc(&d_gray_image, width * height * sizeof(unsigned char)));

    // Copy data from host to device
    cuda_check_error(cudaMemcpy(d_color_image, color_image.ptr(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice));


    // Threads per block
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Start time
    double start = static_cast<double>(getTickCount());

    convertToGray<<<grid, block>>>(d_color_image, d_gray_image, width, height);

    // Check for errors on kernel launch
    cuda_check_error(cudaGetLastError());

    // Synchronize threads
    cuda_check_error(cudaDeviceSynchronize());

    // End time
    double end = static_cast<double>(getTickCount());

    // Copy data from device to host
    cudaMemcpy(gray_image.ptr(), d_gray_image, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_color_image);
    cudaFree(d_gray_image);

    // Time elapsed in ms
    double time = (end - start) * 1000 / getTickFrequency();
    printf("Time elapsed: %f ms\n", time); 

    // Save image
    imwrite("gray_image.jpg", gray_image);

    return 0;
}