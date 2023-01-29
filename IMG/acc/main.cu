#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

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


    // Convert to gray with kernel
    uchar3 *d_color_image;
    unsigned char *d_gray_image;
    cudaMalloc(&d_color_image, width * height * sizeof(uchar3));
    cudaMalloc(&d_gray_image, width * height * sizeof(unsigned char));
    cudaMemcpy(d_color_image, color_image.ptr(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice);



    dim3 block(24, 24);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Start time
    double start = static_cast<double>(getTickCount());

    convertToGray<<<grid, block>>>(d_color_image, d_gray_image, width, height);
    cudaDeviceSynchronize();

    // End time
    double end = static_cast<double>(getTickCount());

    cudaMemcpy(gray_image.ptr(), d_gray_image, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_color_image);
    cudaFree(d_gray_image);



    // Time elapsed in ms
    double time = (end - start) * 1000 / getTickFrequency();
    printf("Time elapsed: %f ms\n", time); 



    imwrite("gray_image.jpg", gray_image);

    return 0;
}