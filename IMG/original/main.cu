#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;


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

    // Start time
    double start = static_cast<double>(getTickCount());
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Vec3b color = color_image.at<Vec3b>(y, x);
            int gray = (color[0] + color[1] + color[2]) / 3;
            gray_image.at<unsigned char>(y, x) = gray;
        }
    }

    // End time
    double end = static_cast<double>(getTickCount());

    // Time elapsed in ms
    double time = (end - start) * 1000 / getTickFrequency();
    printf("Time elapsed: %f ms\n", time); 

    imwrite("gray_image.jpg", gray_image);

    return 0;
}