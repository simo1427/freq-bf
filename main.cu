#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
//#include <assert.h>

__global__ void VecAdd(int* A, int* B, int* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {

    cv::VideoCapture cap = cv::VideoCapture(0);

    if (!cap.isOpened())
    {
        std::cerr << "Unable to open camera\n";
        return 1;
    }

    cv::Mat frame;
    while(true)
    {

        cap.read(frame);

        if (frame.empty()) {
            std::cerr << "Empty frame\n";
            break;
        }

        if (!frame.isContinuous()) {
            std::cerr << "Non-continuous frame\n";
            break;
        }

        // start filtering

        // end filtering

        cv::imshow("Live", frame);
        if(cv::waitKey(5)  == 27) // ESC
            break;

    }

    return 0;
}
