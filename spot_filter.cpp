#include "spot_filter.h"
#include <algorithm>
#include <cmath>

cv::Mat Spot_filter::despeckle_reference(cv::Mat& imgF, int kernel_size)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));

    cv::Mat eroded, opened, dilated, openClosed;

    // Opening: erode then dilate removes bright spikes smaller than the kernel.
    cv::erode(imgF, eroded, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
    cv::dilate(eroded, opened, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);

    // Closing the opened image: dilate then erode removes dark pits.
    cv::dilate(opened, dilated, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
    cv::erode(dilated, openClosed, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);

    return openClosed;
}

void Spot_filter::spot_correction(cv::Mat& img, int kernel_size, int threshold)
{
    CV_Assert(img.type() == CV_16UC1);

    cv::Mat imgF;
    img.convertTo(imgF, CV_32FC1);

    cv::Mat reference = despeckle_reference(imgF, kernel_size);

    for (int y = 0; y < img.rows; ++y) {
        uint16_t* row = img.ptr<uint16_t>(y);
        const float* ref = reference.ptr<float>(y);
        for (int x = 0; x < img.cols; ++x) {
            float diff = std::abs(static_cast<float>(row[x]) - ref[x]);
            if (diff > threshold)
                row[x] = static_cast<uint16_t>(std::clamp(ref[x], 0.0f, 65535.0f));
        }
    }
}
