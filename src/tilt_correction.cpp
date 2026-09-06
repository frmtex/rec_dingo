#include "tilt_correction.h"

void TiltCorrection::apply(cv::Mat& img, double angle_deg)
{
    if (angle_deg == 0.0)
        return;

    cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle_deg, 1.0);

    cv::Mat rotated;
    cv::warpAffine(img, rotated, rotMat, img.size(), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    img = rotated;
}
