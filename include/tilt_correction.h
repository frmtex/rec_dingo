#ifndef TILT_CORRECTION_H
#define TILT_CORRECTION_H
#include <opencv2/opencv.hpp>

// Rotates a projection about its own center by angle_deg, keeping the
// original size (ports scipy.ndimage.rotate(..., reshape=False)).
class TiltCorrection
{
public:
    static void apply(cv::Mat& img, double angle_deg);
};

#endif // TILT_CORRECTION_H
