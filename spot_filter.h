#ifndef SPOT_FILTER_H
#define SPOT_FILTER_H
#include <opencv2/opencv.hpp>

// White/dark spot (hot/dead pixel) correction for 16-bit grayscale images.
//
// The despeckled reference image is built with a morphological open-close
// (OpenCV erode/dilate): opening (erode then dilate) removes bright spikes
// ("white spots"), closing the result (dilate then erode) removes dark pits
// ("dark spots"). Pixels that deviate from that reference by more than
// `threshold` are replaced by it.
class Spot_filter
{
public:
    // `img` must be CV_16UC1 and is modified in place.
    static void spot_correction(cv::Mat& img, int kernel_size, int threshold);

private:
    static cv::Mat despeckle_reference(cv::Mat& imgF, int kernel_size);
};

#endif // SPOT_FILTER_H
