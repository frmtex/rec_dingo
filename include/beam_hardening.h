#ifndef BEAM_HARDENING_H
#define BEAM_HARDENING_H
#include <opencv2/opencv.hpp>

// Manual polynomial beam-hardening (cupping) correction applied to reconstructed
// attenuation values: mu_corrected = c1*mu + c2*mu^2 + c3*mu^3. Identity when
// c1=1, c2=0, c3=0. Coefficients are tuned by eye against the B/M/T preview
// slices - there's no phantom calibration here, just a flattening polynomial.
namespace BeamHardening
{
inline cv::Mat apply(const cv::Mat& slice, double c1, double c2, double c3)
{
    CV_Assert(slice.type() == CV_32FC1);
    cv::Mat mu2, mu3;
    cv::multiply(slice, slice, mu2);
    cv::multiply(mu2, slice, mu3);
    return c1 * slice + c2 * mu2 + c3 * mu3;
}
}

#endif // BEAM_HARDENING_H
