#ifndef ROTATION_AXIS_H
#define ROTATION_AXIS_H
#include <opencv2/opencv.hpp>
#include <vector>

// Estimates the rotation-axis offset and tilt from a 0deg/180deg projection
// pair, by finding the horizontal shift that best aligns proj0 with the
// horizontally-flipped proj180, sampled at several rows and fit with a line.
// Ports find_COR / find_rotation_centre from the Python reconstruction pipeline.
struct CorResult
{
    double offset;    // rotation-axis offset from the image's vertical center line, in pixels
    double tilt_deg;  // tilt of the rotation axis relative to the detector's vertical axis
};

class RotationAxis
{
public:
    // proj0, proj180: CV_32FC1 flat-fielded projections, same size.
    // rois: user-selected row ranges (rect.y .. rect.y+rect.height) where the sample is visible.
    // ystep: rows are sampled every `ystep` pixels within each ROI.
    static CorResult find_axis(const cv::Mat& proj0, const cv::Mat& proj180,
                                const std::vector<cv::Rect>& rois, int ystep = 5);

    // Convenience wrapper returning the absolute rotation-axis column coordinate
    // (detector_width/2 - offset), matching find_rotation_centre()'s return value.
    static double find_centre(const cv::Mat& proj0, const cv::Mat& proj180,
                               const std::vector<cv::Rect>& rois, int ystep, double& tilt_deg_out);
};

#endif // ROTATION_AXIS_H
