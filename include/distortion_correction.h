#ifndef DISTORTION_CORRECTION_H
#define DISTORTION_CORRECTION_H
#include <opencv2/core.hpp>
#include <QString>
#include <memory>
#include <vector>

// Geometric distortion correction for the detector (camera + mirror + lens), measured from images
// of graph paper mounted at the scintillator position.
//
// The graph paper's crossings are detected to sub-pixel accuracy and indexed on a lattice. A model
//     measured position = affine(lattice index) + polynomial(lattice index)
// is fitted to them. The affine part absorbs aspect ratio, shear and the pixel scale; the
// polynomial absorbs everything smooth that is left - the keystone from a tilted mirror
// (perspective), lens distortion, etc. - without assuming a particular lens model, since the
// keystone and the lens distortion can't be told apart from one view and don't need to be.
//
// The correction maps an ideal square grid, with the grid's mean pixel size and its rotation as
// measured, onto the measured one. The rotation is deliberately NOT removed: it mostly says how
// the paper was laid down, not how the detector is built, and the rotation-axis tilt is measured
// from the scan itself (Find CoR/Tilt).
//
// Nothing here is specific to one camera or screen: the grid pitch (in pixels) is found from the
// images themselves, so any camera, lens or screen size works as long as the paper's squares are
// at least ~10 pixels wide and the paper covers the part of the image that matters (outside the
// paper the polynomial part is held at its edge value and only the affine part continues).
class DistortionCorrection
{
public:
    struct Options
    {
        // Degree of the polynomial in the lattice coordinates. 4 is a good default (higher degrees
        // fit the paper's own irregularities and start to extrapolate badly).
        int polyDegree = 4;
    };

    struct Report
    {
        int imageCount = 0;
        int crossings = 0;        // lattice points used for the fit
        int gridColumns = 0;
        int gridRows = 0;
        double pitchPx = 0.0;     // grid cell size in pixels
        double rotationDeg = 0.0; // rotation of the grid relative to the image axes (kept, not corrected)
        double rmsBeforePx = 0.0; // RMS deviation from a perfect square grid (rotation + scale only)
        double maxBeforePx = 0.0;
        double rmsAfterPx = 0.0;  // RMS deviation left after the fitted model
        double maxAfterPx = 0.0;
        QString summary() const;
    };

    // Fits the model to `images`, which must all be the same size and show the same grid; they
    // are averaged first. Any bit depth, single channel. Throws std::runtime_error with a
    // message meant for the user if no usable grid is found.
    static std::shared_ptr<const DistortionCorrection> fit(const std::vector<cv::Mat>& images,
                                                           const Options& options, Report* report = nullptr);

    // Reads every image in `dir` (TIFF/PNG/... through OpenCV, and FITS) and rotates (clockwise,
    // 0/90/180/270 degrees) and optionally flips (rows reversed, as for bottom-up FITS) each so
    // they match the scan's orientation. Throws if the folder is missing or has no readable images.
    static std::vector<cv::Mat> loadGridImages(const QString& dir, int rotateDeg, bool flipRows);

    // Size of the images the model was fitted on - the scan frames must be the same size.
    cv::Size frameSize() const { return frameSize_; }

    // Position, in the measured (scan) image, of the pixel at (x, y) of the corrected image.
    cv::Point2d sourcePosition(double x, double y) const;

    // The part of the scan frame needed to produce corrected pixels for `roi` (all in full-resolution
    // frame coordinates), grown for the distortion and interpolation, clamped to the frame and
    // aligned so it starts a whole number of `binning` blocks before `roi` - i.e. binning it gives
    // block boundaries that agree with binning `roi` alone.
    cv::Rect requiredSourceRect(const cv::Rect& roi, int binning) const;

    // Remap table (CV_32FC2) for cv::remap that turns the image of `srcRect` (as returned by
    // requiredSourceRect and binned by `binning`) into the corrected image of `roi` binned by
    // `binning`: output size (roi.width / binning) x (roi.height / binning).
    cv::Mat buildRemap(const cv::Rect& roi, const cv::Rect& srcRect, int binning) const;

private:
    DistortionCorrection() = default;

    cv::Size frameSize_;
    int degree_ = 4;
    double pitch_ = 1.0;             // lattice coordinate scale used in the polynomial basis
    double i0_ = 0.0, j0_ = 0.0;     // lattice centre used in the polynomial basis
    double iMin_ = 0, iMax_ = 0, jMin_ = 0, jMax_ = 0; // lattice extent that was measured
    cv::Matx<double, 3, 2> affine_;  // [i j 1] * affine_ = affine part of the measured position
    cv::Mat poly_;                   // (terms x 2) polynomial coefficients
    // Output pixel position w = simA_ * (i + j*I) + simB_ (complex numbers: x + y*I) - the ideal
    // square grid placed with the measured mean scale and rotation.
    double simAr_ = 1, simAi_ = 0, simBx_ = 0, simBy_ = 0;
};

#endif // DISTORTION_CORRECTION_H
