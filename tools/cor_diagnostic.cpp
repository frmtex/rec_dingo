// Standalone diagnostic for the "star artifact centered on the rotation axis" symptom: reuses the
// already-verified SinogramReader/FbpReconstructor classes (no new reconstruction math) to check
// two of the most common causes:
//
//   1. A detector-column defect (stuck/miscalibrated pixel, dust) sitting exactly at the rotation
//      axis. Every projection angle reads that same detector column for a ray through the object
//      center, so such a defect shows up as a spike in the column-wise mean across all angles -
//      and it's exactly the case wavelet/Fourier ring-removal filters are least able to correct,
//      since a "ring" of zero radius carries no signal for them to act on.
//   2. Sub-pixel center-of-rotation (COR) misalignment. Sweeps a range of COR offsets and scores
//      each reconstruction by a focus/sharpness metric (variance of the Laplacian, a standard
//      autofocus measure) - misalignment blurs edges, so the sharpest reconstruction in the sweep
//      is the best COR estimate. Also writes each offset's slice as a PNG for visual inspection.
//
// Usage:
//   cor_diagnostic <sino_dir> <n_cols> <row_index> <n_angles> <rot_start_deg> <last_angle_deg>
//                  <out_dir> [cor_min=-3] [cor_max=3] [cor_step=0.25]
//
// <row_index> should be a row that shows the star artifact (e.g. the same slice you noticed it
// on). <sino_dir> is the sino/ directory from a run that started at or before StartStage::Sinograms
// (i.e. sino_NNNNN.tiff files already exist there).
#include "sinogram_io.h"
#include "fbp_reconstructor.h"
#include <opencv2/opencv.hpp>
#include <QString>
#include <QDir>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

std::vector<double> buildLinearAngles(int n_angles, double rot_start_deg, double last_angle_deg)
{
    std::vector<double> angles(n_angles);
    for (int a = 0; a < n_angles; ++a) {
        double frac = (n_angles > 1) ? static_cast<double>(a) / (n_angles - 1) : 0.0;
        angles[a] = (rot_start_deg + last_angle_deg * frac) * M_PI / 180.0;
    }
    return angles;
}

// Flags a detector-column defect at the rotation axis: computes the per-column mean over all
// angles (real object content mostly averages out over many random-ish projection angles; a
// systematic detector defect doesn't), then looks for a spike in that profile - a local second
// difference much larger than its neighbors' - centered on the middle column.
void checkCenterColumn(const cv::Mat& sino, int row_index)
{
    std::cout << "\n=== Center-column check (row " << row_index << ") ===\n";
    const int n_angles = sino.rows;
    const int n_cols = sino.cols;
    const int center = n_cols / 2;
    const int W = std::min(10, center - 1);

    std::vector<double> colMean(2 * W + 1, 0.0);
    for (int i = -W; i <= W; ++i) {
        double sum = 0.0;
        for (int a = 0; a < n_angles; ++a)
            sum += sino.at<float>(a, center + i);
        colMean[i + W] = sum / n_angles;
    }

    std::vector<double> spike(2 * W + 1, 0.0);
    for (int i = 1; i < 2 * W; ++i)
        spike[i] = std::abs(colMean[i] - 0.5 * (colMean[i - 1] + colMean[i + 1]));

    std::vector<double> others;
    for (int i = 1; i < 2 * W; ++i)
        if (i != W)
            others.push_back(spike[i]);
    std::sort(others.begin(), others.end());
    double medOther = others.empty() ? 0.0 : others[others.size() / 2];

    std::cout << std::fixed << std::setprecision(5);
    for (int i = -W; i <= W; ++i) {
        std::cout << "  col " << std::setw(5) << (center + i) << (i == 0 ? " (center)" : "         ")
                  << "  mean=" << std::setw(11) << colMean[i + W]
                  << "  spike=" << std::setw(11) << spike[i + W] << "\n";
    }
    std::cout << "  median neighbor spike = " << medOther << "\n";
    if (medOther > 1e-12 && spike[W] > 3.0 * medOther) {
        std::cout << ">>> Center column's spike is " << (spike[W] / medOther)
                  << "x the neighbor median - looks like a real detector-column defect at the "
                     "rotation axis (check flat-field/dead-pixel correction at that column).\n";
    } else {
        std::cout << "Center column does not stand out from its neighbors - probably not a "
                     "bad-pixel issue at this row.\n";
    }
}

double focusScore(const cv::Mat& slice)
{
    cv::Mat lap;
    cv::Laplacian(slice, lap, CV_32F);
    cv::Mat mask = cv::Mat::zeros(slice.size(), CV_8U);
    cv::circle(mask, cv::Point(slice.cols / 2, slice.rows / 2), static_cast<int>(slice.cols * 0.49), 255, -1);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev, mask);
    return stddev[0] * stddev[0]; // variance of Laplacian: standard focus/sharpness metric
}

void writeSlicePng(const cv::Mat& slice, const QString& path)
{
    double lo, hi;
    cv::minMaxLoc(slice, &lo, &hi);
    cv::Mat vis;
    slice.convertTo(vis, CV_8U, 255.0 / (hi - lo + 1e-9), -lo * 255.0 / (hi - lo + 1e-9));
    cv::imwrite(path.toStdString(), vis);
}

void corSweep(const cv::Mat& sino, const std::vector<double>& angles, int n_cols, const QString& outDir,
              double cor_min, double cor_max, double cor_step)
{
    std::cout << "\n=== COR sweep ===\n";
    FbpReconstructor fbp(n_cols);

    double bestOffset = cor_min;
    double bestScore = -1.0;
    for (double off = cor_min; off <= cor_max + 1e-9; off += cor_step) {
        cv::Mat s = sino.clone();
        if (off != 0.0)
            fbp.shift_sinogram(s, off);
        cv::Mat slice = fbp.reconstruct_slice(s, angles);

        double score = focusScore(slice);
        std::cout << "  offset=" << std::setw(7) << std::fixed << std::setprecision(2) << off
                  << "  sharpness=" << std::setprecision(6) << score << "\n";
        if (score > bestScore) {
            bestScore = score;
            bestOffset = off;
        }

        writeSlicePng(slice, outDir + QString("/cor_%1.png").arg(off, 0, 'f', 2));
    }

    std::cout << "\nSharpest reconstruction at offset = " << bestOffset
              << " - try that as (or as a starting point for) corOffset.\n";
    std::cout << "Per-offset slice PNGs written to " << outDir.toStdString()
              << " - look through them for where the star artifact is weakest/most symmetric.\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 8) {
        std::cerr << "Usage: cor_diagnostic <sino_dir> <n_cols> <row_index> <n_angles> "
                     "<rot_start_deg> <last_angle_deg> <out_dir> [cor_min=-3] [cor_max=3] [cor_step=0.25]\n";
        return 1;
    }

    QString sinoDir = argv[1];
    int n_cols = std::atoi(argv[2]);
    int row_index = std::atoi(argv[3]);
    int n_angles = std::atoi(argv[4]);
    double rot_start = std::atof(argv[5]);
    double last_angle = std::atof(argv[6]);
    QString outDir = argv[7];
    double cor_min = argc > 8 ? std::atof(argv[8]) : -3.0;
    double cor_max = argc > 9 ? std::atof(argv[9]) : 3.0;
    double cor_step = argc > 10 ? std::atof(argv[10]) : 0.25;

    QDir().mkpath(outDir);

    cv::Mat sino;
    try {
        SinogramReader reader(sinoDir, row_index + 1);
        sino = reader.readSlice(row_index);
    } catch (const std::exception& e) {
        std::cerr << "Failed to read sinogram: " << e.what() << "\n";
        return 1;
    }

    if (sino.rows != n_angles || sino.cols != n_cols) {
        std::cerr << "Note: sinogram is " << sino.rows << "x" << sino.cols << ", expected " << n_angles
                   << "x" << n_cols << " - using the file's actual size.\n";
        n_angles = sino.rows;
        n_cols = sino.cols;
    }

    checkCenterColumn(sino, row_index);

    std::vector<double> angles = buildLinearAngles(n_angles, rot_start, last_angle);
    corSweep(sino, angles, n_cols, outDir, cor_min, cor_max, cor_step);

    return 0;
}
