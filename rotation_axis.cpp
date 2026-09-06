#include "rotation_axis.h"
#include <cmath>
#include <stdexcept>

namespace {

// RMSE between proj0 rolled circularly by `t` columns and rowFlip, matching
// np.roll(proj0_row, t, axis=0) - proj_flip_row.
double row_rmse_at_shift(const float* row0, const float* rowFlip, int nd, int t)
{
    double sum = 0.0;
    for (int i = 0; i < nd; ++i) {
        int j = ((i - t) % nd + nd) % nd;
        double d = static_cast<double>(row0[j]) - static_cast<double>(rowFlip[i]);
        sum += d * d;
    }
    return sum / nd;
}

int floor_div(int a, int b)
{
    int q = a / b;
    int r = a % b;
    if (r != 0 && ((r < 0) != (b < 0)))
        --q;
    return q;
}

} // namespace

CorResult RotationAxis::find_axis(const cv::Mat& proj0, const cv::Mat& proj180,
                                   const std::vector<cv::Rect>& rois, int ystep)
{
    if (proj0.size() != proj180.size())
        throw std::invalid_argument("RotationAxis::find_axis: proj0/proj180 size mismatch");
    if (rois.empty())
        throw std::invalid_argument("RotationAxis::find_axis: no ROIs selected");

    const int nd = proj0.cols;
    const int nz = proj0.rows;

    cv::Mat projFlip;
    cv::flip(proj180, projFlip, 1);

    std::vector<int> slices;
    for (const auto& roi : rois) {
        int ymin = std::max(0, roi.y);
        int ymax = std::min(nz - 1, roi.y + roi.height);
        for (int s = ymin; s <= ymax; s += ystep)
            slices.push_back(s);
    }
    if (slices.empty())
        throw std::invalid_argument("RotationAxis::find_axis: ROIs produced no sample rows");

    const int tmin = -nd / 2;
    const int tmax = nd - nd / 2;

    std::vector<double> shift(slices.size());
    for (size_t z = 0; z < slices.size(); ++z) {
        int posz = slices[z];
        const float* row0 = proj0.ptr<float>(posz);
        const float* rowFlip = projFlip.ptr<float>(posz);

        double minimum = 1e7;
        int index_min = 0;
        for (int t = tmin; t <= tmax; ++t) {
            double rmse = row_rmse_at_shift(row0, rowFlip, nd, t);
            if (rmse <= minimum) {
                minimum = rmse;
                index_min = t;
            }
        }
        shift[z] = index_min;
    }

    // Ordinary least-squares linear fit shift ~ m*slice + q.
    const size_t n = slices.size();
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (size_t i = 0; i < n; ++i) {
        double x = slices[i];
        double y = shift[i];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    double denom = static_cast<double>(n) * sum_xx - sum_x * sum_x;
    double m = (denom != 0.0) ? (static_cast<double>(n) * sum_xy - sum_x * sum_y) / denom : 0.0;
    double q = (sum_y - m * sum_x) / static_cast<double>(n);

    double theta = std::atan(0.5 * m) * 180.0 / M_PI;

    int v = static_cast<int>(std::lround(m * nz * 0.5 + q));
    double middle_shift = floor_div(v, 2);

    CorResult result;
    result.offset = middle_shift;
    result.tilt_deg = theta;
    return result;
}

double RotationAxis::find_centre(const cv::Mat& proj0, const cv::Mat& proj180,
                                  const std::vector<cv::Rect>& rois, int ystep, double& tilt_deg_out)
{
    CorResult r = find_axis(proj0, proj180, rois, ystep);
    tilt_deg_out = r.tilt_deg;
    return proj0.cols/2.0 - r.offset;
}
