#include "ring_filter.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

// --- Daubechies decomposition low-pass filter coefficients (orthonormal, sum = sqrt(2)) ---
// Standard published values (Daubechies 1992, "Ten Lectures on Wavelets"), same convention
// PyWavelets uses for its dec_lo arrays. Validated at runtime against the defining orthogonal
// QMF conditions in checkWaveletTable() before first use.
const std::vector<std::vector<float>>& daubechies_tables()
{
    static const std::vector<std::vector<float>> tables = {
        // db1 (Haar)
        { 0.70710678118654752f, 0.70710678118654752f },
        // db2
        { -0.12940952255092145f, 0.22414386804185735f, 0.83651630373746899f, 0.48296291314469025f },
        // db3
        { 0.03522629188570953f, -0.08544127388202666f, -0.13501102001025458f,
          0.45987750211849157f, 0.80689150931109257f, 0.33267055295008261f },
        // db4
        { -0.010597401785069032f, 0.032883011666982945f, 0.030841381835560764f, -0.18703481171909309f,
          -0.02798376941685985f, 0.6308807679298589f, 0.7148465705529157f, 0.23037781330885523f },
        // db5
        { 0.003335725285001549f, -0.012580751999015526f, -0.006241490213011705f, 0.07757149384006515f,
          -0.03224486958502952f, -0.24229488706619015f, 0.13842814590132074f, 0.7243085284385744f,
          0.6038292697974729f, 0.160102397974125f },
        // db6
        { -0.00107730108499558f, 0.004777257511010651f, 0.0005538422009938016f, -0.031582039318031156f,
          0.02752286553001629f, 0.09750160558707936f, -0.12976686756709563f, -0.22626469396516913f,
          0.3152503517092432f, 0.7511339080215775f, 0.4946238903983854f, 0.11154074335008017f },
        // db7
        { 0.0003537138000010399f, -0.0018016407039998328f, 0.00042957797300470274f, 0.012550998556013784f,
          -0.01657454163101562f, -0.03802993693503463f, 0.0806126091510659f, 0.07130921926705004f,
          -0.22403618499416572f, -0.14390600392910627f, 0.4697822874053586f, 0.7291320908465551f,
          0.39653931948230575f, 0.07785205408506236f },
        // db8
        { -0.00011747678400228192f, 0.0006754494059985568f, -0.0003917403729959771f, -0.00487035299301066f,
          0.008746094047015655f, 0.013981027917015516f, -0.04408825393106472f, -0.01736930100202211f,
          0.128747426620186f, 0.00047248457399797254f, -0.2840155429624281f, -0.015829105256023893f,
          0.5853546836548691f, 0.6756307362980128f, 0.3128715909144659f, 0.05441584224308161f },
    };
    return tables;
}

struct Wavelet
{
    std::vector<float> lo; // dec/rec low-pass (h)
    std::vector<float> hi; // dec/rec high-pass (g), derived via g[j] = (-1)^j * h[F-1-j]
    int length() const { return static_cast<int>(lo.size()); }
};

void check_wavelet_table(const std::vector<float>& h, int order)
{
    const double tol = 1e-4;
    double sum = 0.0, sumSq = 0.0;
    for (float v : h) { sum += v; sumSq += static_cast<double>(v) * v; }

    if (std::abs(sum - std::sqrt(2.0)) > tol)
        throw std::runtime_error("RingFilter: db" + std::to_string(order) + " coefficient table failed "
                                  "the sum=sqrt(2) check (transcription error?)");
    if (std::abs(sumSq - 1.0) > tol)
        throw std::runtime_error("RingFilter: db" + std::to_string(order) + " coefficient table failed "
                                  "the orthonormality (sum of squares = 1) check");

    int F = static_cast<int>(h.size());
    for (int k = 1; k < F / 2; ++k) {
        double dot = 0.0;
        for (int j = 0; j + 2 * k < F; ++j)
            dot += static_cast<double>(h[j]) * h[j + 2 * k];
        if (std::abs(dot) > tol)
            throw std::runtime_error("RingFilter: db" + std::to_string(order) + " coefficient table failed "
                                      "the even-shift orthogonality check (transcription error?)");
    }
}

Wavelet make_wavelet(int order)
{
    const auto& tables = daubechies_tables();
    if (order < 1 || order > static_cast<int>(tables.size()))
        throw std::invalid_argument("RingFilter: Daubechies order must be between 1 and " +
                                     std::to_string(tables.size()));

    const std::vector<float>& h = tables[order - 1];
    check_wavelet_table(h, order);

    int F = static_cast<int>(h.size());
    Wavelet w;
    w.lo = h;
    w.hi.resize(F);
    for (int j = 0; j < F; ++j)
        w.hi[j] = ((j % 2 == 0) ? 1.0f : -1.0f) * h[F - 1 - j];
    return w;
}

// Periodic 1D DWT of one signal of even length n, filter length F <= n.
// Forward:  cA[k] = sum_j lo[j] * x[(2k+j) mod n],  cD[k] = sum_j hi[j] * x[(2k+j) mod n]
// This is an orthogonal transform (given lo/hi satisfy the QMF conditions checked above), so its
// exact inverse is the adjoint sum below -- no separate "reconstruction filter" is needed.
void dwt1d(const float* x, int n, const Wavelet& w, float* outLow, float* outHigh)
{
    int F = w.length();
    int half = n / 2;
    for (int k = 0; k < half; ++k) {
        double sumLo = 0.0, sumHi = 0.0;
        for (int j = 0; j < F; ++j) {
            int idx = (2 * k + j) % n;
            sumLo += static_cast<double>(w.lo[j]) * x[idx];
            sumHi += static_cast<double>(w.hi[j]) * x[idx];
        }
        outLow[k] = static_cast<float>(sumLo);
        outHigh[k] = static_cast<float>(sumHi);
    }
}

// Inverse of dwt1d: x[m] = sum_k cA[k]*lo[(m-2k) mod n] + cD[k]*hi[(m-2k) mod n]
void idwt1d(const float* cA, const float* cD, int half, const Wavelet& w, float* out)
{
    int n = half * 2;
    int F = w.length();
    std::vector<double> acc(n, 0.0);
    for (int k = 0; k < half; ++k) {
        for (int j = 0; j < F; ++j) {
            int idx = (2 * k + j) % n;
            acc[idx] += static_cast<double>(cA[k]) * w.lo[j] + static_cast<double>(cD[k]) * w.hi[j];
        }
    }
    for (int i = 0; i < n; ++i)
        out[i] = static_cast<float>(acc[i]);
}

void dwt1d_along_cols(const cv::Mat& src, const Wavelet& w, cv::Mat& outLow, cv::Mat& outHigh)
{
    int half = src.cols / 2;
    outLow.create(src.rows, half, CV_32FC1);
    outHigh.create(src.rows, half, CV_32FC1);
    for (int r = 0; r < src.rows; ++r)
        dwt1d(src.ptr<float>(r), src.cols, w, outLow.ptr<float>(r), outHigh.ptr<float>(r));
}

void dwt1d_along_rows(const cv::Mat& src, const Wavelet& w, cv::Mat& outLow, cv::Mat& outHigh)
{
    cv::Mat srcT = src.t();
    cv::Mat lowT, highT;
    dwt1d_along_cols(srcT, w, lowT, highT);
    outLow = lowT.t();
    outHigh = highT.t();
}

cv::Mat idwt1d_along_cols(const cv::Mat& low, const cv::Mat& high, const Wavelet& w)
{
    cv::Mat out(low.rows, low.cols * 2, CV_32FC1);
    for (int r = 0; r < low.rows; ++r)
        idwt1d(low.ptr<float>(r), high.ptr<float>(r), low.cols, w, out.ptr<float>(r));
    return out;
}

cv::Mat idwt1d_along_rows(const cv::Mat& low, const cv::Mat& high, const Wavelet& w)
{
    cv::Mat outT = idwt1d_along_cols(low.t(), high.t(), w);
    return outT.t();
}

struct Dwt2Result
{
    cv::Mat cA, cH, cV, cD;
};

// Row axis = angle axis, column axis = detector axis. cV (row-low, col-high) is smooth across
// angle but has detail across detector position -- exactly where an angle-constant, detector-
// localized ring/stripe artifact concentrates its energy.
Dwt2Result dwt2(const cv::Mat& src, const Wavelet& w)
{
    cv::Mat rowLow, rowHigh;
    dwt1d_along_rows(src, w, rowLow, rowHigh);

    Dwt2Result r;
    dwt1d_along_cols(rowLow, w, r.cA, r.cV);
    dwt1d_along_cols(rowHigh, w, r.cH, r.cD);
    return r;
}

cv::Mat idwt2(const cv::Mat& cA, const cv::Mat& cH, const cv::Mat& cV, const cv::Mat& cD, const Wavelet& w)
{
    cv::Mat rowLow = idwt1d_along_cols(cA, cV, w);
    cv::Mat rowHigh = idwt1d_along_cols(cH, cD, w);
    return idwt1d_along_rows(rowLow, rowHigh, w);
}

// In-place circular shift of `m`'s rows by `shiftBy` (result[i] = m[(i-shiftBy) mod rows]).
void circular_row_shift(cv::Mat& m, int shiftBy)
{
    int n = m.rows;
    shiftBy = ((shiftBy % n) + n) % n;
    if (shiftBy == 0)
        return;
    cv::Mat tmp = m.clone();
    tmp.rowRange(n - shiftBy, n).copyTo(m.rowRange(0, shiftBy));
    tmp.rowRange(0, n - shiftBy).copyTo(m.rowRange(shiftBy, n));
}

// Damps low-angular-frequency content of cV in the Fourier domain (ports the per-level cV
// processing in _ring_removal_wavelets_sinogram). Only the row (angle) axis needs an fftshift:
// the damping weight is column-independent, so the corresponding column-axis fftshift/ifftshift
// pair cancels out and can be skipped entirely.
void damp_vertical_stripes(cv::Mat& cV, double sigma)
{
    int nrow = cV.rows;
    CV_Assert(nrow % 2 == 0);

    cv::Mat planes[] = { cV, cv::Mat::zeros(cV.size(), CV_32F) };
    cv::Mat complexImg;
    cv::merge(planes, 2, complexImg);
    cv::dft(complexImg, complexImg, cv::DFT_COMPLEX_OUTPUT);

    circular_row_shift(complexImg, nrow / 2);

    for (int i = 0; i < nrow; ++i) {
        double y_hat = (-nrow + 2.0 * i + 1.0) / 2.0;
        double damp = 1.0 - std::exp(-(y_hat * y_hat) / (2.0 * sigma * sigma));
        complexImg.row(i) *= damp;
    }

    circular_row_shift(complexImg, nrow / 2);

    cv::Mat inv;
    cv::idft(complexImg, inv, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    cV = inv;
}

cv::Mat pad_mean_edge(const cv::Mat& sino, int pad)
{
    if (pad <= 0)
        return sino.clone();

    cv::Mat colMean;
    cv::reduce(sino, colMean, 0, cv::REDUCE_AVG, CV_32F);

    cv::Mat padded(sino.rows + 2 * pad, sino.cols, CV_32FC1);
    for (int r = 0; r < pad; ++r)
        colMean.copyTo(padded.row(r));
    sino.copyTo(padded.rowRange(pad, pad + sino.rows));
    for (int r = 0; r < pad; ++r)
        colMean.copyTo(padded.row(pad + sino.rows + r));

    cv::Mat out;
    cv::copyMakeBorder(padded, out, 0, 0, pad, pad, cv::BORDER_REPLICATE);
    return out;
}

// Grows `m` (edge-replicated) so both dimensions are divisible by 2^(level+1) and comfortably
// larger than the filter support at the coarsest level. The extra factor of two beyond 2^level
// matters because cV at the deepest level has padded.rows/2^level rows -- damp_vertical_stripes
// needs that row count itself to be even, i.e. padded.rows divisible by 2^(level+1).
cv::Mat ensure_divisible(const cv::Mat& m, int level, int filterLen)
{
    int factor = 1 << (level + 1);
    int minSize = filterLen * factor;

    int newRows = std::max(((m.rows + factor - 1) / factor) * factor,
                            ((minSize + factor - 1) / factor) * factor);
    int newCols = std::max(((m.cols + factor - 1) / factor) * factor,
                            ((minSize + factor - 1) / factor) * factor);

    if (newRows == m.rows && newCols == m.cols)
        return m;

    cv::Mat out;
    cv::copyMakeBorder(m, out, 0, newRows - m.rows, 0, newCols - m.cols, cv::BORDER_REPLICATE);
    return out;
}

// Median of a mutable vector via nth_element (partial sort, O(n) average case).
float median_of(std::vector<float>& v)
{
    size_t n = v.size();
    size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    float m = v[mid];
    if (n % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + mid - 1, v.begin() + mid);
        m = 0.5f * (m + v[mid - 1]);
    }
    return m;
}

// Median of each column across all rows. Transposed first so each column becomes a contiguous
// row, which is both simpler and cache-friendlier than striding through the original layout.
std::vector<float> column_median_profile(const cv::Mat& sinogram)
{
    cv::Mat t;
    cv::transpose(sinogram, t); // t.rows == sinogram.cols, t.cols == sinogram.rows
    int ncol = t.rows, nrow = t.cols;

    std::vector<float> profile(ncol);
    std::vector<float> buf(nrow);
    for (int c = 0; c < ncol; ++c) {
        const float* rowPtr = t.ptr<float>(c);
        std::copy(rowPtr, rowPtr + nrow, buf.begin());
        profile[c] = median_of(buf);
    }
    return profile;
}

// 1D median filter with edge-replicated boundaries.
std::vector<float> median_filter_1d(const std::vector<float>& profile, int window)
{
    int n = static_cast<int>(profile.size());
    if (window < 1)
        window = 1;
    if (window % 2 == 0)
        ++window;
    int half = window / 2;

    std::vector<float> out(n);
    std::vector<float> buf;
    buf.reserve(window);
    for (int i = 0; i < n; ++i) {
        buf.clear();
        for (int k = -half; k <= half; ++k) {
            int idx = std::max(0, std::min(n - 1, i + k));
            buf.push_back(profile[idx]);
        }
        out[i] = median_of(buf);
    }
    return out;
}

// Median absolute deviation - a robust (outlier-resistant) stand-in for standard deviation.
double median_abs_deviation(const std::vector<float>& values)
{
    std::vector<float> tmp = values;
    float med = median_of(tmp);
    std::vector<float> absDev(values.size());
    for (size_t i = 0; i < values.size(); ++i)
        absDev[i] = std::abs(values[i] - med);
    return median_of(absDev);
}

} // namespace

void RingFilter::remove_stripes(cv::Mat& sinogram, int level, double sigma, int order, int pad,
                                 int maskInnerRadius, int maskOuterRadius)
{
    CV_Assert(sinogram.type() == CV_32FC1);
    if (level < 1)
        throw std::invalid_argument("RingFilter::remove_stripes: level must be >= 1");

    const int center = sinogram.cols / 2;

    // See remove_stripes' declaration: always protect a small floor immediately around the
    // rotation axis, regardless of the caller-specified mask - this is the one region where this
    // filter's "can't tell a stripe from real content" blind spot is unavoidable (a ring of
    // vanishing radius has no distinguishing signal at all), and leaving it fully exposed was
    // observed producing a starburst artifact. 0.5% of the sinogram width, floored at 3 columns.
    const int centerFloor = std::max(3, sinogram.cols / 200);
    int floorLo = std::max(0, center - centerFloor);
    int floorHi = std::min(sinogram.cols, center + centerFloor);

    int rightLo = 0, rightHi = 0, leftLo = 0, leftHi = 0;
    bool haveUserMask = maskOuterRadius > 0 && maskOuterRadius > maskInnerRadius;
    if (haveUserMask) {
        rightLo = std::min(sinogram.cols, center + maskInnerRadius);
        rightHi = std::min(sinogram.cols, center + maskOuterRadius);
        leftHi = std::max(0, center - maskInnerRadius);
        leftLo = std::max(0, center - maskOuterRadius);
    }
    cv::Mat original = sinogram.clone();

    Wavelet w = make_wavelet(order);

    cv::Mat padded = pad_mean_edge(sinogram, pad);
    padded = ensure_divisible(padded, level, w.length());

    std::vector<Dwt2Result> levels(level);
    cv::Mat current = padded;
    for (int l = 0; l < level; ++l) {
        levels[l] = dwt2(current, w);
        current = levels[l].cA;
    }

    for (int l = 0; l < level; ++l)
        damp_vertical_stripes(levels[l].cV, sigma);

    cv::Mat approx = current;
    for (int l = level - 1; l >= 0; --l) {
        cv::Size target = levels[l].cH.size();
        if (approx.size() != target)
            approx = approx(cv::Rect(0, 0, target.width, target.height)).clone();
        approx = idwt2(approx, levels[l].cH, levels[l].cV, levels[l].cD, w);
    }

    cv::Rect roi(pad, pad, sinogram.cols, sinogram.rows);
    approx(roi).copyTo(sinogram);

    if (floorLo < floorHi)
        original.colRange(floorLo, floorHi).copyTo(sinogram.colRange(floorLo, floorHi));
    if (haveUserMask) {
        if (rightLo < rightHi)
            original.colRange(rightLo, rightHi).copyTo(sinogram.colRange(rightLo, rightHi));
        if (leftLo < leftHi)
            original.colRange(leftLo, leftHi).copyTo(sinogram.colRange(leftLo, leftHi));
    }
}

void RingFilter::remove_large_stripes(cv::Mat& sinogram, double snr, int smoothWindow,
                                       int maskInnerRadius, int maskOuterRadius)
{
    CV_Assert(sinogram.type() == CV_32FC1);
    const int ncol = sinogram.cols;
    if (ncol < 3)
        return;

    std::vector<float> profile = column_median_profile(sinogram);
    std::vector<float> smooth = median_filter_1d(profile, smoothWindow);

    std::vector<float> deviation(ncol);
    for (int c = 0; c < ncol; ++c)
        deviation[c] = profile[c] - smooth[c];

    double robustSigma = 1.4826 * median_abs_deviation(deviation);
    double threshold = snr * robustSigma;

    std::vector<char> flagged(ncol, 0);
    if (threshold > 0.0) {
        for (int c = 0; c < ncol; ++c)
            flagged[c] = std::abs(deviation[c]) > threshold;
    }

    // Dilate by one column each side to catch a stripe's shoulders too.
    std::vector<char> dilated = flagged;
    for (int c = 0; c < ncol; ++c) {
        if (!flagged[c])
            continue;
        if (c > 0) dilated[c - 1] = 1;
        if (c + 1 < ncol) dilated[c + 1] = 1;
    }

    const int center = ncol / 2;
    const bool haveMask = maskOuterRadius > 0 && maskOuterRadius > maskInnerRadius;
    for (int c = 0; c < ncol; ++c) {
        if (!dilated[c])
            continue;
        if (haveMask) {
            int dist = std::abs(c - center);
            if (dist >= maskInnerRadius && dist < maskOuterRadius)
                continue;
        }
        float dev = deviation[c];
        for (int r = 0; r < sinogram.rows; ++r)
            sinogram.at<float>(r, c) -= dev;
    }
}

void RingFilter::remove_small_stripes(cv::Mat& sinogram, int window,
                                       int maskInnerRadius, int maskOuterRadius)
{
    CV_Assert(sinogram.type() == CV_32FC1);
    const int nrow = sinogram.rows;
    const int ncol = sinogram.cols;
    if (ncol < 3 || nrow < 2)
        return;

    // sortedVals(r, c) is the r-th smallest value in column c; sortedIdx(r, c) is which original
    // row it came from - the permutation needed to un-sort afterward.
    cv::Mat sortedVals, sortedIdx;
    cv::sort(sinogram, sortedVals, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
    cv::sortIdx(sinogram, sortedIdx, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);

    const int center = ncol / 2;
    const bool haveMask = maskOuterRadius > 0 && maskOuterRadius > maskInnerRadius;

    std::vector<float> rowBuf(ncol);
    for (int r = 0; r < nrow; ++r) {
        const float* rowPtr = sortedVals.ptr<float>(r);
        std::copy(rowPtr, rowPtr + ncol, rowBuf.begin());
        std::vector<float> smoothRow = median_filter_1d(rowBuf, window);

        const int* idxPtr = sortedIdx.ptr<int>(r);
        for (int c = 0; c < ncol; ++c) {
            if (haveMask) {
                int dist = std::abs(c - center);
                if (dist >= maskInnerRadius && dist < maskOuterRadius)
                    continue;
            }
            // Every (r, c) pair here maps to a distinct (idxPtr[c], c) - sortedIdx's column c is
            // a permutation of 0..nrow-1 - so this loop over r touches every original element of
            // an unmasked column exactly once, always reading from sortedVals/sortedIdx rather
            // than the (already being overwritten) sinogram itself.
            sinogram.at<float>(idxPtr[c], c) = smoothRow[c];
        }
    }
}
