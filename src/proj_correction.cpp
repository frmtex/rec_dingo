#include "proj_correction.h"
#include "spot_filter.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <QString>
#include <QFileDialog>
#include <QRect>
#include <filesystem>
#include <fftw3.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

std::vector<float> phase_retrieval(const std::vector<float>& image,
                                                   int nx, int ny,
                                                   float alpha, float pix = 1.0f)
{
    auto is_pow2 = [](int v) { return v > 0 && (v & (v - 1)) == 0; };
    if (!is_pow2(nx) || !is_pow2(ny)) {
        throw std::invalid_argument(
            "phase_retrieval: nx and ny must be powers of two "
            "(pad the image before calling)");
    }
    if ((size_t)nx * (size_t)ny != image.size()) {
        throw std::invalid_argument("phase_retrieval: image size != nx*ny");
    }

    const size_t n = (size_t)nx * (size_t)ny;

    // fftwf_plan_dft_2d(n0, n1, ...) treats the buffer as n0 rows x n1 columns,
    // row-major - matching the (nx rows x ny cols) layout the filter below indexes
    // via i*ny+j, so no axis-order translation is needed (unlike vDSP_fft2d_zip's
    // own convention).
    fftwf_complex* data = fftwf_alloc_complex(n);
    for (size_t k = 0; k < n; ++k) {
        data[k][0] = image[k];
        data[k][1] = 0.0f;
    }

    fftwf_plan forward = fftwf_plan_dft_2d(nx, ny, data, data, FFTW_FORWARD, FFTW_ESTIMATE);
    if (!forward) { fftwf_free(data); throw std::runtime_error("fftwf_plan_dft_2d (forward) failed"); }

    // --- forward 2D FFT, in place ---
    fftwf_execute(forward);

    // --- build the real-valued Paganin filter, FFT-frequency ordered ---
    std::vector<float> filter(n);
    for (int i = 0; i < nx; ++i) {
        int kxi = (i <= nx / 2) ? i : i - nx;          // FFT freq order
        float kx = 2.0f * (float)M_PI * kxi / (nx * pix);
        float coskx = std::cos(kx * pix);
        for (int j = 0; j < ny; ++j) {
            int kyi = (j <= ny / 2) ? j : j - ny;
            float ky = 2.0f * (float)M_PI * kyi / (ny * pix);
            float cosky = std::cos(ky * pix);
            filter[(size_t)i * ny + j] =
                1.0f - (2.0f * alpha / (pix * pix)) * (coskx + cosky - 2.0f);
        }
    }

    // --- fim / filter  (complex / real -> divide both planes) ---
    for (size_t k = 0; k < n; ++k) {
        data[k][0] /= filter[k];
        data[k][1] /= filter[k];
    }

    // --- inverse 2D FFT, in place ---
    fftwf_plan inverse = fftwf_plan_dft_2d(nx, ny, data, data, FFTW_BACKWARD, FFTW_ESTIMATE);
    if (!inverse) { fftwf_destroy_plan(forward); fftwf_free(data); throw std::runtime_error("fftwf_plan_dft_2d (inverse) failed"); }
    fftwf_execute(inverse);

    // Undo FFTW's unnormalized round-trip scaling (same as the vDSP path).
    float scale = 1.0f / (float)n;
    for (size_t k = 0; k < n; ++k) {
        data[k][0] *= scale;
        data[k][1] *= scale;
    }

    // --- magnitude, then -log ---
    std::vector<float> phret(n);
    for (size_t k = 0; k < n; ++k) {
        float mag = std::sqrt(data[k][0] * data[k][0] + data[k][1] * data[k][1]);
        phret[k] = -std::log(mag);
    }

    fftwf_destroy_plan(forward);
    fftwf_destroy_plan(inverse);
    fftwf_free(data);
    return phret;
}

namespace fs = std::filesystem;

void Proj_correction::applySpotCorrection(cv::Mat& img, int kernel_size, int threshold)
{
    if (!use_gpu) {
        Spot_filter::spot_correction(img, kernel_size, threshold);
        return;
    }
    if (!spot_filter_gpu || img.rows != spot_filter_gpu_rows || img.cols != spot_filter_gpu_cols) {
        spot_filter_gpu = std::make_unique<SpotFilterCudaBackend>(img.rows, img.cols);
        spot_filter_gpu_rows = img.rows;
        spot_filter_gpu_cols = img.cols;
    }
    spot_filter_gpu->spot_correction(img, kernel_size, threshold);
}

std::vector<float> Proj_correction::applyPhaseRetrieval(const std::vector<float>& image, int nx, int ny,
                                                          float alpha, float pix)
{
    if (!use_gpu)
        return phase_retrieval(image, nx, ny, alpha, pix);
    if (!phase_retrieval_gpu || nx != phase_retrieval_gpu_nx || ny != phase_retrieval_gpu_ny) {
        phase_retrieval_gpu = std::make_unique<PhaseRetrievalCudaBackend>(nx, ny);
        phase_retrieval_gpu_nx = nx;
        phase_retrieval_gpu_ny = ny;
    }
    return phase_retrieval_gpu->retrieve(image, alpha, pix);
}

cv::Mat Proj_correction::read_cropped_binned(const QString& filename, const cv::Rect& fullResRoi) const
{
    cv::Mat img_1 = cv::imread(filename.toStdString(), cv::IMREAD_UNCHANGED);
    cv::Mat img_roi = img_1(fullResRoi);
    if (binning <= 1)
        return img_roi;

    cv::Mat binned;
    cv::resize(img_roi, binned, cv::Size(fullResRoi.width / binning, fullResRoi.height / binning),
               0, 0, cv::INTER_AREA);
    return binned;
}

void Proj_correction::load_op_di()
{
    QString ob_path = data_path + "/ob/", di_path = data_path + "/di/";
    QDir ob_dir = ob_path, di_dir = di_path;
    QStringList ob_list =  ob_dir.entryList(QDir::Files | QDir::NoDotAndDotDot);
    QStringList di_list =  di_dir.entryList(QDir::Files | QDir::NoDotAndDotDot);
    int x,y,h,w;
    roi_rect.getRect(&x,&y,&w,&h);
    cv::Rect roi(x,y,w,h);
    int kernel_size = spot_kernel_size;

    std::vector<cv::Mat> images;
    for (int i = 0; i < ob_list.size(); ++i) {
        QString filename = ob_path + ob_list.at(i);

        cv::Mat img_roi = read_cropped_binned(filename, roi);
        applySpotCorrection(img_roi, kernel_size, 200);
        images.push_back(img_roi);

    }

    if (images.empty()) return;

    // Initialize a 32-bit floating point accumulator with zeros
    cv::Mat accumulator_ob = cv::Mat::zeros(images[0].size(), CV_32FC1);

    // Accumulate all images into the floating-point matrix
    for (const auto& img : images) {
        cv::Mat floatImg;
        img.convertTo(floatImg, CV_32FC1);
        cv::accumulate(floatImg, accumulator_ob);
    }

    // Divide by the number of images to calculate the mean
    accumulator_ob /= static_cast<double>(images.size());
    ob_corr = accumulator_ob;

    images.clear();

    for (int i = 0; i < di_list.size(); ++i) {
        QString filename = di_path + di_list.at(i);

        cv::Mat img_roi = read_cropped_binned(filename, roi);
        applySpotCorrection(img_roi, kernel_size, 100);
        images.push_back(img_roi);

    }

    if (images.empty()) return;

    cv::Mat accumulator_di = cv::Mat::zeros(images[0].size(), CV_32FC1);
    // Accumulate all images into the floating-point matrix
    for (const auto& img : images) {
        cv::Mat floatImg;
        img.convertTo(floatImg, CV_32FC1);
        cv::accumulate(floatImg, accumulator_di);
    }

    // Divide by the number of images to calculate the mean
    accumulator_di /= static_cast<double>(images.size());

    // Convert back to standard 16-bit unsigned format
    di_corr =  accumulator_di;

    //accumulator.convertTo(averageImage, CV_16UC1);

    // 4. Process color imag
    //cv::imwrite("/Users/ugarbe/Desktop/test_data/test_out_di.tiff", averageImage);

};

double Proj_correction::roi_mean(const cv::Mat& img) const
{
    cv::Rect r(intensity_roi.x(), intensity_roi.y(), intensity_roi.width(), intensity_roi.height());
    r &= cv::Rect(0, 0, img.cols, img.rows);
    if (r.width <= 0 || r.height <= 0)
        return 0.0;
    return cv::mean(img(r))[0];
}

void Proj_correction::setIntensityRoi(const QRect& roi)
{
    if (roi.width() <= 0 || roi.height() <= 0)
        throw std::runtime_error("Proj_correction::setIntensityRoi: ROI is empty - draw a region first");

    intensity_roi = roi;
    // Disabled while establishing the reference so this call to get_projection_corr() returns the
    // raw (unscaled) projection 0, not one already normalized against a not-yet-set reference.
    intensity_roi_enabled = false;
    cv::Mat reference = get_projection_corr(0);
    intensity_reference = roi_mean(reference);
    intensity_roi_enabled = true;
}

cv::Mat Proj_correction::get_projection_corr(int index)
{
    QString proj_path = data_path + "/scan/";
    QDir proj_dir = proj_path;
    QStringList proj_list =  proj_dir.entryList(QDir::Files | QDir::NoDotAndDotDot);
    int x,y,h,w;
    roi_rect.getRect(&x,&y,&w,&h);
    cv::Rect roi(x,y,w,h);

    int kernel_size = spot_kernel_size;

    QString filename = proj_path + proj_list.at(index);
    cv::Mat img_roi = read_cropped_binned(filename, roi);
    applySpotCorrection(img_roi, kernel_size, 200);

    cv::Mat im_out;
    img_roi.convertTo(im_out, CV_32FC1);
    im_out = (im_out - di_corr)/(ob_corr-di_corr);

    // Beam-intensity fluctuation correction: rescale so this projection's own mean in the
    // beam-only ROI matches the reference established from projection 0 in setIntensityRoi().
    if (intensity_roi_enabled) {
        double current = roi_mean(im_out);
        if (current > 0.0)
            im_out *= (intensity_reference / current);
    }

    return im_out;
}

cv::Mat Proj_correction::get_projection_from_corr(int index)
{
    QString corr_path = data_path + "/corr/";
    QDir corr_dir = corr_path;
    QStringList corr_list = corr_dir.entryList(QDir::Files | QDir::NoDotAndDotDot);

    QString filename = corr_path + corr_list.at(index);
    cv::Mat img = cv::imread(filename.toStdString(), cv::IMREAD_UNCHANGED);
    if (img.empty())
        throw std::runtime_error(("Proj_correction::get_projection_from_corr: could not read " + filename).toStdString());
    return img;
}

void Proj_correction::get_first_image_corr(){

    cv::Mat im_out = get_projection_corr(0);
    im_out.convertTo(im_show, CV_16UC1, 65000);

};

cv::Mat Proj_correction::get_projection_corrected_full(int index)
{
    QString proj_path = data_path + "/scan/";
    QDir proj_dir = proj_path;
    QStringList proj_list = proj_dir.entryList(QDir::Files | QDir::NoDotAndDotDot);

    int x, y, h, w;
    roi_rect.getRect(&x, &y, &w, &h);
    cv::Rect roi(x, y, w, h); // full-resolution crop rect; read_cropped_binned bins after cropping
    if (binning > 1) { w /= binning; h /= binning; } // padding/reshape math below works in binned size

    QString filename = proj_path + proj_list.at(index);
    cv::Mat img_roi = read_cropped_binned(filename, roi);
    applySpotCorrection(img_roi, spot_kernel_size, 200);

    cv::Mat im_out;
    img_roi.convertTo(im_out, CV_32FC1);
    im_out = (im_out - di_corr) / (ob_corr - di_corr);

    // Beam-intensity fluctuation correction - see get_projection_corr() for details; kept in
    // sync with it so this matches what get_projection_corr() would return for the same index.
    if (intensity_roi_enabled) {
        double current = roi_mean(im_out);
        if (current > 0.0)
            im_out *= (intensity_reference / current);
    }

    int w_new = static_cast<int>(log2(w)) + 1;
    int w_f = static_cast<int>(std::pow(2, w_new));
    int h_new = static_cast<int>(log2(h)) + 1;
    int h_f = static_cast<int>(std::pow(2, h_new));

    int pad_left, pad_right, pad_top, pad_bottom;
    if (w % 2 == 0) {
        pad_left = abs((w_f - w) / 2);
        pad_right = abs((w_f - w) / 2);
    } else {
        pad_left = abs((w_f - w) / 2);
        pad_right = abs((w_f - w) / 2 + 1);
    }
    if (h % 2 == 0) {
        pad_top = abs((h_f - h) / 2);
        pad_bottom = abs((h_f - h) / 2);
    } else {
        pad_top = abs((h_f - h) / 2);
        pad_bottom = abs((h_f - h) / 2 + 1);
    }

    cv::copyMakeBorder(im_out, im_out, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    im_out = im_out.reshape(1, 1);

    std::vector<float> im_test;
    im_out.copyTo(im_test);

    std::vector<float> im_out_new = applyPhaseRetrieval(im_test, h_f, w_f, 1.0, 1.0);

    cv::Rect cropRoi(pad_left, pad_top, w, h);
    // im_out_pad is a view into the local im_out_new vector - clone before it goes out of scope
    // (the original inline code got away without cloning since it wrote the file immediately in
    // the same scope; returning the Mat from a function requires an owned copy).
    cv::Mat im_out_pad(h_f, w_f, CV_32FC1, im_out_new.data());
    return im_out_pad(cropRoi).clone();
}

void Proj_correction::run_scan(const std::function<void(int, int)>& progressCallback){

    QString proj_path = data_path + "/scan/";
    QString proj_out = data_path + "/corr/";

    fs::create_directories(proj_out.toStdString());

    QDir proj_dir = proj_path;
    QStringList proj_list =  proj_dir.entryList(QDir::Files | QDir::NoDotAndDotDot);

    for (int i = 0; i < proj_list.size(); ++i) {
        cv::Mat im_out_final = get_projection_corrected_full(i);

        QString filename_out = proj_out + proj_list.at(i);
        cv::imwrite(filename_out.toStdString(), im_out_final);

        if (progressCallback)
            progressCallback(i + 1, proj_list.size());
    }

};