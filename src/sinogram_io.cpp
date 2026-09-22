#include "sinogram_io.h"
#include <QDir>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

QString sinogram_slice_path(const QString& sinoDir, int rowIndex)
{
    return sinoDir + QString("/sino_%1.tiff").arg(rowIndex, 5, 10, QChar('0'));
}

SinogramWriter::SinogramWriter(QString sinoDir, int n_cols)
    : sinoDir_(std::move(sinoDir)), n_cols_(n_cols)
{
    fs::create_directories(sinoDir_.toStdString());
}

void SinogramWriter::beginChunk(int rowStart, int numRows, int n_angles)
{
    rowStart_ = rowStart;
    // cv::Mat's copy constructor is shallow (shared, refcounted data), so
    // vector::assign(numRows, someMat) would give every element the SAME
    // underlying buffer - each row needs its own cv::Mat::zeros() call.
    sinograms_.clear();
    sinograms_.reserve(numRows);
    for (int r = 0; r < numRows; ++r)
        sinograms_.push_back(cv::Mat::zeros(n_angles, n_cols_, CV_32FC1));
}

void SinogramWriter::addProjectionRows(int angleIndex, const cv::Mat& rowsBlock)
{
    if (rowsBlock.cols != n_cols_ || rowsBlock.rows != static_cast<int>(sinograms_.size()))
        throw std::invalid_argument("SinogramWriter::addProjectionRows: size mismatch with active chunk");

    for (int r = 0; r < rowsBlock.rows; ++r)
        rowsBlock.row(r).copyTo(sinograms_[r].row(angleIndex));
}

void SinogramWriter::finishChunk()
{
    for (size_t r = 0; r < sinograms_.size(); ++r) {
        QString path = sinogram_slice_path(sinoDir_, rowStart_ + static_cast<int>(r));
        cv::imwrite(path.toStdString(), sinograms_[r]);
    }
    sinograms_.clear();
}

SinogramReader::SinogramReader(QString sinoDir, int n_rows)
    : sinoDir_(std::move(sinoDir)), n_rows_(n_rows)
{
}

cv::Mat SinogramReader::readSlice(int rowIndex) const
{
    QString path = sinogram_slice_path(sinoDir_, rowIndex);
    cv::Mat sino = cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED);
    if (sino.empty())
        throw std::runtime_error(("SinogramReader: could not read " + path).toStdString());
    return sino;
}
