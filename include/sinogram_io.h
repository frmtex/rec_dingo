#ifndef SINOGRAM_IO_H
#define SINOGRAM_IO_H
#include <opencv2/opencv.hpp>
#include <QString>
#include <vector>

// Shared filename convention for per-vertical-slice sinogram files, one
// CV_32FC1 TIFF of shape (n_angles x n_cols) per row of the reconstructed volume.
QString sinogram_slice_path(const QString& sinoDir, int rowIndex);

// Accumulates a chunk of vertical rows (across all projection angles) in
// memory and flushes it to individual sinogram TIFF files. Used so that only
// `numRows` rows are ever held in RAM at once, regardless of scan size:
// the caller loops chunk-by-chunk over the vertical extent, and for each
// chunk loops once over all projection angles feeding rows in via
// addProjectionRows().
class SinogramWriter
{
public:
    SinogramWriter(QString sinoDir, int n_cols);

    void beginChunk(int rowStart, int numRows, int n_angles);
    // rowsBlock: numRows x n_cols CV_32FC1, the vertical row range [rowStart, rowStart+numRows)
    // of the (tilt-corrected) projection at angleIndex.
    void addProjectionRows(int angleIndex, const cv::Mat& rowsBlock);
    void finishChunk();

private:
    QString sinoDir_;
    int n_cols_;
    int rowStart_ = 0;
    std::vector<cv::Mat> sinograms_; // one (n_angles x n_cols) buffer per row in the active chunk
};

// Sequential reader for the reconstruction pass: one sinogram slice at a time.
class SinogramReader
{
public:
    SinogramReader(QString sinoDir, int n_rows);

    int rowCount() const { return n_rows_; }
    cv::Mat readSlice(int rowIndex) const;

private:
    QString sinoDir_;
    int n_rows_;
};

#endif // SINOGRAM_IO_H
