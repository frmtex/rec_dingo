#include "post_process_worker.h"
#include "beam_hardening.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <atomic>
#include <filesystem>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

PostProcessWorker::PostProcessWorker(Params params, QObject* parent)
    : QObject(parent), params_(std::move(params))
{
}

namespace {

std::vector<fs::path> listRecoFiles(const QString& recoDir)
{
    std::vector<fs::path> files;
    if (!fs::exists(recoDir.toStdString()))
        return files;
    for (const auto& entry : fs::directory_iterator(recoDir.toStdString())) {
        if (entry.is_regular_file() && entry.path().extension() == ".tiff")
            files.push_back(entry.path());
    }
    // Filenames are zero-padded (reco_00000.tiff...), so lexical sort is numeric sort.
    std::sort(files.begin(), files.end());
    return files;
}

cv::Mat readSlice(const fs::path& path)
{
    cv::Mat slice = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (slice.empty())
        throw std::runtime_error("Failed to read " + path.string());
    return slice;
}

// Runs perFile(i) for every index in [0, count) across a thread pool - same pattern as
// reconstruction_worker.cpp's row-parallel run(). Each call to perFile is independent (its own
// file read/write); any shared state it touches (accumulators, progress) must synchronize itself.
// Rethrows the first exception seen, after every thread has stopped.
template <typename Func>
void parallelForIndices(size_t count, const Func& perFile)
{
    unsigned int numThreads = std::max(1u, std::thread::hardware_concurrency());
    numThreads = std::min(numThreads, static_cast<unsigned int>(std::max<size_t>(1, count)));

    std::atomic<size_t> nextIndex{0};
    std::atomic<bool> failed{false};
    std::mutex errorMutex;
    std::string firstError;

    auto worker = [&]() {
        for (;;) {
            size_t i = nextIndex.fetch_add(1);
            if (i >= count || failed.load(std::memory_order_relaxed))
                return;
            try {
                perFile(i);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(errorMutex);
                if (!failed.exchange(true))
                    firstError = e.what();
                return;
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(numThreads);
    for (unsigned int t = 0; t < numThreads; ++t)
        pool.emplace_back(worker);
    for (auto& th : pool)
        th.join();

    if (failed.load())
        throw std::runtime_error(firstError);
}

} // namespace

void PostProcessWorker::run()
{
    try {
        if (!params_.beamHardeningEnabled && !params_.convert16Enabled)
            throw std::runtime_error("Enable at least one post-processing step");

        QString recoDir = params_.workingPath + "/reco/";
        std::vector<fs::path> files = listRecoFiles(recoDir);
        if (files.empty())
            throw std::runtime_error("No reconstructed slices found in reco/ - run Reconstruction first");

        QString postDir = params_.workingPath + "/post/";
        fs::create_directories(postDir.toStdString());

        // BeamHardening::apply is a pure function (fresh local buffers per call, no shared/static
        // state), so this is safe to call concurrently from multiple threads on different slices.
        auto applyBh = [this](cv::Mat& slice) {
            if (params_.beamHardeningEnabled)
                slice = BeamHardening::apply(slice, params_.bhC1, params_.bhC2, params_.bhC3);
        };

        if (!params_.convert16Enabled) {
            // Beam hardening only: single pass, float32 output - no dataset-wide range needed.
            std::atomic<size_t> done{0};
            parallelForIndices(files.size(), [&](size_t i) {
                cv::Mat slice = readSlice(files[i]);
                applyBh(slice);
                QString outPath = postDir + QString::fromStdString(files[i].filename().string());
                cv::imwrite(outPath.toStdString(), slice);
                size_t d = done.fetch_add(1) + 1;
                emit progress(static_cast<int>(100.0 * d / files.size()),
                              QString("Beam hardening: %1/%2").arg(d).arg(files.size()));
            });
            emit finished();
            return;
        }

        // 16-bit conversion (optionally with beam-hardening correction applied first): the output
        // range has to be known before any slice can be written, so this streams the dataset up to
        // three times - min/max, then a histogram for percentile-based clipping (so a few outlier
        // pixels don't compress the useful range), then the actual clip/scale/write. Cheap next to
        // the disk I/O each pass already does.
        double loMinMax = std::numeric_limits<double>::max();
        double hiMinMax = std::numeric_limits<double>::lowest();
        {
            std::mutex minMaxMutex;
            std::atomic<size_t> done{0};
            parallelForIndices(files.size(), [&](size_t i) {
                cv::Mat slice = readSlice(files[i]);
                applyBh(slice);
                double mn, mx;
                cv::minMaxLoc(slice, &mn, &mx);
                {
                    std::lock_guard<std::mutex> lock(minMaxMutex);
                    loMinMax = std::min(loMinMax, mn);
                    hiMinMax = std::max(hiMinMax, mx);
                }
                size_t d = done.fetch_add(1) + 1;
                emit progress(static_cast<int>(33.0 * d / files.size()),
                              QString("Scanning range: %1/%2").arg(d).arg(files.size()));
            });
        }

        double loVal = loMinMax;
        double hiVal = hiMinMax;
        const bool wantsClip = params_.clipLowPercent > 0.0 || params_.clipHighPercent < 100.0;
        if (hiMinMax > loMinMax && wantsClip) {
            constexpr int kHistBins = 65536;
            std::vector<uint64_t> hist(kHistBins, 0);
            const double range = hiMinMax - loMinMax;
            std::mutex histMutex;
            std::atomic<size_t> done{0};
            parallelForIndices(files.size(), [&](size_t i) {
                cv::Mat slice = readSlice(files[i]);
                applyBh(slice);
                // Build a local histogram per file (no synchronization needed for the hot
                // per-pixel loop), then merge once per file under the lock.
                std::vector<uint64_t> localHist(kHistBins, 0);
                for (int r = 0; r < slice.rows; ++r) {
                    const float* row = slice.ptr<float>(r);
                    for (int c = 0; c < slice.cols; ++c) {
                        int bin = static_cast<int>((row[c] - loMinMax) / range * (kHistBins - 1));
                        bin = std::clamp(bin, 0, kHistBins - 1);
                        localHist[static_cast<size_t>(bin)]++;
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(histMutex);
                    for (int b = 0; b < kHistBins; ++b)
                        hist[static_cast<size_t>(b)] += localHist[static_cast<size_t>(b)];
                }
                size_t d = done.fetch_add(1) + 1;
                emit progress(33 + static_cast<int>(33.0 * d / files.size()),
                              QString("Building histogram: %1/%2").arg(d).arg(files.size()));
            });

            uint64_t total = 0;
            for (uint64_t count : hist)
                total += count;
            const uint64_t loTarget = static_cast<uint64_t>(total * params_.clipLowPercent / 100.0);
            const uint64_t hiTarget = static_cast<uint64_t>(total * params_.clipHighPercent / 100.0);

            int loBin = 0, hiBin = kHistBins - 1;
            uint64_t cum = 0;
            for (int b = 0; b < kHistBins; ++b) {
                cum += hist[static_cast<size_t>(b)];
                if (cum >= loTarget) { loBin = b; break; }
            }
            cum = 0;
            for (int b = 0; b < kHistBins; ++b) {
                cum += hist[static_cast<size_t>(b)];
                if (cum >= hiTarget) { hiBin = b; break; }
            }

            loVal = loMinMax + (static_cast<double>(loBin) / (kHistBins - 1)) * range;
            hiVal = loMinMax + (static_cast<double>(hiBin) / (kHistBins - 1)) * range;
            if (hiVal <= loVal) {
                loVal = loMinMax;
                hiVal = hiMinMax;
            }
        }

        const double scale = (hiVal > loVal) ? (65535.0 / (hiVal - loVal)) : 1.0;
        std::atomic<size_t> doneWrite{0};
        parallelForIndices(files.size(), [&](size_t i) {
            cv::Mat slice = readSlice(files[i]);
            applyBh(slice);
            cv::Mat clipped;
            cv::min(slice, static_cast<float>(hiVal), clipped);
            cv::max(clipped, static_cast<float>(loVal), clipped);
            cv::Mat out16;
            clipped.convertTo(out16, CV_16UC1, scale, -loVal * scale);
            QString outPath = postDir + QString::fromStdString(files[i].filename().string());
            cv::imwrite(outPath.toStdString(), out16);
            size_t d = doneWrite.fetch_add(1) + 1;
            emit progress(66 + static_cast<int>(34.0 * d / files.size()),
                          QString("Writing 16-bit: %1/%2").arg(d).arg(files.size()));
        });

        emit finished();
    } catch (const std::exception& e) {
        emit failed(QString::fromStdString(e.what()));
    }
}
