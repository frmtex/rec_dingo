#ifndef POST_PROCESS_WORKER_H
#define POST_PROCESS_WORKER_H
#include <QObject>
#include <QString>
#include <QVector>

// Histogram of (a sample of) the reconstructed slices, as shown next to the 16-bit conversion
// controls. counts are uniform bins spanning [lo, hi].
struct RecoHistogram
{
    QVector<double> counts;
    double lo = 0.0;
    double hi = 0.0;
    int slicesSampled = 0;
    int slicesTotal = 0;
};

// Runs on the already-reconstructed slices in reco/ (written by ReconstructionWorker::run()),
// writing results to post/ so the original float32 data is never overwritten and post-processing
// can be re-run with different settings without re-reconstructing. Runs on a worker thread.
class PostProcessWorker : public QObject
{
    Q_OBJECT

public:
    struct Params
    {
        QString workingPath; // reco/ read from workingPath+"/reco/", output written to workingPath+"/post/"

        bool beamHardeningEnabled = false;
        double bhC1 = 1.0; // mu_corrected = bhC1*mu + bhC2*mu^2 + bhC3*mu^3
        double bhC2 = 0.0;
        double bhC3 = 0.0;

        // Rescales to the full uint16 range [0, 65535]. The clip percentiles are taken from a
        // histogram built over the whole dataset (all slices, after beam-hardening correction if
        // enabled), not just the literal min/max, so a handful of outlier pixels don't compress
        // the useful range. Set to 0/100 for literal min/max with no clipping.
        bool convert16Enabled = false;
        double clipLowPercent = 0.01;
        double clipHighPercent = 99.99;

        // When set, the 16-bit conversion clips to exactly [rangeLow, rangeHigh] (in the same
        // units as the slices, after beam-hardening correction if enabled) instead of deriving the
        // range from clipLowPercent/clipHighPercent - which also skips the dataset-wide min/max and
        // histogram passes. Used when the range was picked on the histogram display.
        bool useAbsoluteRange = false;
        double rangeLow = 0.0;
        double rangeHigh = 0.0;

        // runHistogram() reads at most this many slices, evenly spaced across the dataset (always
        // including the first and last), into histogramBins bins - a full pass over every slice
        // would take as long as the conversion itself.
        int histogramMaxSlices = 48;
        int histogramBins = 4096;
    };

    explicit PostProcessWorker(Params params, QObject* parent = nullptr);

public slots:
    void run();
    // Emits histogramReady for a sample of reco/ (with beam-hardening correction applied first if
    // enabled) - see Params::histogramMaxSlices. Doesn't write anything.
    void runHistogram();

signals:
    void progress(int percent, QString message);
    void finished();
    void failed(QString error);
    void histogramReady(RecoHistogram histogram);

private:
    Params params_;
};

Q_DECLARE_METATYPE(RecoHistogram)

#endif // POST_PROCESS_WORKER_H
