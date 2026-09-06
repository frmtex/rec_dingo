#ifndef POST_PROCESS_WORKER_H
#define POST_PROCESS_WORKER_H
#include <QObject>
#include <QString>

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
    };

    explicit PostProcessWorker(Params params, QObject* parent = nullptr);

public slots:
    void run();

signals:
    void progress(int percent, QString message);
    void finished();
    void failed(QString error);

private:
    Params params_;
};

#endif // POST_PROCESS_WORKER_H
