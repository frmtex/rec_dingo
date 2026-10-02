#ifndef INMEMORY_PIPELINE_WORKER_H
#define INMEMORY_PIPELINE_WORKER_H
#include <QObject>
#include <vector>
#include "proj_correction.h"
#include "reconstruction_worker.h"

// Runs the sole reconstruction path: scan -> corrected-projections -> sinograms -> FBP -> reco/,
// keeping the corrected-projection and sinogram stages entirely in RAM instead of writing corr/
// and sino/ to disk (measured: on a typical dataset, e.g. 1201 angles x 1960 x 1960, the
// projection stack alone is ~18GB as float32 - trivial against a machine with enough RAM to run
// this app on such a dataset at all). scan/ob/di are still read from disk, and reco/ (and post/,
// via the existing separate Post Processing step) are still written to disk as the deliverable
// output. Optionally (Params::keepSlicesInRam, off by default) every reconstructed slice also stays
// resident in RAM afterward (see reconstructedSlices()), on top of the projection/sinogram RAM
// above, so Post Processing's histogram sampling can reuse them instead of reading reco/ back.
//
// Reuses ReconstructionWorker::Params (rather than a parallel struct) since both classes are part
// of the same overall pipeline (ReconstructionWorker::runPreview() handles the fast B/M/T preview;
// this handles the full run). For StartStage::CorrectedProjections, Stage 1 reads from
// Proj_correction::correctedProjection() (populated by a prior "Correct Scan") instead of
// recomputing the correction - run() throws if that cache is empty.
class InMemoryPipelineWorker : public QObject
{
    Q_OBJECT

public:
    // proj_correction is not owned and must remain valid (and untouched by other threads) for the
    // duration of run().
    InMemoryPipelineWorker(Proj_correction* proj_correction, ReconstructionWorker::Params params,
                            QObject* parent = nullptr);

    // Every reconstructed slice from the most recent run(), in row order - kept in RAM (alongside
    // the reco/ files run() still writes to disk) so a subsequent histogram sample can reuse them
    // instead of reading reco/ back off disk; see MainWindow::slot_load_reco_histogram(). Empty
    // unless Params::keepSlicesInRam was set, and until finished() has fired. Safe to read from another thread once finished() is delivered:
    // by then every row thread inside run() has already joined.
    const std::vector<cv::Mat>& reconstructedSlices() const { return reconstructedSlices_; }

public slots:
    void run();

signals:
    void progress(int percent, QString message);
    void finished();
    void failed(QString error);

private:
    Proj_correction* proj_;
    ReconstructionWorker::Params params_;
    std::vector<cv::Mat> reconstructedSlices_;
    // Same logic as ReconstructionWorker::buildAngles() (private there, so duplicated here rather
    // than shared - small enough that duplication is simpler than introducing a shared base for it).
    std::vector<double> buildAngles() const;
};

#endif // INMEMORY_PIPELINE_WORKER_H
