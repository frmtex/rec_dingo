#ifndef INMEMORY_PIPELINE_WORKER_H
#define INMEMORY_PIPELINE_WORKER_H
#include <QObject>
#include <vector>
#include "proj_correction.h"
#include "reconstruction_worker.h"

// Runs the same scan -> corrected-projections -> sinograms -> FBP -> reco/ pipeline as
// CorrScanWorker + ReconstructionWorker, but keeps the corrected-projection and sinogram stages
// entirely in RAM instead of writing corr/ and sino/ to disk - for large-RAM machines where the
// disk round-trip in the middle of the pipeline is pure overhead (measured: on a typical dataset,
// e.g. 1201 angles x 1960 x 1960, the projection stack alone is ~18GB as float32 - trivial against
// a 1TB machine, not something to default to generally). scan/ob/di are still read from disk, and
// reco/ (and post/, via the existing separate Post Processing step) are still written to disk as
// the deliverable output.
//
// Reuses ReconstructionWorker::Params (rather than a parallel struct) so this stays easy to unify
// with the disk-streaming path later behind a runtime toggle. Only StartStage::RawScan and
// StartStage::CorrectedProjections make sense here - there's no on-disk sino/ to resume from in a
// mode whose whole point is never writing it; run() throws if StartStage::Sinograms is requested.
class InMemoryPipelineWorker : public QObject
{
    Q_OBJECT

public:
    // proj_correction is not owned and must remain valid (and untouched by other threads) for the
    // duration of run().
    InMemoryPipelineWorker(Proj_correction* proj_correction, ReconstructionWorker::Params params,
                            QObject* parent = nullptr);

public slots:
    void run();

signals:
    void progress(int percent, QString message);
    void finished();
    void failed(QString error);

private:
    Proj_correction* proj_;
    ReconstructionWorker::Params params_;
    // Same logic as ReconstructionWorker::buildAngles() (private there, so duplicated here rather
    // than shared - small enough that duplication is simpler than introducing a shared base for it).
    std::vector<double> buildAngles() const;
};

#endif // INMEMORY_PIPELINE_WORKER_H
