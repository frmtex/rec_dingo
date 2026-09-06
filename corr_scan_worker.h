#ifndef CORR_SCAN_WORKER_H
#define CORR_SCAN_WORKER_H
#include <QObject>
#include <QString>
#include "proj_correction.h"

// Runs Proj_correction::run_scan() (flat-field + spot filter + phase retrieval over every raw
// projection, writing corr/) on a worker thread, reporting progress via signals - mirrors
// ReconstructionWorker's thread-and-signal pattern.
class CorrScanWorker : public QObject
{
    Q_OBJECT

public:
    // proj_correction is not owned and must remain valid (and untouched by other threads) for the
    // duration of run().
    explicit CorrScanWorker(Proj_correction* proj_correction, QObject* parent = nullptr);

public slots:
    void run();

signals:
    void progress(int percent, QString message);
    void finished();
    void failed(QString error);

private:
    Proj_correction* proj_;
};

#endif // CORR_SCAN_WORKER_H
