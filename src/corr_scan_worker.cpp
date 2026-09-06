#include "corr_scan_worker.h"

CorrScanWorker::CorrScanWorker(Proj_correction* proj_correction, QObject* parent)
    : QObject(parent), proj_(proj_correction)
{
}

void CorrScanWorker::run()
{
    try {
        proj_->run_scan([this](int done, int total) {
            int pct = total > 0 ? static_cast<int>(100.0 * done / total) : 0;
            emit progress(pct, QString("Correcting projection %1/%2").arg(done).arg(total));
        });
        emit finished();
    } catch (const std::exception& e) {
        emit failed(QString::fromStdString(e.what()));
    }
}
