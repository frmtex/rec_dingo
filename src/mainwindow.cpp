#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QImage>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QPixmap>
#include <QRect>
#include <QPoint>
#include <QMouseEvent>
#include <QPainter>
#include <QLabel>
#include <QPushButton>
#include <QGraphicsPixmapItem>
#include <QSignalBlocker>
#include <QMap>
#include <QFile>
#include <QTextStream>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "customview.h"
#include "proj_correction.h"
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "rotation_axis.h"
#include "beam_hardening.h"
#include "ring_removal_polar.h"
#include "angle_file_reader.h"
#include "corr_scan_worker.h"
#include "histogram_widget.h"

namespace {
// Fits a 16-bit gray image inside maxW x maxH (keeping aspect ratio, enlarging if smaller) with
// full 16-bit precision. QImage::scaled(SmoothTransformation) on Grayscale16 data is NOT safe
// for this: on low-contrast dark images it collapsed every pixel to (nearly) the same gray value
// - measured: a 900x1200 image with mean 1050 and sigma 20 came out with a single distinct value
// - which destroys exactly the faint structure a display window is meant to bring out.
QImage fitGray16(const cv::Mat& gray, int maxW, int maxH)
{
    cv::Mat src16;
    if (gray.type() == CV_16UC1)
        src16 = gray;
    else
        gray.convertTo(src16, CV_16U, gray.depth() == CV_8U ? 257.0 : 1.0);

    const double scale = std::min(static_cast<double>(maxW) / src16.cols, static_cast<double>(maxH) / src16.rows);
    const cv::Size size(std::max(1, static_cast<int>(std::lround(src16.cols * scale))),
                        std::max(1, static_cast<int>(std::lround(src16.rows * scale))));
    cv::Mat dst;
    cv::resize(src16, dst, size, 0, 0, scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
    return QImage(dst.data, dst.cols, dst.rows, static_cast<int>(dst.step), QImage::Format_Grayscale16).copy();
}

QImage fitGray16(const QImage& gray16, int maxW, int maxH)
{
    cv::Mat wrapped(gray16.height(), gray16.width(), CV_16UC1, const_cast<uchar*>(gray16.constBits()),
                    static_cast<size_t>(gray16.bytesPerLine()));
    return fitGray16(wrapped, maxW, maxH);
}
} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

#if defined(__APPLE__)
    // No CUDA polar-ring backend on macOS (see reconstruction_worker.cpp's __APPLE__ guards) -
    // that path always runs on the CPU there, so disable the checkbox rather than let it look
    // like a live option that silently does nothing.
    ui->checkBox_ringUseGpu->setChecked(false);
    ui->checkBox_ringUseGpu->setEnabled(false);
    // Same reasoning for GPU spot filter/phase retrieval (see proj_correction.h's __APPLE__ guards).
    ui->checkBox_useGpuProjCorrection->setChecked(false);
    ui->checkBox_useGpuProjCorrection->setEnabled(false);
#endif

    connect(ui->pushButton_ini, SIGNAL(clicked()), this, SLOT(slotFileOpen()));
    connect(ui->pushButton_roi, SIGNAL(clicked()), this, SLOT(slotGetROI()));
    connect(ui->pushButton_corr, SIGNAL(clicked()), this, SLOT(slotGetCorr()));
    connect(ui->pushButton_first, SIGNAL(clicked()), this, SLOT(slot_First_Set()));
    connect(ui->pushButton_corr_scan, SIGNAL(clicked()), this, SLOT(slot_corr_scan()));
    connect(ui->pushButton_angleFileBrowse, SIGNAL(clicked()), this, SLOT(slot_browse_angle_file()));

    connect(ui->pushButton_addRoi, SIGNAL(clicked()), this, SLOT(slot_add_cor_roi()));
    connect(ui->pushButton_clearRoi, SIGNAL(clicked()), this, SLOT(slot_clear_cor_roi()));
    connect(ui->pushButton_findCor, SIGNAL(clicked()), this, SLOT(slot_find_cor()));
    connect(ui->pushButton_runReco, SIGNAL(clicked()), this, SLOT(slot_run_reconstruction()));

    connect(ui->pushButton_preview, SIGNAL(clicked()), this, SLOT(slot_run_preview()));
    connect(ui->comboBox_previewSlice, SIGNAL(currentIndexChanged(int)), this, SLOT(slot_show_preview_slice(int)));

    connect(ui->pushButton_runPostProcess, SIGNAL(clicked()), this, SLOT(slot_run_post_process()));

    setupDisplayHistogram();
    setupRecoHistogram();

    // Live-preview the beam-hardening correction on whichever B/M/T slice is showing, the same
    // way the ring-mask overlay updates live - no separate "preview" button needed.
    connect(ui->checkBox_bhEnable, &QCheckBox::toggled, this,
            [this](bool) { refresh_preview_overlay(); });
    connect(ui->doubleSpinBox_bhC1, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { refresh_preview_overlay(); });
    connect(ui->doubleSpinBox_bhC2, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { refresh_preview_overlay(); });
    connect(ui->doubleSpinBox_bhC3, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { refresh_preview_overlay(); });

    // Live-preview the post-reconstruction ring removal the same way, before beam hardening.
    connect(ui->checkBox_ringRemovalEnable, &QCheckBox::toggled, this,
            [this](bool) { refresh_preview_overlay(); });
    connect(ui->doubleSpinBox_ringThresh, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { refresh_preview_overlay(); });
    connect(ui->doubleSpinBox_ringThreshMax, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { refresh_preview_overlay(); });
    connect(ui->doubleSpinBox_ringThreshMin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { refresh_preview_overlay(); });
    connect(ui->spinBox_ringThetaMin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int) { refresh_preview_overlay(); });
    connect(ui->spinBox_ringWidth, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int) { refresh_preview_overlay(); });
    connect(ui->comboBox_ringBoundaryMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { refresh_preview_overlay(); });
    // Also affects the polar ring filter's processing radius when the wavelet mask isn't active
    // (see display_preview_slice()).
    connect(ui->doubleSpinBox_circMask, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { refresh_preview_overlay(); });

    // Live-redraw the mask overlay on whichever preview slice is showing when these change,
    // without needing to recompute the preview itself.
    connect(ui->spinBox_ringMaskInnerRadius, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int) { refresh_preview_overlay(); });
    connect(ui->spinBox_ringMaskOuterRadius, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int) { refresh_preview_overlay(); });
    connect(ui->checkBox_ringEnable, &QCheckBox::toggled, this,
            [this](bool) { refresh_preview_overlay(); });

    // Editing the CoR offset or tilt by hand only takes effect in "Manual value" mode (see
    // buildReconstructionParams()); auto-switch into it so a manual edit is never silently
    // ignored. slot_find_cor() blocks these signals around its own setValue() calls so running
    // Find CoR/Tilt doesn't itself flip you out of Manual mode.
    connect(ui->doubleSpinBox_corOffset, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { ui->radioButton_corManual->setChecked(true); });
    connect(ui->doubleSpinBox_tilt, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { ui->radioButton_corManual->setChecked(true); });
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::slotFileOpen()
{
  statusBar()->showMessage(tr("Opening file..."));

  QString fileName = QFileDialog::getOpenFileName(this,tr("choose a file"),"/media/ulg/Data",tr("*.ini"));
  if (!fileName.isEmpty())
  {
    workingpath = fileName;
    QFile file(fileName);
    if ( !file.open(QIODevice::ReadOnly) ) {
    return;
    }
    int test = fileName.lastIndexOf('/');
    workingpath.resize(test);
    ui->lineEdit_mainpath->setText(workingpath);
    invalidateRecoHistogram(tr("New dataset opened"));

    QTextStream steuerStream( &file);
    QString iniline;
    steuerelemente.clear(); // otherwise a second slotFileOpen() in the same session would append
                             // onto stale lines from the previous file, corrupting the fixed .at(N)
                             // indices below and confusing the sentinel search further down.
    while(!steuerStream.atEnd())steuerelemente << steuerStream.readLine();

    iniline = steuerelemente.at(1);
    iniline.remove("angles=");
    ui->label_angles->setText(iniline);
    angles = iniline.toInt();

    iniline = steuerelemente.at(2);
    iniline.remove("last_angle=");
    ui->label_last_angle->setText(iniline);
    last_angle = iniline.toFloat();

    iniline = steuerelemente.at(3);
    iniline.remove("corr_projection=");
    ui->label_corr_projection->setText(iniline);
    corr_projection = iniline.toInt();

    iniline = steuerelemente.at(4);
    iniline.remove("repeats=");
    ui->label_repeats->setText(iniline);
    repeats = iniline.toInt();

    iniline = steuerelemente.at(5);
    iniline.remove("pixel_size=");
    ui->label_pixel_size->setText(iniline);
    pixel_size = iniline.toInt();

    // Pre-fill (but don't force-enable) the angle file path if the SICS instrument log sits
    // alongside the scan, since that's where it's written in practice - the tick box still
    // decides whether it's actually used.
    QString defaultAngleFile = workingpath + "/sics_out.txt";
    if (QFile::exists(defaultAngleFile))
        ui->lineEdit_angleFile->setText(defaultAngleFile);

    // If this settings.ini was previously updated by saveSettingsIni() (see there for the sentinel
    // format), restore the reconstruction settings it recorded - lets a sinogram or corrected-
    // projection dataset be reloaded later and reconstructed again with the same starting point,
    // then tweaked from there.
    static const QString kSettingsSentinel =
        "# rec_dingo settings (auto-generated below - safe to delete, will be regenerated)";
    int sentinelIdx = steuerelemente.indexOf(kSettingsSentinel);
    if (sentinelIdx >= 0) {
        QMap<QString, QString> kv;
        for (int i = sentinelIdx + 1; i < steuerelemente.size(); ++i) {
            const QString& l = steuerelemente.at(i);
            int eq = l.indexOf('=');
            if (eq > 0)
                kv[l.left(eq)] = l.mid(eq + 1);
        }
        auto getInt = [&](const QString& k, int def) { return kv.contains(k) ? kv.value(k).toInt() : def; };
        auto getDouble = [&](const QString& k, double def) { return kv.contains(k) ? kv.value(k).toDouble() : def; };
        auto getBool = [&](const QString& k, bool def) { return kv.contains(k) ? kv.value(k).toInt() != 0 : def; };

        QRect roi(getInt("roi_x", 0), getInt("roi_y", 0), getInt("roi_w", 0), getInt("roi_h", 0));
        if (roi.width() > 0 && roi.height() > 0)
            rect_roi_final = roi;

        ui->comboBox_binning->setCurrentIndex(getInt("binning", 0));
        ui->comboBox_spotKernel->setCurrentIndex(getInt("spot_kernel_index", 1));
        ui->checkBox_useAngleFile->setChecked(getBool("use_angle_file", false));
        if (kv.contains("angle_file_path"))
            ui->lineEdit_angleFile->setText(kv.value("angle_file_path"));

        // Restored as the resolved numeric value, in Manual mode - simpler and more robust than
        // trying to reproduce the ROI-based Find CoR/Tilt workflow that originally computed it.
        if (kv.contains("tilt_deg")) {
            ui->doubleSpinBox_tilt->setValue(getDouble("tilt_deg", 0.0));
            ui->radioButton_corManual->setChecked(true);
        }
        if (kv.contains("cor_offset")) {
            ui->doubleSpinBox_corOffset->setValue(getDouble("cor_offset", 0.0));
            ui->radioButton_corManual->setChecked(true);
        }

        ui->checkBox_ringEnable->setChecked(getBool("ring_enabled", true));
        ui->spinBox_ringLevel->setValue(getInt("ring_level", 3));
        ui->doubleSpinBox_ringSigma->setValue(getDouble("ring_sigma", 2.0));
        ui->spinBox_ringOrder->setValue(getInt("ring_order", 3));
        ui->spinBox_ringPad->setValue(getInt("ring_pad", 200));
        ui->spinBox_ringMaskInnerRadius->setValue(getInt("ring_mask_inner", 0));
        ui->spinBox_ringMaskOuterRadius->setValue(getInt("ring_mask_outer", 0));

        ui->checkBox_ringRemovalEnable->setChecked(getBool("polar_ring_enabled", false));
        ui->doubleSpinBox_ringThresh->setValue(getDouble("polar_ring_thresh", 300.0));
        ui->doubleSpinBox_ringThreshMax->setValue(getDouble("polar_ring_thresh_max", 300.0));
        ui->doubleSpinBox_ringThreshMin->setValue(getDouble("polar_ring_thresh_min", -100.0));
        ui->spinBox_ringThetaMin->setValue(getInt("polar_ring_theta_min", 30));
        ui->spinBox_ringWidth->setValue(getInt("polar_ring_width", 30));
        ui->comboBox_ringBoundaryMode->setCurrentIndex(getBool("polar_ring_wrap", true) ? 0 : 1);
#if !defined(__APPLE__)
        ui->checkBox_ringUseGpu->setChecked(getBool("polar_ring_use_gpu", false));
#endif

        ui->comboBox_fbpFilter->setCurrentIndex(getInt("fbp_filter", 3));
        ui->comboBox_algorithm->setCurrentIndex(getInt("recon_algorithm", 0));
        ui->doubleSpinBox_circMask->setValue(getDouble("circ_mask_ratio", 0.99));
    }

    openImage();

    setWindowTitle(fileName);
    QString message=tr("Loaded document: ")+fileName;
    statusBar()->showMessage(message, 2000);
  }
  else
  {
    statusBar()->showMessage(tr("Opening aborted"), 2000);
  }
}

void MainWindow::slot_browse_angle_file()
{
    QString startDir = workingpath.isEmpty() ? QString("/Users/ugarbe/Desktop/test_data") : workingpath;
    QString fileName = QFileDialog::getOpenFileName(this, tr("Choose angle log file"), startDir, tr("*.txt"));
    if (!fileName.isEmpty())
        ui->lineEdit_angleFile->setText(fileName);
}

void MainWindow::slotGetROI()
{
    QRect roi = ui->graphicsView->roiInImageCoords();
    qDebug() << "roi (image coords):" << roi;

    ui->x1_roi->setText("X1=" + QString::number(roi.x()));
    ui->x2_roi->setText("Y1=" + QString::number(roi.y()));
    ui->y1_roi->setText("W=" + QString::number(roi.width()));
    ui->y2_roi->setText("H=" + QString::number(roi.height()));

    rect_roi_final = roi;
    update_view();
}

void MainWindow::slotGetCorr()
{
    if (!first_set) {
        statusBar()->showMessage(tr("Load first set before picking the intensity-correction ROI"), 3000);
        return;
    }

    QRect roi = ui->graphicsView->roiInImageCoords();

    ui->x1_corr->setText("X1c=" + QString::number(roi.left()));
    ui->y1_corr->setText("Y1c=" + QString::number(roi.top()));
    ui->x2_corr->setText("X2c=" + QString::number(roi.right()));
    ui->y2_corr->setText("Y2c=" + QString::number(roi.bottom()));

    try {
        first_set->setIntensityRoi(roi);
        // Changes what get_projection_corr() returns for every index, so any cached preview
        // sinograms built before this point are now stale.
        preview_sino_cache = ReconstructionWorker::PreviewCache();
        statusBar()->showMessage(tr("Intensity-correction ROI set (reference taken from projection 0)"), 4000);
    } catch (const std::exception& e) {
        statusBar()->showMessage(tr("Failed to set intensity-correction ROI: ") + QString(e.what()), 5000);
    }
}

void MainWindow::openImage()
{

    QDir scan_dir = workingpath + "/scan/";
    QStringList proj_list =  scan_dir.entryList(QDir::Files | QDir::NoDotAndDotDot);

    QString first_image = workingpath + "/scan/" + proj_list.at(0);

    qDebug() << first_image;
    cv::Mat image = cv::imread(first_image.toStdString(), cv::IMREAD_UNCHANGED);

    // Verify the file loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return;
    }


    org_cols = image.cols;
    org_rows = image.rows;

    scaledImage = fitGray16(image, 600, 800);
    ui->graphicsView->setDisplayedImageSize(scaledImage.size(), QSize(org_cols, org_rows));

    showScaledImage(true);
    ui->graphicsView->show();

}

void MainWindow::update_view()
{

    QRect rect_roi;
    rect_roi = ui->graphicsView->rect_final.toRect();

    QImage roi_image = scaledImage.copy(rect_roi);

    float image_ratio;
    image_ratio = static_cast<float>(roi_image.height())/static_cast<float>(roi_image.width());

    qDebug() << "ratio" << image_ratio;

    scaledImage = fitGray16(roi_image, 600, 800);
    ui->graphicsView->setDisplayedImageSize(scaledImage.size(), rect_roi.size());

    showScaledImage(false);
    ui->graphicsView->show();

}

void MainWindow::slot_First_Set()
{
    // Loading a fresh Proj_correction changes the ob/di flat-field data underneath any
    // previously-cached preview sinograms, even if their key fields still match.
    preview_sino_cache = ReconstructionWorker::PreviewCache();

    first_set = new Proj_correction();
    first_set->setData(workingpath,rect_roi_final);
    first_set->setSpotKernelSize(spot_kernel_size_from_ui());
    first_set->setBinning(ui->comboBox_binning->currentIndex() + 1);
    first_set->setUseGpu(ui->checkBox_useGpuProjCorrection->isChecked());
    first_set->load_op_di();
    first_set->get_first_image_corr();

    scaledImage = fitGray16(first_set->im_show, 600, 800);
    ui->graphicsView->setDisplayedImageSize(scaledImage.size(), QSize(first_set->im_show.cols, first_set->im_show.rows));

    showScaledImage(true);
    ui->graphicsView->show();


}

void MainWindow::slot_corr_scan()
{
    if (!first_set) {
        statusBar()->showMessage(tr("Load first set before correcting the scan"), 3000);
        return;
    }
    if (reco_thread || post_thread || corr_scan_thread) {
        statusBar()->showMessage(tr("Another operation is already running"), 3000);
        return;
    }

    // run_scan() rewrites corr/ on disk, which a cached preview built with
    // StartStage::CorrectedProjections would otherwise reuse stale data from.
    preview_sino_cache = ReconstructionWorker::PreviewCache();

    // Re-apply in case any of these were changed after "Load first set" ran.
    first_set->setSpotKernelSize(spot_kernel_size_from_ui());
    first_set->setBinning(ui->comboBox_binning->currentIndex() + 1);
    first_set->setUseGpu(ui->checkBox_useGpuProjCorrection->isChecked());

    ui->pushButton_corr_scan->setEnabled(false);
    ui->progressBar_reco->setValue(0);
    ui->label_corr_scan_status->setText(tr("Starting..."));

    corr_scan_thread = new QThread(this);
    corr_scan_worker = new CorrScanWorker(first_set);
    corr_scan_worker->moveToThread(corr_scan_thread);

    connect(corr_scan_thread, &QThread::started, corr_scan_worker, &CorrScanWorker::run);
    connect(corr_scan_worker, &CorrScanWorker::progress, this, &MainWindow::slot_corr_scan_progress);
    connect(corr_scan_worker, &CorrScanWorker::finished, this, &MainWindow::slot_corr_scan_finished);
    connect(corr_scan_worker, &CorrScanWorker::failed, this, &MainWindow::slot_corr_scan_failed);
    connect(corr_scan_worker, &CorrScanWorker::finished, corr_scan_thread, &QThread::quit);
    connect(corr_scan_worker, &CorrScanWorker::failed, corr_scan_thread, &QThread::quit);
    connect(corr_scan_thread, &QThread::finished, corr_scan_worker, &QObject::deleteLater);
    connect(corr_scan_thread, &QThread::finished, corr_scan_thread, &QObject::deleteLater);
    connect(corr_scan_thread, &QThread::finished, this, [this]() {
        corr_scan_thread = nullptr;
        corr_scan_worker = nullptr;
    });

    corr_scan_thread->start();
}

void MainWindow::slot_corr_scan_progress(int percent, QString message)
{
    ui->progressBar_reco->setValue(percent);
    ui->label_corr_scan_status->setText(message);
}

void MainWindow::slot_corr_scan_finished()
{
    ui->label_corr_scan_status->setText(tr("Correct Scan complete"));
    ui->pushButton_corr_scan->setEnabled(true);
    saveSettingsIni();

    scaledImage = fitGray16(first_set->im_show, 600, 800);
    ui->graphicsView->setDisplayedImageSize(scaledImage.size(), QSize(first_set->im_show.cols, first_set->im_show.rows));

    showScaledImage(true);
    ui->graphicsView->show();
}

void MainWindow::slot_corr_scan_failed(QString error)
{
    ui->label_corr_scan_status->setText(tr("Failed: ") + error);
    statusBar()->showMessage(tr("Correct Scan failed: ") + error, 8000);
    ui->pushButton_corr_scan->setEnabled(true);
}

void MainWindow::slot_add_cor_roi()
{
    QRect roi = ui->graphicsView->roiInImageCoords();

    cor_rois.push_back(cv::Rect(0, roi.y(), 1, roi.height()));

    QString message = tr("CoR ROI added: y=%1..%2 (%3 total)")
                           .arg(roi.y()).arg(roi.y() + roi.height()).arg(cor_rois.size());
    statusBar()->showMessage(message, 3000);
}

void MainWindow::slot_clear_cor_roi()
{
    cor_rois.clear();
    statusBar()->showMessage(tr("CoR ROIs cleared"), 2000);
}

void MainWindow::slot_find_cor()
{
    if (!first_set) {
        statusBar()->showMessage(tr("Load first set before finding the rotation axis"), 3000);
        return;
    }
    if (cor_rois.empty()) {
        statusBar()->showMessage(tr("Add at least one ROI before finding the rotation axis"), 3000);
        return;
    }

    statusBar()->showMessage(tr("Finding rotation axis..."));

    // Match whatever binning reconstruction will use, so the resulting offset/tilt are already
    // in the right pixel domain and don't need any rescaling later.
    first_set->setBinning(ui->comboBox_binning->currentIndex() + 1);

    cv::Mat proj0 = first_set->get_projection_corr(0);
    int idx180 = (corr_projection == angles) ? (corr_projection - 1) : corr_projection;
    cv::Mat proj180 = first_set->get_projection_corr(idx180);

    try {
        CorResult result = RotationAxis::find_axis(proj0, proj180, cor_rois, /*ystep=*/5);
        cor_offset = result.offset;
        tilt_deg = result.tilt_deg;

        {
            // Block signals so this programmatic update doesn't itself trigger the
            // switch-to-Manual-mode handler on doubleSpinBox_corOffset/_tilt below.
            const QSignalBlocker blockOffset(ui->doubleSpinBox_corOffset);
            const QSignalBlocker blockTilt(ui->doubleSpinBox_tilt);
            ui->doubleSpinBox_corOffset->setValue(cor_offset);
            ui->doubleSpinBox_tilt->setValue(tilt_deg);
        }

        QString message = tr("Rotation axis found: offset=%1 px, tilt=%2 deg")
                               .arg(cor_offset).arg(tilt_deg);
        statusBar()->showMessage(message, 5000);
    } catch (const std::exception& e) {
        statusBar()->showMessage(tr("Failed to find rotation axis: ") + QString(e.what()), 5000);
    }
}

int MainWindow::spot_kernel_size_from_ui() const
{
    switch (ui->comboBox_spotKernel->currentIndex()) {
    case 0: return 3;
    case 2: return 7;
    default: return 5;
    }
}

void MainWindow::saveSettingsIni() const
{
    if (workingpath.isEmpty())
        return;

    static const QString kSettingsSentinel =
        "# rec_dingo settings (auto-generated below - safe to delete, will be regenerated)";

    QString path = workingpath + "/settings.ini";
    QStringList lines;
    QFile inFile(path);
    if (inFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&inFile);
        while (!in.atEnd())
            lines << in.readLine();
        inFile.close();
    }

    // Preserve everything above the sentinel verbatim (the scan-metadata lines the acquisition
    // software wrote - angles=/last_angle=/etc. - which this app only ever reads, never edits);
    // drop and regenerate everything from a previous save of this section, if present.
    int sentinelIdx = lines.indexOf(kSettingsSentinel);
    if (sentinelIdx >= 0)
        lines = lines.mid(0, sentinelIdx);

    lines << kSettingsSentinel;
    lines << QString("roi_x=%1").arg(rect_roi_final.x());
    lines << QString("roi_y=%1").arg(rect_roi_final.y());
    lines << QString("roi_w=%1").arg(rect_roi_final.width());
    lines << QString("roi_h=%1").arg(rect_roi_final.height());
    lines << QString("binning=%1").arg(ui->comboBox_binning->currentIndex());
    lines << QString("spot_kernel_index=%1").arg(ui->comboBox_spotKernel->currentIndex());
    lines << QString("use_angle_file=%1").arg(ui->checkBox_useAngleFile->isChecked() ? 1 : 0);
    lines << QString("angle_file_path=%1").arg(ui->lineEdit_angleFile->text());
    lines << QString("tilt_deg=%1").arg(
        ui->radioButton_corAuto->isChecked() ? tilt_deg : ui->doubleSpinBox_tilt->value(), 0, 'g', 10);
    lines << QString("cor_offset=%1").arg(
        ui->radioButton_corAuto->isChecked() ? cor_offset : ui->doubleSpinBox_corOffset->value(), 0, 'g', 10);
    lines << QString("ring_enabled=%1").arg(ui->checkBox_ringEnable->isChecked() ? 1 : 0);
    lines << QString("ring_level=%1").arg(ui->spinBox_ringLevel->value());
    lines << QString("ring_sigma=%1").arg(ui->doubleSpinBox_ringSigma->value(), 0, 'g', 10);
    lines << QString("ring_order=%1").arg(ui->spinBox_ringOrder->value());
    lines << QString("ring_pad=%1").arg(ui->spinBox_ringPad->value());
    lines << QString("ring_mask_inner=%1").arg(ui->spinBox_ringMaskInnerRadius->value());
    lines << QString("ring_mask_outer=%1").arg(ui->spinBox_ringMaskOuterRadius->value());
    lines << QString("polar_ring_enabled=%1").arg(ui->checkBox_ringRemovalEnable->isChecked() ? 1 : 0);
    lines << QString("polar_ring_thresh=%1").arg(ui->doubleSpinBox_ringThresh->value(), 0, 'g', 10);
    lines << QString("polar_ring_thresh_max=%1").arg(ui->doubleSpinBox_ringThreshMax->value(), 0, 'g', 10);
    lines << QString("polar_ring_thresh_min=%1").arg(ui->doubleSpinBox_ringThreshMin->value(), 0, 'g', 10);
    lines << QString("polar_ring_theta_min=%1").arg(ui->spinBox_ringThetaMin->value());
    lines << QString("polar_ring_width=%1").arg(ui->spinBox_ringWidth->value());
    lines << QString("polar_ring_wrap=%1").arg(ui->comboBox_ringBoundaryMode->currentIndex() == 0 ? 1 : 0);
    lines << QString("polar_ring_use_gpu=%1").arg(ui->checkBox_ringUseGpu->isChecked() ? 1 : 0);
    lines << QString("fbp_filter=%1").arg(ui->comboBox_fbpFilter->currentIndex());
    lines << QString("recon_algorithm=%1").arg(ui->comboBox_algorithm->currentIndex());
    lines << QString("circ_mask_ratio=%1").arg(ui->doubleSpinBox_circMask->value(), 0, 'g', 10);

    QFile outFile(path);
    if (!outFile.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate))
        return; // best-effort - a failed settings save shouldn't be treated as a run failure
    QTextStream out(&outFile);
    for (const QString& l : lines)
        out << l << "\n";
}

ReconstructionWorker::Params MainWindow::buildReconstructionParams()
{
    // Re-apply in case the spot-filter window or binning was changed after "Load first set" ran.
    first_set->setSpotKernelSize(spot_kernel_size_from_ui());
    first_set->setBinning(ui->comboBox_binning->currentIndex() + 1);

    ReconstructionWorker::Params params;
    params.workingPath = workingpath;
    switch (ui->comboBox_startStage->currentIndex()) {
    case 1: params.startStage = ReconstructionWorker::StartStage::CorrectedProjections; break;
    case 2: params.startStage = ReconstructionWorker::StartStage::Sinograms; break;
    default: params.startStage = ReconstructionWorker::StartStage::RawScan; break;
    }
    params.n_angles = angles;
    params.rotationStartDeg = 0.0;
    params.lastAngleDeg = last_angle;
    params.roiRect = rect_roi_final;
    params.tiltDeg = ui->radioButton_corAuto->isChecked() ? tilt_deg : ui->doubleSpinBox_tilt->value();
    params.corOffset = ui->radioButton_corAuto->isChecked() ? cor_offset : ui->doubleSpinBox_corOffset->value();
    params.ringEnabled = ui->checkBox_ringEnable->isChecked();
    params.ringLevel = ui->spinBox_ringLevel->value();
    params.ringSigma = ui->doubleSpinBox_ringSigma->value();
    params.ringOrder = ui->spinBox_ringOrder->value();
    params.ringPad = ui->spinBox_ringPad->value();
    params.ringMaskInnerRadius = ui->spinBox_ringMaskInnerRadius->value();
    params.ringMaskOuterRadius = ui->spinBox_ringMaskOuterRadius->value();
    params.polarRingEnabled = ui->checkBox_ringRemovalEnable->isChecked();
    params.polarRingThresh = ui->doubleSpinBox_ringThresh->value();
    params.polarRingThreshMax = ui->doubleSpinBox_ringThreshMax->value();
    params.polarRingThreshMin = ui->doubleSpinBox_ringThreshMin->value();
    params.polarRingThetaMinDeg = ui->spinBox_ringThetaMin->value();
    params.polarRingWidth = ui->spinBox_ringWidth->value();
    params.polarRingWrapBoundary = ui->comboBox_ringBoundaryMode->currentIndex() == 0;
    params.polarRingUseGpu = ui->checkBox_ringUseGpu->isChecked();
    switch (ui->comboBox_fbpFilter->currentIndex()) {
    case 0: params.fbpFilter = FbpFilterType::RamLak; break;
    case 1: params.fbpFilter = FbpFilterType::SheppLogan; break;
    case 2: params.fbpFilter = FbpFilterType::Cosine; break;
    case 4: params.fbpFilter = FbpFilterType::Hann; break;
    default: params.fbpFilter = FbpFilterType::Hamming; break;
    }
    switch (ui->comboBox_algorithm->currentIndex()) {
    case 1: params.algorithm = ReconstructionWorker::ReconAlgorithm::Gridrec; break;
    default: params.algorithm = ReconstructionWorker::ReconAlgorithm::Fbp; break;
    }
    params.circMaskRatio = ui->doubleSpinBox_circMask->value();
    params.binning = ui->comboBox_binning->currentIndex() + 1;

    if (ui->checkBox_useAngleFile->isChecked()) {
        try {
            params.anglesDeg = readAnglesFromSicsFile(ui->lineEdit_angleFile->text());
            if (static_cast<int>(params.anglesDeg.size()) != params.n_angles)
                throw std::runtime_error(
                    QString("file has %1 angles, expected %2")
                        .arg(params.anglesDeg.size())
                        .arg(params.n_angles)
                        .toStdString());
            params.useAngleFile = true;
        } catch (const std::exception& e) {
            // Fall back to the evenly-spaced angles already set above rather than failing the
            // whole run - the file is wrong/missing, but that's not a reason to block reconstruction.
            statusBar()->showMessage(tr("Angle file error, using evenly-spaced angles instead: ") +
                                          QString(e.what()),
                                      8000);
            params.useAngleFile = false;
        }
    }

    return params;
}

void MainWindow::slot_run_reconstruction()
{
    if (!first_set) {
        statusBar()->showMessage(tr("Load first set before running the reconstruction"), 3000);
        return;
    }
    if (reco_thread || post_thread || corr_scan_thread) {
        statusBar()->showMessage(tr("A reconstruction, preview, or post-processing run is already in progress"), 3000);
        return;
    }

    // A full run rewrites sino/ on disk, which a cached preview built with StartStage::Sinograms
    // would otherwise reuse stale in-memory data instead of.
    preview_sino_cache = ReconstructionWorker::PreviewCache();

    ReconstructionWorker::Params params = buildReconstructionParams();

    set_reconstruction_controls_enabled(false);
    ui->progressBar_reco->setValue(0);
    ui->label_reco_status->setText(tr("Starting..."));

    reco_thread = new QThread(this);
    reco_worker = new ReconstructionWorker(first_set, params, ReconstructionWorker::PreviewCache());
    reco_worker->moveToThread(reco_thread);

    connect(reco_thread, &QThread::started, reco_worker, &ReconstructionWorker::run);
    connect(reco_worker, &ReconstructionWorker::progress, this, &MainWindow::slot_reco_progress);
    connect(reco_worker, &ReconstructionWorker::finished, this, &MainWindow::slot_reco_finished);
    connect(reco_worker, &ReconstructionWorker::failed, this, &MainWindow::slot_reco_failed);
    connect(reco_worker, &ReconstructionWorker::finished, reco_thread, &QThread::quit);
    connect(reco_worker, &ReconstructionWorker::failed, reco_thread, &QThread::quit);
    connect(reco_thread, &QThread::finished, reco_worker, &QObject::deleteLater);
    connect(reco_thread, &QThread::finished, reco_thread, &QObject::deleteLater);
    connect(reco_thread, &QThread::finished, this, [this]() {
        reco_thread = nullptr;
        reco_worker = nullptr;
    });

    reco_thread->start();
}

void MainWindow::slot_reco_progress(int percent, QString message)
{
    ui->progressBar_reco->setValue(percent);
    ui->label_reco_status->setText(message);
}

void MainWindow::slot_reco_finished()
{
    ui->label_reco_status->setText(tr("Reconstruction complete"));
    set_reconstruction_controls_enabled(true);
    saveSettingsIni();
    invalidateRecoHistogram(tr("reco/ was regenerated"));
}

void MainWindow::slot_reco_failed(QString error)
{
    ui->label_reco_status->setText(tr("Failed: ") + error);
    statusBar()->showMessage(tr("Reconstruction failed: ") + error, 8000);
    set_reconstruction_controls_enabled(true);
}

void MainWindow::set_reconstruction_controls_enabled(bool enabled)
{
    ui->pushButton_runReco->setEnabled(enabled);
    ui->pushButton_findCor->setEnabled(enabled);
    ui->pushButton_addRoi->setEnabled(enabled);
    ui->pushButton_clearRoi->setEnabled(enabled);
    ui->pushButton_preview->setEnabled(enabled);
}

void MainWindow::slot_run_preview()
{
    if (!first_set) {
        statusBar()->showMessage(tr("Load first set before previewing"), 3000);
        return;
    }
    if (reco_thread || post_thread || corr_scan_thread) {
        statusBar()->showMessage(tr("A reconstruction, preview, or post-processing run is already in progress"), 3000);
        return;
    }

    ReconstructionWorker::Params params = buildReconstructionParams();

    set_reconstruction_controls_enabled(false);
    ui->comboBox_previewSlice->setEnabled(false);
    ui->progressBar_reco->setValue(0);
    ui->label_reco_status->setText(tr("Starting preview..."));

    reco_thread = new QThread(this);
    reco_worker = new ReconstructionWorker(first_set, params, preview_sino_cache);
    reco_worker->moveToThread(reco_thread);

    connect(reco_thread, &QThread::started, reco_worker, &ReconstructionWorker::runPreview);
    connect(reco_worker, &ReconstructionWorker::progress, this, &MainWindow::slot_reco_progress);
    connect(reco_worker, &ReconstructionWorker::previewFinished, this, &MainWindow::slot_preview_ready);
    connect(reco_worker, &ReconstructionWorker::failed, this, &MainWindow::slot_reco_failed);
    connect(reco_worker, &ReconstructionWorker::previewFinished, reco_thread, &QThread::quit);
    connect(reco_worker, &ReconstructionWorker::failed, reco_thread, &QThread::quit);
    connect(reco_thread, &QThread::finished, reco_worker, &QObject::deleteLater);
    connect(reco_thread, &QThread::finished, reco_thread, &QObject::deleteLater);
    connect(reco_thread, &QThread::finished, this, [this]() {
        reco_thread = nullptr;
        reco_worker = nullptr;
    });

    reco_thread->start();
}

void MainWindow::slot_preview_ready(cv::Mat bottom, cv::Mat mid, cv::Mat top, ReconstructionWorker::PreviewCache cache)
{
    preview_slices[0] = bottom;
    preview_slices[1] = mid;
    preview_slices[2] = top;
    preview_sino_cache = cache;

    ui->label_reco_status->setText(tr("Preview ready"));
    set_reconstruction_controls_enabled(true);

    ui->comboBox_previewSlice->setEnabled(true);
    ui->comboBox_previewSlice->setCurrentIndex(1); // default to Mid
    display_preview_slice(1);
}

void MainWindow::slot_show_preview_slice(int index)
{
    display_preview_slice(index);
}

QImage MainWindow::matToPreviewImage(const cv::Mat& slice) const
{
    if (slice.empty())
        return QImage();

    CV_Assert(slice.type() == CV_32FC1);
    std::vector<float> vals;
    vals.reserve(static_cast<size_t>(slice.total()));
    if (slice.isContinuous()) {
        const float* data = slice.ptr<float>(0);
        vals.assign(data, data + slice.total());
    } else {
        for (int r = 0; r < slice.rows; ++r) {
            const float* row = slice.ptr<float>(r);
            vals.insert(vals.end(), row, row + slice.cols);
        }
    }

    // Robust auto-contrast: clip to the 1st/99th percentile so a handful of outlier pixels
    // (e.g. reconstruction edge artifacts) don't wash out the rest of the image.
    size_t loIdx = static_cast<size_t>(vals.size() * 0.01);
    size_t hiIdx = std::min(vals.size() - 1, static_cast<size_t>(vals.size() * 0.99));
    std::nth_element(vals.begin(), vals.begin() + loIdx, vals.end());
    float lo = vals[loIdx];
    std::nth_element(vals.begin(), vals.begin() + hiIdx, vals.end());
    float hi = vals[hiIdx];
    if (hi <= lo)
        hi = lo + 1.0f;

    cv::Mat norm;
    slice.convertTo(norm, CV_8UC1, 255.0 / (hi - lo), -lo * 255.0 / (hi - lo));

    QImage img(norm.data, norm.cols, norm.rows, static_cast<int>(norm.step), QImage::Format_Grayscale8);
    return img.copy(); // deep copy - norm is about to go out of scope
}

void MainWindow::display_preview_slice(int index)
{
    if (index < 0 || index > 2 || preview_slices[index].empty())
        return;

    cv::Mat displaySlice = preview_slices[index];
    if (ui->checkBox_ringRemovalEnable->isChecked()) {
        // Mirror buildReconstructionParams()'s choice of polar-ring radius cap, so the live
        // preview matches what the real run will actually do.
        double maskRatio = ui->doubleSpinBox_circMask->value();
        int maskOuter = ui->spinBox_ringMaskOuterRadius->value();
        int maskInner = ui->spinBox_ringMaskInnerRadius->value();
        if (ui->checkBox_ringEnable->isChecked() && maskOuter > maskInner && maskOuter > 0
            && displaySlice.cols > 0)
            maskRatio = maskOuter / (displaySlice.cols / 2.0);

        displaySlice = PolarRingRemoval::remove_ring(
            displaySlice, ui->doubleSpinBox_ringThresh->value(), ui->doubleSpinBox_ringThreshMax->value(),
            ui->doubleSpinBox_ringThreshMin->value(), ui->spinBox_ringThetaMin->value(),
            ui->spinBox_ringWidth->value(), ui->comboBox_ringBoundaryMode->currentIndex() == 0,
            /*parallel=*/true, maskRatio);
    }
    if (ui->checkBox_bhEnable->isChecked())
        displaySlice = BeamHardening::apply(displaySlice, ui->doubleSpinBox_bhC1->value(),
                                             ui->doubleSpinBox_bhC2->value(), ui->doubleSpinBox_bhC3->value());

    QImage img = matToPreviewImage(displaySlice);
    QImage scaled = img.scaled(600, 800, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QImage rgb = scaled.convertToFormat(QImage::Format_RGB32);

    int innerR = ui->spinBox_ringMaskInnerRadius->value();
    int outerR = ui->spinBox_ringMaskOuterRadius->value();
    if (ui->checkBox_ringEnable->isChecked() && outerR > 0 && outerR > innerR && img.width() > 0) {
        // reconstruct_slice() always produces a square (n_cols x n_cols) image centered on the
        // rotation axis, so the mask - also centered there - scales by the same factor as the
        // display downscale, regardless of the ring filter's other settings.
        double scaleFactor = static_cast<double>(rgb.width()) / img.width();
        QPainter painter(&rgb);
        painter.setRenderHint(QPainter::Antialiasing);
        QPointF center(rgb.width() / 2.0, rgb.height() / 2.0);
        QPen pen(Qt::red);
        pen.setWidth(2);
        painter.setPen(pen);
        if (innerR > 0) {
            double r = innerR * scaleFactor;
            painter.drawEllipse(center, r, r);
        }
        double rOuter = outerR * scaleFactor;
        painter.drawEllipse(center, rOuter, rOuter);
    }

    disableDisplayHistogram();
    if (scene)
        scene->clear();
    scene = new QGraphicsScene(this);
    scene->addPixmap(QPixmap::fromImage(rgb));
    ui->graphicsView->resetRoi();
    ui->graphicsView->setScene(scene);
    ui->graphicsView->show();

    static const char* labels[3] = { "Bottom", "Mid", "Top" };
    statusBar()->showMessage(tr("Preview: %1 slice").arg(labels[index]), 3000);
}

void MainWindow::refresh_preview_overlay()
{
    int index = ui->comboBox_previewSlice->currentIndex();
    if (index >= 0 && index < 3 && !preview_slices[index].empty())
        display_preview_slice(index);
}

PostProcessWorker::Params MainWindow::buildPostProcessParams() const
{
    PostProcessWorker::Params params;
    params.workingPath = workingpath;
    params.beamHardeningEnabled = ui->checkBox_bhEnable->isChecked();
    params.bhC1 = ui->doubleSpinBox_bhC1->value();
    params.bhC2 = ui->doubleSpinBox_bhC2->value();
    params.bhC3 = ui->doubleSpinBox_bhC3->value();
    params.convert16Enabled = ui->checkBox_16bitEnable->isChecked();
    params.clipLowPercent = ui->doubleSpinBox_clipLow->value();
    params.clipHighPercent = ui->doubleSpinBox_clipHigh->value();
    if (recoHist_ && recoHist_->hasHistogram()) {
        params.useAbsoluteRange = true;
        params.rangeLow = recoHist_->rangeLow();
        params.rangeHigh = recoHist_->rangeHigh();
    }
    return params;
}

void MainWindow::slot_run_post_process()
{
    if (workingpath.isEmpty()) {
        statusBar()->showMessage(tr("Load a dataset before running post-processing"), 3000);
        return;
    }
    if (reco_thread || post_thread || corr_scan_thread) {
        statusBar()->showMessage(tr("A reconstruction, preview, or post-processing run is already in progress"), 3000);
        return;
    }
    if (!ui->checkBox_bhEnable->isChecked() && !ui->checkBox_16bitEnable->isChecked()) {
        statusBar()->showMessage(tr("Enable beam hardening correction and/or 16-bit conversion first"), 3000);
        return;
    }

    PostProcessWorker::Params params = buildPostProcessParams();

    ui->pushButton_runPostProcess->setEnabled(false);
    loadHistButton_->setEnabled(false);
    ui->progressBar_reco->setValue(0);
    ui->label_post_status->setText(tr("Starting..."));

    post_thread = new QThread(this);
    post_worker = new PostProcessWorker(params);
    post_worker->moveToThread(post_thread);

    connect(post_thread, &QThread::started, post_worker, &PostProcessWorker::run);
    connect(post_worker, &PostProcessWorker::progress, this, &MainWindow::slot_post_progress);
    connect(post_worker, &PostProcessWorker::finished, this, &MainWindow::slot_post_finished);
    connect(post_worker, &PostProcessWorker::failed, this, &MainWindow::slot_post_failed);
    connect(post_worker, &PostProcessWorker::finished, post_thread, &QThread::quit);
    connect(post_worker, &PostProcessWorker::failed, post_thread, &QThread::quit);
    connect(post_thread, &QThread::finished, post_worker, &QObject::deleteLater);
    connect(post_thread, &QThread::finished, post_thread, &QObject::deleteLater);
    connect(post_thread, &QThread::finished, this, [this]() {
        post_thread = nullptr;
        post_worker = nullptr;
    });

    post_thread->start();
}

void MainWindow::slot_post_progress(int percent, QString message)
{
    ui->progressBar_reco->setValue(percent);
    ui->label_post_status->setText(message);
}

void MainWindow::slot_post_finished()
{
    ui->label_post_status->setText(tr("Post-processing complete"));
    ui->pushButton_runPostProcess->setEnabled(true);
    loadHistButton_->setEnabled(true);
}

void MainWindow::slot_post_failed(QString error)
{
    ui->label_post_status->setText(tr("Failed: ") + error);
    statusBar()->showMessage(tr("Post-processing failed: ") + error, 8000);
    ui->pushButton_runPostProcess->setEnabled(true);
    loadHistButton_->setEnabled(true);
}
void MainWindow::setupDisplayHistogram()
{
    const QRect view = ui->graphicsView->geometry();
    const int panelX = view.right() + 15;
    const int panelW = 560;
    if (width() < panelX + panelW + 10)
        resize(panelX + panelW + 10, height());

    displayHistLabel_ = new QLabel(tr("Display range (projection images)"), ui->centralwidget);
    displayHistLabel_->setGeometry(panelX, view.top(), panelW, 20);
    QFont f = displayHistLabel_->font();
    f.setBold(true);
    displayHistLabel_->setFont(f);

    displayHist_ = new HistogramRangeControl(ui->centralwidget);
    displayHist_->setGeometry(panelX, view.top() + 24, panelW, 190);
    displayHist_->setIntegerData(true);
    displayHist_->setAutoPercentiles(0.5, 99.5);
    displayHist_->setEnabled(false);
    displayHistLabel_->show();
    displayHist_->show();

    connect(displayHist_, &HistogramRangeControl::rangeChanged, this,
            [this](double, double) { renderDisplayWindow(); });
}

void MainWindow::showScaledImage(bool resetWindow)
{
    ui->graphicsView->resetRoi();
    QGraphicsScene* old = scene;
    scene = new QGraphicsScene(this);
    displayItem_ = nullptr;
    ui->graphicsView->setScene(scene);
    if (old)
        old->deleteLater();

    if (scaledImage.isNull())
        return;
    if (scaledImage.format() != QImage::Format_Grayscale16)
        scaledImage = scaledImage.convertToFormat(QImage::Format_Grayscale16);

    cv::Mat src(scaledImage.height(), scaledImage.width(), CV_16UC1,
                const_cast<uchar*>(scaledImage.constBits()), static_cast<size_t>(scaledImage.bytesPerLine()));
    double mn = 0.0, mx = 0.0;
    cv::minMaxLoc(src, &mn, &mx);
    if (mx <= mn)
        mx = mn + 1.0;

    // Whole-gray-level bins: with more bins than levels the integer data would fall into every
    // other bin and the histogram would look like a comb.
    constexpr int kMaxBins = 2048;
    const int levels = static_cast<int>(mx - mn) + 1;
    const int binWidth = std::max(1, (levels + kMaxBins - 1) / kMaxBins);
    const int bins = (levels + binWidth - 1) / binWidth;
    QVector<double> counts(bins, 0.0);
    for (int r = 0; r < src.rows; ++r) {
        const uint16_t* row = src.ptr<uint16_t>(r);
        for (int c = 0; c < src.cols; ++c)
            counts[static_cast<int>((row[c] - static_cast<int>(mn)) / binWidth)] += 1.0;
    }

    displayHist_->setEnabled(true);
    displayHistLabel_->setText(tr("Display range (projection images)"));
    displayHist_->setHistogram(counts, mn, mn + static_cast<double>(bins) * binWidth, /*keepRange=*/!resetWindow);
    renderDisplayWindow();
}

void MainWindow::renderDisplayWindow()
{
    if (scaledImage.isNull() || !scene || scaledImage.format() != QImage::Format_Grayscale16)
        return;

    double lo = 0.0, hi = 65535.0; // no histogram yet: the plain, un-windowed 16-bit mapping
    if (displayHist_->hasHistogram()) {
        lo = displayHist_->rangeLow();
        hi = displayHist_->rangeHigh();
    }
    if (hi <= lo)
        hi = lo + 1.0;

    cv::Mat src(scaledImage.height(), scaledImage.width(), CV_16UC1,
                const_cast<uchar*>(scaledImage.constBits()), static_cast<size_t>(scaledImage.bytesPerLine()));
    cv::Mat out8;
    src.convertTo(out8, CV_8UC1, 255.0 / (hi - lo), -lo * 255.0 / (hi - lo));
    QImage windowed(out8.data, out8.cols, out8.rows, static_cast<int>(out8.step), QImage::Format_Grayscale8);
    const QPixmap pixmap = QPixmap::fromImage(windowed);

    // Update the pixmap in place so an ROI rectangle drawn on the view survives a window change.
    if (displayItem_)
        displayItem_->setPixmap(pixmap);
    else
        displayItem_ = scene->addPixmap(pixmap);
}

void MainWindow::disableDisplayHistogram()
{
    displayItem_ = nullptr; // the caller is about to replace/clear the scene that owns it
    displayHist_->clear();
    displayHist_->setEnabled(false);
    displayHistLabel_->setText(tr("Display range (n/a for reconstruction previews)"));
}

void MainWindow::setupRecoHistogram()
{
    recoHist_ = new HistogramRangeControl(ui->tab_3);
    recoHist_->setGeometry(20, 188, 740, 145);
    recoHist_->setAutoPercentiles(0.01, 99.99);
    recoHist_->show();

    loadHistButton_ = new QPushButton(tr("Show histogram"), ui->tab_3);
    loadHistButton_->setGeometry(485, 112, 140, 28);
    loadHistButton_->setToolTip(
        tr("Sample reco/ and show its histogram. Drag the handles (or edit Min/Max) to choose the "
           "range mapped onto 0-65535; values outside are clipped. Once shown, the conversion uses "
           "exactly this range instead of Clip %."));
    loadHistButton_->show();

    // The status text is now also used for histogram messages, which don't fit the 400px label.
    ui->label_post_status->setGeometry(215, 155, 545, 20);

    connect(loadHistButton_, &QPushButton::clicked, this, &MainWindow::slot_load_reco_histogram);
    connect(recoHist_, &HistogramRangeControl::rangeChanged, this,
            [this](double lo, double hi) { syncClipBoxesFromRecoRange(lo, hi); });
    connect(ui->doubleSpinBox_clipLow, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { syncRecoRangeFromClipBoxes(); });
    connect(ui->doubleSpinBox_clipHigh, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { syncRecoRangeFromClipBoxes(); });

    // The histogram is of beam-hardening-corrected values when that's enabled, so changing the
    // correction makes the handles' values meaningless.
    const QString bhChanged = tr("Beam-hardening settings changed");
    connect(ui->checkBox_bhEnable, &QCheckBox::toggled, this,
            [this, bhChanged](bool) { invalidateRecoHistogram(bhChanged); });
    for (QDoubleSpinBox* box : {ui->doubleSpinBox_bhC1, ui->doubleSpinBox_bhC2, ui->doubleSpinBox_bhC3})
        connect(box, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                [this, bhChanged](double) { invalidateRecoHistogram(bhChanged); });
}

void MainWindow::invalidateRecoHistogram(const QString& reason)
{
    if (!recoHist_ || !recoHist_->hasHistogram())
        return;
    recoHist_->clear();
    ui->label_post_status->setText(tr("%1: histogram cleared, using Clip % again").arg(reason));
}

void MainWindow::syncClipBoxesFromRecoRange(double lo, double hi)
{
    QSignalBlocker blockLow(ui->doubleSpinBox_clipLow);
    QSignalBlocker blockHigh(ui->doubleSpinBox_clipHigh);
    ui->doubleSpinBox_clipLow->setValue(100.0 * recoHist_->fractionAtValue(lo));
    ui->doubleSpinBox_clipHigh->setValue(100.0 * recoHist_->fractionAtValue(hi));
}

void MainWindow::syncRecoRangeFromClipBoxes()
{
    if (!recoHist_->hasHistogram())
        return;
    recoHist_->setRange(recoHist_->valueAtFraction(ui->doubleSpinBox_clipLow->value() / 100.0),
                        recoHist_->valueAtFraction(ui->doubleSpinBox_clipHigh->value() / 100.0));
}

void MainWindow::slot_load_reco_histogram()
{
    if (workingpath.isEmpty()) {
        statusBar()->showMessage(tr("Load a dataset before showing the histogram"), 3000);
        return;
    }
    if (reco_thread || post_thread || corr_scan_thread) {
        statusBar()->showMessage(tr("A reconstruction, preview, or post-processing run is already in progress"), 3000);
        return;
    }

    PostProcessWorker::Params params = buildPostProcessParams();

    ui->pushButton_runPostProcess->setEnabled(false);
    loadHistButton_->setEnabled(false);
    ui->progressBar_reco->setValue(0);
    ui->label_post_status->setText(tr("Sampling reco/ for the histogram..."));

    post_thread = new QThread(this);
    post_worker = new PostProcessWorker(params);
    post_worker->moveToThread(post_thread);

    connect(post_thread, &QThread::started, post_worker, &PostProcessWorker::runHistogram);
    connect(post_worker, &PostProcessWorker::progress, this, &MainWindow::slot_post_progress);
    connect(post_worker, &PostProcessWorker::histogramReady, this, &MainWindow::slot_reco_histogram_ready);
    connect(post_worker, &PostProcessWorker::failed, this, &MainWindow::slot_post_failed);
    connect(post_worker, &PostProcessWorker::histogramReady, post_thread, &QThread::quit);
    connect(post_worker, &PostProcessWorker::failed, post_thread, &QThread::quit);
    connect(post_thread, &QThread::finished, post_worker, &QObject::deleteLater);
    connect(post_thread, &QThread::finished, post_thread, &QObject::deleteLater);
    connect(post_thread, &QThread::finished, this, [this]() {
        post_thread = nullptr;
        post_worker = nullptr;
    });

    post_thread->start();
}

void MainWindow::slot_reco_histogram_ready(RecoHistogram histogram)
{
    ui->pushButton_runPostProcess->setEnabled(true);
    loadHistButton_->setEnabled(true);
    ui->progressBar_reco->setValue(100);

    if (histogram.counts.isEmpty()) {
        ui->label_post_status->setText(tr("reco/ has no value range (all slices are constant)"));
        return;
    }
    recoHist_->setHistogram(histogram.counts, histogram.lo, histogram.hi);
    syncRecoRangeFromClipBoxes();
    ui->label_post_status->setText(
        tr("Histogram of %1/%2 slices. 16-bit conversion clips exactly to the selected range.")
            .arg(histogram.slicesSampled).arg(histogram.slicesTotal));
}
