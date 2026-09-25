#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QGraphicsScene>
#include <QMouseEvent>
#include <QThread>
#include <vector>
#include "proj_correction.h"
#include "customview.h"
#include "reconstruction_worker.h"
#include "post_process_worker.h"
#include "corr_scan_worker.h"
#include "inmemory_pipeline_worker.h"
#include "histogram_widget.h"
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class QGraphicsPixmapItem;
class QLabel;
class QPushButton;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void openImage();
    CustomView *customview;
    QGraphicsScene *scene = nullptr;

    QString workingpath;
    int angles, corr_projection, repeats, pixel_size, org_cols, org_rows,image_x, image_y;
    float last_angle;

public slots:
    void slotFileOpen();
    void slotGetROI();
    void slotGetCorr();
    void slot_First_Set();
    void slot_corr_scan();
    void slot_browse_angle_file();

    void slot_add_cor_roi();
    void slot_clear_cor_roi();
    void slot_find_cor();
    void slot_run_reconstruction();
    void slot_run_inmemory();
    void slot_run_preview();
    void slot_show_preview_slice(int index);
    void slot_run_post_process();

private slots:
    void slot_reco_progress(int percent, QString message);
    void slot_reco_finished();
    void slot_reco_failed(QString error);
    void slot_preview_ready(cv::Mat bottom, cv::Mat mid, cv::Mat top, ReconstructionWorker::PreviewCache cache);
    void slot_post_progress(int percent, QString message);
    void slot_post_finished();
    void slot_post_failed(QString error);
    void slot_load_reco_histogram();
    void slot_reco_histogram_ready(RecoHistogram histogram);
    void slot_corr_scan_progress(int percent, QString message);
    void slot_corr_scan_finished();
    void slot_corr_scan_failed(QString error);

private:
    Ui::MainWindow *ui;
    Proj_correction *first_set;
    void update_view();
    void correct_first_image();
    QStringList steuerelemente;
    QString first_image;
    QImage scaledImage; // Grayscale16, always the raw (un-windowed) pixel data - see showScaledImage()
    QRect rect_roi_final;

    std::vector<cv::Rect> cor_rois;
    double cor_offset = 0.0;
    double tilt_deg = 0.0;

    QThread *reco_thread = nullptr;
    ReconstructionWorker *reco_worker = nullptr;
    void set_reconstruction_controls_enabled(bool enabled);
    int spot_kernel_size_from_ui() const;
    ReconstructionWorker::Params buildReconstructionParams();

    // "Run In-Memory" reuses reco_thread's slot/signal wiring (slot_reco_progress/finished/failed)
    // since InMemoryPipelineWorker's signals have the same shapes - but runs on its own QThread, so
    // it's tracked separately and still counts as "busy" everywhere reco_thread is checked.
    QThread *inmemory_thread = nullptr;
    InMemoryPipelineWorker *inmemory_worker = nullptr;

    // Updates workingpath/settings.ini in place with the current reconstruction settings (ROI,
    // binning, CoR/tilt, ring filters, FBP filter, circular mask) - called once corr/ or sino/
    // finishes writing, so a later slotFileOpen() on the same settings.ini restores them. The
    // original scan-metadata lines (angles=/last_angle=/etc., written by the acquisition software,
    // never by this app) are preserved verbatim above a sentinel comment line; everything from that
    // sentinel onward is this app's own section and is fully regenerated on each save.
    void saveSettingsIni() const;

    // Display window for projection images (the view on the right of the tab widget): min/max
    // sliders over a histogram of scaledImage. Purely a display mapping - never touches the data
    // used for ROI selection or reconstruction - so dark images can be stretched to pick an ROI.
    HistogramRangeControl* displayHist_ = nullptr;
    QLabel* displayHistLabel_ = nullptr;
    QGraphicsPixmapItem* displayItem_ = nullptr;
    void setupDisplayHistogram();
    // Puts scaledImage into the view through the current display window. resetWindow = true (a
    // newly loaded image) recomputes the histogram and picks an auto-contrast window; false (a
    // zoom into the same image) recomputes the histogram but keeps the window where it still fits.
    void showScaledImage(bool resetWindow);
    void renderDisplayWindow();
    void disableDisplayHistogram(); // for views the window doesn't apply to (reconstruction previews)

    // Histogram of reco/ next to the 16-bit conversion controls; its handles and the Clip % boxes
    // stay in sync, and the run uses the handles' exact values once a histogram has been loaded.
    HistogramRangeControl* recoHist_ = nullptr;
    QPushButton* loadHistButton_ = nullptr;
    void setupRecoHistogram();
    void invalidateRecoHistogram(const QString& reason);
    void syncClipBoxesFromRecoRange(double lo, double hi);
    void syncRecoRangeFromClipBoxes();

    cv::Mat preview_slices[3]; // bottom, mid, top - cached so switching the combo doesn't recompute
    QImage matToPreviewImage(const cv::Mat& slice, double lo, double hi) const;
    void display_preview_slice(int index);
    void refresh_preview_overlay(); // redraws whichever preview slice is showing, if any

    // The showing preview slice after the polar ring filter and beam hardening - cached so that
    // dragging the 16-bit range handles only redoes the (cheap) windowing, not those two steps.
    // previewShownIndex_ is -1 whenever the view shows something other than a preview slice.
    cv::Mat previewProcessed_;
    int previewShownIndex_ = -1;
    // Gray-level window for the preview: the Post Processing histogram's range if one is loaded
    // (exactly what the 16-bit conversion will clip to), else the Clip % percentiles of this slice
    // while 16-bit conversion is enabled, else a plain 1st/99th percentile auto-contrast.
    void previewWindow(const cv::Mat& slice, double& lo, double& hi) const;
    void renderPreviewWindow();
    void refresh_preview_window(); // re-windows the showing preview slice from the cache, if any

    // Sinograms built by the last preview run, reused across "Preview B/M/T" clicks whenever only
    // CoR offset and/or ring-filter/FBP-filter settings changed - see ReconstructionWorker::PreviewCache.
    ReconstructionWorker::PreviewCache preview_sino_cache;

    QThread *post_thread = nullptr;
    PostProcessWorker *post_worker = nullptr;
    PostProcessWorker::Params buildPostProcessParams() const;

    QThread *corr_scan_thread = nullptr;
    CorrScanWorker *corr_scan_worker = nullptr;
};
#endif // MAINWINDOW_H
