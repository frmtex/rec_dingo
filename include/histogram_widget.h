#ifndef HISTOGRAM_WIDGET_H
#define HISTOGRAM_WIDGET_H

#include <QWidget>
#include <QVector>

class QDoubleSpinBox;
class QCheckBox;
class QPushButton;

// Paints a histogram over [dataLow, dataHigh] with two draggable vertical handles marking a
// selected [low, high] range. Values are in the data's own units (16-bit gray levels, or
// reconstructed attenuation values) - the widget doesn't care which.
class HistogramView : public QWidget
{
    Q_OBJECT

public:
    explicit HistogramView(QWidget* parent = nullptr);

    // counts are uniform bins spanning [lo, hi]. Resets the selected range to the full span
    // unless keepRange is set and the previous range still overlaps the new data span (in which
    // case it's clamped into it).
    void setHistogram(const QVector<double>& counts, double lo, double hi, bool keepRange = false);
    void clear();
    bool hasHistogram() const { return counts_.size() > 1 && dataHigh_ > dataLow_; }

    // Programmatic: clamps into the data span, keeps low < high, and never emits rangeChanged.
    void setRange(double lo, double hi);
    double rangeLow() const { return low_; }
    double rangeHigh() const { return high_; }
    double dataLow() const { return dataLow_; }
    double dataHigh() const { return dataHigh_; }

    void setLogScale(bool log);

    // Zoom the horizontal axis to [lo, hi] (clamped into the data span); the histogram bins and
    // the selected range are unchanged. A single hot/saturated pixel stretches the data span so
    // far that the useful part shrinks to a few pixels wide - this brings it back into reach.
    void setViewSpan(double lo, double hi);
    void resetViewSpan();

    // Cumulative-distribution lookups, interpolated within a bin: valueAtFraction(0.5) is the
    // histogram's median, fractionAtValue is its inverse. Fractions are in [0, 1].
    double valueAtFraction(double fraction) const;
    double fractionAtValue(double value) const;

    QSize sizeHint() const override { return QSize(400, 130); }
    QSize minimumSizeHint() const override { return QSize(150, 80); }

signals:
    // Emitted only for user interaction (dragging a handle or clicking to move the nearest one).
    void rangeChanged(double low, double high);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    QRectF plotRect() const;
    double valueToX(double v) const;
    double xToValue(double x) const;
    void dragTo(double x);

    QVector<double> counts_;
    double dataLow_ = 0.0;
    double dataHigh_ = 0.0;
    double viewLow_ = 0.0;
    double viewHigh_ = 0.0;
    double low_ = 0.0;
    double high_ = 0.0;
    bool logScale_ = true;
    enum class Handle { None, Low, High };
    Handle dragging_ = Handle::None;
};

// HistogramView plus numeric Min/Max entry, Auto/Full buttons and a log-scale toggle.
class HistogramRangeControl : public QWidget
{
    Q_OBJECT

public:
    explicit HistogramRangeControl(QWidget* parent = nullptr);

    // Resets the range to the Auto percentiles, unless keepRange is set (see HistogramView).
    void setHistogram(const QVector<double>& counts, double lo, double hi, bool keepRange = false);
    void clear();
    bool hasHistogram() const { return view_->hasHistogram(); }

    // Programmatic (no rangeChanged), same as HistogramView::setRange.
    void setRange(double lo, double hi);
    double rangeLow() const { return view_->rangeLow(); }
    double rangeHigh() const { return view_->rangeHigh(); }
    // The span the histogram covers (the selectable range lies within it).
    double dataLow() const { return view_->dataLow(); }
    double dataHigh() const { return view_->dataHigh(); }

    // What the Auto button selects, as percentiles of the histogram (default 0.5 / 99.5).
    void setAutoPercentiles(double lowPercent, double highPercent);
    // Integer data (16-bit gray levels) shows no decimals in the Min/Max boxes.
    void setIntegerData(bool integer);

    double valueAtFraction(double fraction) const { return view_->valueAtFraction(fraction); }
    double fractionAtValue(double value) const { return view_->fractionAtValue(value); }

signals:
    // User interaction only (drag, Min/Max entry, Auto, Full).
    void rangeChanged(double low, double high);

private slots:
    void onViewRangeChanged(double lo, double hi);
    void onSpinEdited();
    void onAuto();
    void onFull();
    void onZoomToggled(bool zoomed);

private:
    void syncSpinBoxes();
    void configureSpinBoxesForData();

    HistogramView* view_;
    QDoubleSpinBox* minSpin_;
    QDoubleSpinBox* maxSpin_;
    QPushButton* autoButton_;
    QPushButton* fullButton_;
    QPushButton* zoomButton_;
    QCheckBox* logCheck_;
    double autoLowPercent_ = 0.5;
    double autoHighPercent_ = 99.5;
    bool integerData_ = false;
};

#endif // HISTOGRAM_WIDGET_H
