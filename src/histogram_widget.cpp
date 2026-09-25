#include "histogram_widget.h"

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPushButton>
#include <QSignalBlocker>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>

namespace {
constexpr double kGrabPixels = 8.0;
constexpr double kMarginLeft = 6.0;
constexpr double kMarginRight = 6.0;
constexpr double kMarginTop = 6.0;
constexpr double kMarginBottom = 18.0;
} // namespace

HistogramView::HistogramView(QWidget* parent) : QWidget(parent)
{
    setMouseTracking(true);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void HistogramView::setHistogram(const QVector<double>& counts, double lo, double hi, bool keepRange)
{
    const double prevLow = low_;
    const double prevHigh = high_;
    const bool hadRange = hasHistogram();

    counts_ = counts;
    dataLow_ = lo;
    dataHigh_ = hi;
    viewLow_ = lo;
    viewHigh_ = hi;
    if (!hasHistogram()) {
        clear();
        return;
    }

    if (keepRange && hadRange) {
        low_ = std::clamp(prevLow, dataLow_, dataHigh_);
        high_ = std::clamp(prevHigh, dataLow_, dataHigh_);
        if (high_ <= low_) {
            low_ = dataLow_;
            high_ = dataHigh_;
        }
    } else {
        low_ = dataLow_;
        high_ = dataHigh_;
    }
    update();
}

void HistogramView::clear()
{
    counts_.clear();
    dataLow_ = dataHigh_ = viewLow_ = viewHigh_ = low_ = high_ = 0.0;
    dragging_ = Handle::None;
    update();
}

void HistogramView::setRange(double lo, double hi)
{
    if (!hasHistogram())
        return;
    lo = std::clamp(lo, dataLow_, dataHigh_);
    hi = std::clamp(hi, dataLow_, dataHigh_);
    if (hi <= lo)
        return;
    low_ = lo;
    high_ = hi;
    update();
}

void HistogramView::setViewSpan(double lo, double hi)
{
    if (!hasHistogram())
        return;
    lo = std::max(lo, dataLow_);
    hi = std::min(hi, dataHigh_);
    if (hi <= lo)
        return;
    viewLow_ = lo;
    viewHigh_ = hi;
    update();
}

void HistogramView::resetViewSpan()
{
    viewLow_ = dataLow_;
    viewHigh_ = dataHigh_;
    update();
}

void HistogramView::setLogScale(bool log)
{
    logScale_ = log;
    update();
}

double HistogramView::valueAtFraction(double fraction) const
{
    if (!hasHistogram())
        return 0.0;
    double total = 0.0;
    for (double c : counts_)
        total += c;
    if (fraction <= 0.0 || total <= 0.0)
        return dataLow_;
    if (fraction >= 1.0)
        return dataHigh_;

    const double target = fraction * total;
    const double binWidth = (dataHigh_ - dataLow_) / counts_.size();
    double cum = 0.0;
    for (int b = 0; b < counts_.size(); ++b) {
        const double next = cum + counts_[b];
        if (next >= target) {
            const double within = counts_[b] > 0.0 ? (target - cum) / counts_[b] : 0.0;
            return dataLow_ + (b + within) * binWidth;
        }
        cum = next;
    }
    return dataHigh_;
}

double HistogramView::fractionAtValue(double value) const
{
    if (!hasHistogram())
        return 0.0;
    double total = 0.0;
    for (double c : counts_)
        total += c;
    if (total <= 0.0 || value <= dataLow_)
        return 0.0;
    if (value >= dataHigh_)
        return 1.0;

    const double pos = (value - dataLow_) / (dataHigh_ - dataLow_) * counts_.size();
    const int bin = std::min(static_cast<int>(pos), static_cast<int>(counts_.size()) - 1);
    double cum = 0.0;
    for (int b = 0; b < bin; ++b)
        cum += counts_[b];
    cum += counts_[bin] * (pos - bin);
    return cum / total;
}

QRectF HistogramView::plotRect() const
{
    return QRectF(kMarginLeft, kMarginTop, width() - kMarginLeft - kMarginRight,
                  height() - kMarginTop - kMarginBottom);
}

double HistogramView::valueToX(double v) const
{
    const QRectF r = plotRect();
    return r.left() + (v - viewLow_) / (viewHigh_ - viewLow_) * r.width();
}

double HistogramView::xToValue(double x) const
{
    const QRectF r = plotRect();
    const double t = std::clamp((x - r.left()) / r.width(), 0.0, 1.0);
    return viewLow_ + t * (viewHigh_ - viewLow_);
}

void HistogramView::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, false);
    const QRectF r = plotRect();

    p.fillRect(r, palette().color(QPalette::Base));
    p.setPen(palette().color(QPalette::Mid));
    p.drawRect(r);

    if (!hasHistogram()) {
        p.setPen(palette().color(QPalette::PlaceholderText));
        p.drawText(r, Qt::AlignCenter, tr("No histogram"));
        return;
    }

    // One bar per pixel column, each the tallest bin that column covers - so a narrow spike (e.g.
    // a saturated or zero-valued pixel population) stays visible however many bins there are.
    const int nb = counts_.size();
    const int cols = std::max(1, static_cast<int>(r.width()));
    const double binsPerUnit = nb / (dataHigh_ - dataLow_);
    QVector<double> heights(cols, 0.0);
    double maxH = 0.0;
    for (int c = 0; c < cols; ++c) {
        const double v0 = viewLow_ + (viewHigh_ - viewLow_) * c / cols;
        const double v1 = viewLow_ + (viewHigh_ - viewLow_) * (c + 1) / cols;
        const int b0 = std::clamp(static_cast<int>((v0 - dataLow_) * binsPerUnit), 0, nb - 1);
        const int b1 = std::clamp(static_cast<int>((v1 - dataLow_) * binsPerUnit), b0 + 1, nb);
        double m = 0.0;
        for (int b = b0; b < b1; ++b)
            m = std::max(m, counts_[b]);
        const double h = logScale_ ? std::log1p(m) : m;
        heights[c] = h;
        maxH = std::max(maxH, h);
    }
    if (maxH > 0.0) {
        QColor bar = palette().color(QPalette::Text);
        bar.setAlpha(150);
        p.setPen(Qt::NoPen);
        p.setBrush(bar);
        for (int c = 0; c < cols; ++c) {
            const double h = heights[c] / maxH * (r.height() - 1.0);
            if (h > 0.0)
                p.drawRect(QRectF(r.left() + c, r.bottom() - h, 1.0, h));
        }
    }

    // Dim everything outside the selected range.
    QColor dim = palette().color(QPalette::Window);
    dim.setAlpha(170);
    const double xl = std::clamp(valueToX(low_), r.left(), r.right());
    const double xh = std::clamp(valueToX(high_), r.left(), r.right());
    p.fillRect(QRectF(r.left(), r.top(), xl - r.left(), r.height()), dim);
    p.fillRect(QRectF(xh, r.top(), r.right() - xh, r.height()), dim);

    // Handles: a line through the plot with a triangular grip below it.
    const QColor accent = palette().color(QPalette::Highlight);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setPen(QPen(accent, 2));
    p.drawLine(QPointF(xl, r.top()), QPointF(xl, r.bottom()));
    p.drawLine(QPointF(xh, r.top()), QPointF(xh, r.bottom()));
    p.setPen(Qt::NoPen);
    p.setBrush(accent);
    for (double x : {xl, xh}) {
        QPainterPath tri;
        tri.moveTo(x, r.bottom() + 1);
        tri.lineTo(x - 5, r.bottom() + 9);
        tri.lineTo(x + 5, r.bottom() + 9);
        tri.closeSubpath();
        p.drawPath(tri);
    }

    p.setRenderHint(QPainter::Antialiasing, false);
    p.setPen(palette().color(QPalette::Text));
    QFont f = font();
    f.setPointSizeF(f.pointSizeF() * 0.85);
    p.setFont(f);
    const QRectF labelRow(r.left(), r.bottom() + 2, r.width(), kMarginBottom - 2);
    p.drawText(labelRow, Qt::AlignLeft | Qt::AlignBottom, QString::number(viewLow_, 'g', 5));
    p.drawText(labelRow, Qt::AlignRight | Qt::AlignBottom, QString::number(viewHigh_, 'g', 5));
}

void HistogramView::mousePressEvent(QMouseEvent* event)
{
    if (!hasHistogram() || event->button() != Qt::LeftButton)
        return;
    const double x = event->pos().x();
    const double dLow = std::abs(x - valueToX(low_));
    const double dHigh = std::abs(x - valueToX(high_));
    // Pick whichever handle is nearer - including a click away from both, so the range can be set
    // by clicking as well as dragging. When the handles sit on top of each other, the click side
    // decides which one moves, so they can always be separated again.
    if (dLow == dHigh)
        dragging_ = x < valueToX(low_) ? Handle::Low : Handle::High;
    else
        dragging_ = dLow < dHigh ? Handle::Low : Handle::High;
    dragTo(x);
}

void HistogramView::mouseMoveEvent(QMouseEvent* event)
{
    if (dragging_ != Handle::None) {
        dragTo(event->pos().x());
        return;
    }
    if (!hasHistogram()) {
        unsetCursor();
        return;
    }
    const double x = event->pos().x();
    const bool near = std::abs(x - valueToX(low_)) <= kGrabPixels || std::abs(x - valueToX(high_)) <= kGrabPixels;
    setCursor(near ? Qt::SizeHorCursor : Qt::ArrowCursor);
}

void HistogramView::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton)
        dragging_ = Handle::None;
}

void HistogramView::dragTo(double x)
{
    // Keep the handles at least one pixel apart in data units so low < high always holds.
    const double minGap = (viewHigh_ - viewLow_) / std::max(1.0, plotRect().width());
    double v = xToValue(x);
    if (dragging_ == Handle::Low)
        v = std::min(v, high_ - minGap);
    else
        v = std::max(v, low_ + minGap);
    v = std::clamp(v, dataLow_, dataHigh_);

    double newLow = low_;
    double newHigh = high_;
    (dragging_ == Handle::Low ? newLow : newHigh) = v;
    if (newHigh <= newLow || (newLow == low_ && newHigh == high_))
        return;
    low_ = newLow;
    high_ = newHigh;
    update();
    emit rangeChanged(low_, high_);
}

HistogramRangeControl::HistogramRangeControl(QWidget* parent) : QWidget(parent)
{
    view_ = new HistogramView(this);

    minSpin_ = new QDoubleSpinBox(this);
    maxSpin_ = new QDoubleSpinBox(this);
    for (QDoubleSpinBox* s : {minSpin_, maxSpin_}) {
        s->setKeyboardTracking(false);
        s->setButtonSymbols(QAbstractSpinBox::NoButtons);
        s->setMinimumWidth(90);
    }
    autoButton_ = new QPushButton(tr("Auto"), this);
    fullButton_ = new QPushButton(tr("Full range"), this);
    zoomButton_ = new QPushButton(tr("Zoom"), this);
    zoomButton_->setCheckable(true);
    zoomButton_->setToolTip(tr("Zoom the horizontal axis to the current Min/Max selection"));
    logCheck_ = new QCheckBox(tr("Log"), this);
    logCheck_->setChecked(true);

    auto* row = new QHBoxLayout;
    row->setContentsMargins(0, 0, 0, 0);
    row->addWidget(new QLabel(tr("Min"), this));
    row->addWidget(minSpin_);
    row->addWidget(new QLabel(tr("Max"), this));
    row->addWidget(maxSpin_);
    row->addStretch(1);
    row->addWidget(autoButton_);
    row->addWidget(fullButton_);
    row->addWidget(zoomButton_);
    row->addWidget(logCheck_);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(2);
    layout->addWidget(view_, 1);
    layout->addLayout(row);

    connect(view_, &HistogramView::rangeChanged, this, &HistogramRangeControl::onViewRangeChanged);
    connect(minSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { onSpinEdited(); });
    connect(maxSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { onSpinEdited(); });
    connect(autoButton_, &QPushButton::clicked, this, &HistogramRangeControl::onAuto);
    connect(fullButton_, &QPushButton::clicked, this, &HistogramRangeControl::onFull);
    connect(zoomButton_, &QPushButton::toggled, this, &HistogramRangeControl::onZoomToggled);
    connect(logCheck_, &QCheckBox::toggled, view_, &HistogramView::setLogScale);

    configureSpinBoxesForData();
}

void HistogramRangeControl::setHistogram(const QVector<double>& counts, double lo, double hi, bool keepRange)
{
    {
        QSignalBlocker blockZoom(zoomButton_);
        zoomButton_->setChecked(false);
    }
    view_->setHistogram(counts, lo, hi, keepRange);
    configureSpinBoxesForData();
    if (view_->hasHistogram() && !keepRange)
        view_->setRange(view_->valueAtFraction(autoLowPercent_ / 100.0),
                        view_->valueAtFraction(autoHighPercent_ / 100.0));
    syncSpinBoxes();

    // Start zoomed when the selection is a small part of the span - typically because a few
    // outlier pixels stretch the span, which would otherwise leave the useful part of the
    // histogram a few pixels wide.
    if (view_->hasHistogram() &&
        (view_->rangeHigh() - view_->rangeLow()) < 0.25 * (view_->dataHigh() - view_->dataLow()))
        zoomButton_->setChecked(true);
}

void HistogramRangeControl::clear()
{
    {
        QSignalBlocker blockZoom(zoomButton_);
        zoomButton_->setChecked(false);
    }
    view_->clear();
    configureSpinBoxesForData();
}

void HistogramRangeControl::setRange(double lo, double hi)
{
    view_->setRange(lo, hi);
    syncSpinBoxes();
}

void HistogramRangeControl::setAutoPercentiles(double lowPercent, double highPercent)
{
    autoLowPercent_ = lowPercent;
    autoHighPercent_ = highPercent;
}

void HistogramRangeControl::setIntegerData(bool integer)
{
    integerData_ = integer;
    configureSpinBoxesForData();
}

void HistogramRangeControl::configureSpinBoxesForData()
{
    const bool has = view_->hasHistogram();
    minSpin_->setEnabled(has);
    maxSpin_->setEnabled(has);
    autoButton_->setEnabled(has);
    fullButton_->setEnabled(has);
    zoomButton_->setEnabled(has);

    QSignalBlocker b1(minSpin_), b2(maxSpin_);
    if (!has) {
        minSpin_->setRange(0.0, 0.0);
        maxSpin_->setRange(0.0, 0.0);
        minSpin_->setValue(0.0);
        maxSpin_->setValue(0.0);
        return;
    }

    const double span = view_->dataHigh() - view_->dataLow();
    int decimals = 0;
    if (!integerData_) {
        // Enough decimals that ~1000 steps across the span are distinguishable, plus one guard digit.
        double step = span / 1000.0;
        while (step < 1.0 && decimals < 9) {
            step *= 10.0;
            ++decimals;
        }
        decimals = std::min(9, decimals + 1);
    }
    for (QDoubleSpinBox* s : {minSpin_, maxSpin_}) {
        s->setDecimals(decimals);
        s->setRange(view_->dataLow(), view_->dataHigh());
        s->setSingleStep(std::max(span / 500.0, integerData_ ? 1.0 : 0.0));
    }
}

void HistogramRangeControl::syncSpinBoxes()
{
    QSignalBlocker b1(minSpin_), b2(maxSpin_);
    minSpin_->setValue(view_->rangeLow());
    maxSpin_->setValue(view_->rangeHigh());
}

void HistogramRangeControl::onViewRangeChanged(double lo, double hi)
{
    syncSpinBoxes();
    emit rangeChanged(lo, hi);
}

void HistogramRangeControl::onSpinEdited()
{
    if (!view_->hasHistogram())
        return;
    const double lo = minSpin_->value();
    const double hi = maxSpin_->value();
    if (hi <= lo) {
        // Reject an inverted/empty range: snap the boxes back to what the view still holds.
        syncSpinBoxes();
        return;
    }
    view_->setRange(lo, hi);
    emit rangeChanged(view_->rangeLow(), view_->rangeHigh());
}

void HistogramRangeControl::onAuto()
{
    view_->setRange(view_->valueAtFraction(autoLowPercent_ / 100.0),
                    view_->valueAtFraction(autoHighPercent_ / 100.0));
    if (zoomButton_->isChecked())
        onZoomToggled(true); // re-fit the zoom to the new selection
    syncSpinBoxes();
    emit rangeChanged(view_->rangeLow(), view_->rangeHigh());
}

void HistogramRangeControl::onFull()
{
    {
        QSignalBlocker blockZoom(zoomButton_);
        zoomButton_->setChecked(false);
    }
    view_->resetViewSpan();
    view_->setRange(view_->dataLow(), view_->dataHigh());
    syncSpinBoxes();
    emit rangeChanged(view_->rangeLow(), view_->rangeHigh());
}

void HistogramRangeControl::onZoomToggled(bool zoomed)
{
    if (!zoomed) {
        view_->resetViewSpan();
        return;
    }
    // Selection plus half its width of context on each side, so the handles aren't jammed
    // against the plot edges.
    const double pad = 0.5 * (view_->rangeHigh() - view_->rangeLow());
    view_->setViewSpan(view_->rangeLow() - pad, view_->rangeHigh() + pad);
}
