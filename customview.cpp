#include "customview.h"
#include <QDebug>
#include <QGraphicsRectItem>
#include <cmath>


CustomView::CustomView(QWidget *parent) : QGraphicsView(parent), roiRect(nullptr), isDrawing(false) {
}
void CustomView::mousePressEvent(QMouseEvent *event) {
    if (roiRect) {
        scene()->removeItem(roiRect);
        delete roiRect;
        roiRect = nullptr;
    }
    if (event->button() == Qt::LeftButton) {
        startPos = mapToScene(event->pos()); // Map to scene coordinates
        isDrawing = true;

        roiRect = new QGraphicsRectItem(QRectF(startPos, QSizeF(0, 0)));
        roiRect->setPen(QPen(Qt::red, 2, Qt::DashLine));
        scene()->addItem(roiRect);
    } else {
        QGraphicsView::mousePressEvent(event);
    }
}

void CustomView::mouseMoveEvent(QMouseEvent *event) {
    if (isDrawing && roiRect) {
        QPointF currentPos = mapToScene(event->pos());
        QRectF rect(startPos, currentPos);
        roiRect->setRect(rect.normalized()); // Ensure positive width/height
    } else {
        QGraphicsView::mouseMoveEvent(event);
    }
}

void CustomView::mouseReleaseEvent(QMouseEvent *event) {
    if (isDrawing && event->button() == Qt::LeftButton) {
        isDrawing = false;

        if (roiRect) {
            rect_final = roiRect->mapRectToScene(roiRect->rect()); // now contains your final ROI in scene coordinates
            qDebug() << "final" << rect_final;
        }
    } else {
        QGraphicsView::mouseReleaseEvent(event);
    }
}

void CustomView::resetRoi()
{
    roiRect = nullptr;
    isDrawing = false;
}

void CustomView::setDisplayedImageSize(QSize displayedSize, QSize originalSize)
{
    displayed_size = displayedSize;
    original_size = originalSize;
}

QRect CustomView::roiInImageCoords() const
{
    if (displayed_size.width() <= 0 || displayed_size.height() <= 0)
        return rect_final.toRect();

    double sx = static_cast<double>(original_size.width()) / displayed_size.width();
    double sy = static_cast<double>(original_size.height()) / displayed_size.height();

    int x = static_cast<int>(std::lround(rect_final.x() * sx));
    int y = static_cast<int>(std::lround(rect_final.y() * sy));
    int w = static_cast<int>(std::lround(rect_final.width() * sx));
    int h = static_cast<int>(std::lround(rect_final.height() * sy));

    return QRect(x, y, w, h);
}