#ifndef CLICKLABEL_H
#define CLICKLABEL_H

#include <QObject>
#include <QWidget>
#include <QLabel>
#include <QMouseEvent>
#include <QGraphicsView>
#include <QMouseEvent>


class CustomView : public QGraphicsView {
    Q_OBJECT
public:
    CustomView(QWidget *parent = nullptr);
    QRectF rect_final;

    // Call whenever a new image is placed in the scene: displayedSize is the
    // actual size (in scene/pixel coordinates) of what's shown, originalSize
    // is the size of the unscaled image it was derived from. Lets ROI
    // selections be converted back to original-image pixel coordinates with
    // the correct per-axis scale, instead of every caller guessing a scale
    // factor from context.
    void setDisplayedImageSize(QSize displayedSize, QSize originalSize);

    // Converts rect_final from displayed/scene pixel coordinates into the
    // original image's pixel coordinates. Falls back to rect_final as-is if
    // setDisplayedImageSize() hasn't been called yet.
    QRect roiInImageCoords() const;

    // Forgets the in-progress/last ROI rect item without deleting it. Call
    // this before replacing or clearing the view's scene (e.g. scene->clear()
    // or scene->deleteLater()), since that already destroys the item and
    // would otherwise leave roiRect dangling.
    void resetRoi();

protected:
    void mousePressEvent(QMouseEvent *event) ;
    void mouseMoveEvent(QMouseEvent *event) ;
    void mouseReleaseEvent(QMouseEvent *event) ;

private:
    QPointF startPos;
    QGraphicsRectItem *roiRect;
    bool isDrawing;

    QSize displayed_size;
    QSize original_size;
};
#endif // CLICKLABEL_H
