#ifndef IMAGE_LABEL_H
#define IMAGE_LABEL_H

#include <QLabel>
#include <QMouseEvent>

class ClickableLabel : public QLabel {
    Q_OBJECT
public:
    explicit ClickableLabel(const QString &text, QWidget *parent = nullptr);
signals:
    void clicked(const QPoint &pos);

protected:
    void mousePressEvent(QMouseEvent *event) override;
};

#endif // IMAGE_LABEL_H
