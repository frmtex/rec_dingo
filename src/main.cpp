#include "mainwindow.h"

#include <QApplication>
#include <QMetaType>

int main(int argc, char *argv[])
{
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<ReconstructionWorker::PreviewCache>("ReconstructionWorker::PreviewCache");
    qRegisterMetaType<RecoHistogram>("RecoHistogram");

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
