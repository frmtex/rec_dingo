TEMPLATE = app
TARGET = cor_diagnostic
CONFIG += console c++20
CONFIG -= app_bundle
QT = core

# Standalone command-line diagnostic - reuses the same SinogramReader/FbpReconstructor sources
# (and the same platform backend split) as the main rec_dingo.pro, just without the GUI/Qt Widgets
# dependency. See cor_diagnostic.cpp for what it does and why.

macx {
    QMAKE_MACOSX_DEPLOYMENT_TARGET = 26
    INCLUDEPATH += /opt/homebrew/opt/opencv@4/include/opencv4 \
                   /opt/homebrew/opt/opencv@4/include \
                   /opt/homebrew/include
    LIBS += -L/opt/homebrew/opt/opencv@4/lib \
            -L/opt/homebrew/lib \
            -framework Accelerate \
            -lopencv_core \
            -lopencv_highgui \
            -lopencv_imgproc \
            -lopencv_imgcodecs \
            -lopencv_videoio \
            -lopencv_video \
            -lopencv_objdetect \
            -lopencv_features2d
} else {
    CONFIG += link_pkgconfig
    PKGCONFIG += opencv4
}

INCLUDEPATH += $$PWD/../include

SOURCES += \
    cor_diagnostic.cpp \
    $$PWD/../src/sinogram_io.cpp \
    $$PWD/../src/fbp_reconstructor.cpp

HEADERS += \
    $$PWD/../include/sinogram_io.h \
    $$PWD/../include/fbp_reconstructor.h \
    $$PWD/../include/slice_reconstructor.h

macx {
    HEADERS += $$PWD/../include/fbp_accelerate_backend.h
    SOURCES += $$PWD/../src/fbp_accelerate_backend.cpp
} else {
    HEADERS += $$PWD/../include/fbp_cuda_backend.h
    CUDA_SOURCES += $$PWD/../src/fbp_reconstructor_cuda.cu
    CUDA_ARCH = sm_61
    LIBS += -lcudart -lcufft

    cuda.name = cuda ${QMAKE_FILE_IN}
    cuda.input = CUDA_SOURCES
    cuda.output = ${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = nvcc -std=c++17 -O2 -arch=$$CUDA_ARCH -I$$PWD/../include -c ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
    cuda.variable_out = OBJECTS
    QMAKE_EXTRA_COMPILERS += cuda
}
