QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++20

CONFIG += link_pkgconfig
PKGCONFIG += opencv4

LIBS += -lfftw3f

INCLUDEPATH += $$PWD/include

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    src/customview.cpp \
    src/main.cpp \
    src/mainwindow.cpp \
    src/proj_correction.cpp \
    src/spot_filter.cpp \
    src/tilt_correction.cpp \
    src/rotation_axis.cpp \
    src/sinogram_io.cpp \
    src/fbp_reconstructor.cpp \
    src/ring_filter.cpp \
    src/ring_removal_polar.cpp \
    src/reconstruction_worker.cpp \
    src/post_process_worker.cpp \
    src/angle_file_reader.cpp \
    src/corr_scan_worker.cpp

HEADERS += \
    include/customview.h \
    include/mainwindow.h \
    include/proj_correction.h \
    include/spot_filter.h \
    include/tilt_correction.h \
    include/rotation_axis.h \
    include/sinogram_io.h \
    include/fbp_reconstructor.h \
    include/fbp_cuda_backend.h \
    include/ring_filter.h \
    include/ring_removal_polar.h \
    include/ring_removal_polar_cuda.h \
    include/spot_filter_cuda.h \
    include/phase_retrieval_cuda.h \
    include/reconstruction_worker.h \
    include/post_process_worker.h \
    include/beam_hardening.h \
    include/angle_file_reader.h \
    include/corr_scan_worker.h

FORMS += \
    forms/mainwindow.ui

# --- CUDA (reconstruction backprojection/ramp-filter kernels, optional GPU polar ring removal,
# optional GPU spot filter/phase retrieval) ---
# Quadro P2000 = Pascal, compute capability 6.1.
CUDA_SOURCES += src/fbp_reconstructor_cuda.cu src/ring_removal_polar_cuda.cu \
    src/spot_filter_cuda.cu src/phase_retrieval_cuda.cu
CUDA_ARCH = sm_61

# nvidia-cuda-toolkit's Ubuntu packaging installs nvcc onto PATH and
# cufft.h/libcufft into the normal system include/lib directories, so no
# extra -I/-L should be needed. If the build can't find cuda_runtime.h or
# -lcufft, set CUDA_DIR below to wherever `dpkg -L nvidia-cuda-toolkit` shows
# them and uncomment the INCLUDEPATH/QMAKE_LIBDIR lines.
# CUDA_DIR = /usr/lib/nvidia-cuda-toolkit
# INCLUDEPATH += $$CUDA_DIR/include
# QMAKE_LIBDIR += $$CUDA_DIR/lib64

LIBS += -lcudart -lcufft

cuda.name = cuda ${QMAKE_FILE_IN}
cuda.input = CUDA_SOURCES
cuda.output = ${QMAKE_FILE_BASE}_cuda.o
# ring_removal_polar_cuda.h (unlike fbp_cuda_backend.h) includes OpenCV directly (its API takes/
# returns cv::Mat), so nvcc needs OpenCV's include path too - pkg-config gives the same path g++
# gets via PKGCONFIG above. Both .cu files quote-include their own header from include/, so that
# needs to be on nvcc's path too, same as INCLUDEPATH above does for g++.
cuda.commands = nvcc -std=c++17 -O2 -arch=$$CUDA_ARCH -I$$PWD/include $$system(pkg-config --cflags opencv4) -c ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.variable_out = OBJECTS
QMAKE_EXTRA_COMPILERS += cuda

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
