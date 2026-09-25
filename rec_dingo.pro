QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++20

macx {
    QMAKE_MACOSX_DEPLOYMENT_TARGET = 26
    # Homebrew's opencv@4 is keg-only (not symlinked into /opt/homebrew), so it isn't on
    # pkg-config's default search path the way the Linux build's link_pkgconfig expects -
    # link against it directly instead.
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
            -lopencv_features2d \
            -lfftw3f
} else:win32-msvc {
    # The official opencv.org Windows package ships MSVC-built libraries only (no pkg-config .pc
    # files - it ships OpenCVConfig.cmake instead, for CMake) - link against it directly, same
    # idea as the macOS branch above. Only MSVC Qt kits work here: MinGW can't link against these
    # MSVC-ABI libraries, and nvcc (below) needs cl.exe as its host compiler on Windows anyway.
    OPENCV_DIR = C:/opencv
    # Visual Studio toolset folder under build/x64 - check what's actually there (vc14/vc15/vc16)
    # if this doesn't match your install.
    OPENCV_VC_DIR = $$OPENCV_DIR/build/x64/vc16

    INCLUDEPATH += $$OPENCV_DIR/build/include
    LIBS += -L$$OPENCV_VC_DIR/lib
    # Exact "world" library name depends on your OpenCV version - check $$OPENCV_VC_DIR/lib for
    # the exact filename (e.g. opencv_world4100.lib for 4.10.0, opencv_world490.lib for 4.9.0)
    # and change the line below to match if it's not 4.10.0.
    CONFIG(debug, debug|release) {
        LIBS += -lopencv_world4100d
    } else {
        LIBS += -lopencv_world4100
    }

    # FFTW has no official Windows package. Easiest path: `vcpkg install fftw3:x64-windows`, then
    # point FFTW_DIR at vcpkg's install prefix (vcpkg\installed\x64-windows). Otherwise, grab the
    # precompiled DLLs from fftw.org/install/windows.html and generate libfftw3f-3.lib from the
    # .def file with MSVC's `lib.exe` per their instructions.
    FFTW_DIR = C:/vcpkg/installed/x64-windows
    INCLUDEPATH += $$FFTW_DIR/include
    LIBS += -L$$FFTW_DIR/lib -llibfftw3f-3
} else {
    CONFIG += link_pkgconfig
    PKGCONFIG += opencv4

    LIBS += -lfftw3f
}

INCLUDEPATH += $$PWD/include

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    src/customview.cpp \
    src/histogram_widget.cpp \
    src/main.cpp \
    src/mainwindow.cpp \
    src/proj_correction.cpp \
    src/spot_filter.cpp \
    src/tilt_correction.cpp \
    src/rotation_axis.cpp \
    src/sinogram_io.cpp \
    src/fbp_reconstructor.cpp \
    src/gridrec_reconstructor.cpp \
    src/ring_filter.cpp \
    src/ring_removal_polar.cpp \
    src/reconstruction_worker.cpp \
    src/post_process_worker.cpp \
    src/angle_file_reader.cpp \
    src/corr_scan_worker.cpp

HEADERS += \
    include/customview.h \
    include/histogram_widget.h \
    include/mainwindow.h \
    include/proj_correction.h \
    include/spot_filter.h \
    include/tilt_correction.h \
    include/rotation_axis.h \
    include/sinogram_io.h \
    include/fbp_reconstructor.h \
    include/gridrec_reconstructor.h \
    include/slice_reconstructor.h \
    include/ring_filter.h \
    include/ring_removal_polar.h \
    include/reconstruction_worker.h \
    include/post_process_worker.h \
    include/beam_hardening.h \
    include/angle_file_reader.h \
    include/corr_scan_worker.h

FORMS += \
    forms/mainwindow.ui

macx {
    # macOS: FBP backprojection/ramp filter run on Accelerate/vDSP (FbpAccelerateBackend). There's
    # no macOS equivalent of the CUDA polar-ring backend, so that filter always takes the CPU path
    # (PolarRingRemoval::remove_ring, already shared cross-platform) - see the __APPLE__ guards in
    # reconstruction_worker.cpp and mainwindow.cpp.
    HEADERS += include/fbp_accelerate_backend.h include/gridrec_accelerate_backend.h
    SOURCES += src/fbp_accelerate_backend.cpp src/gridrec_accelerate_backend.cpp
} else {
    # --- CUDA (reconstruction backprojection/ramp-filter kernels, optional GPU polar ring removal,
    # optional GPU spot filter/phase retrieval) ---
    # Quadro P2000 = Pascal, compute capability 6.1.
    HEADERS += include/fbp_cuda_backend.h include/ring_removal_polar_cuda.h \
        include/spot_filter_cuda.h include/phase_retrieval_cuda.h include/gridrec_cuda_backend.h
    CUDA_SOURCES += src/fbp_reconstructor_cuda.cu src/ring_removal_polar_cuda.cu \
        src/spot_filter_cuda.cu src/phase_retrieval_cuda.cu src/gridrec_reconstructor_cuda.cu
    CUDA_ARCH = sm_61

    win32-msvc {
        # CUDA_PATH is set automatically by the CUDA Toolkit installer (e.g.
        # "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"). nvcc needs cl.exe as its
        # host compiler - Qt Creator's MSVC kit already puts it on PATH before invoking qmake/
        # nmake/jom, so no -ccbin should be needed unless you're building from a plain shell that
        # hasn't run vcvarsall.bat.
        INCLUDEPATH += $$(CUDA_PATH)/include
        LIBS += -L$$(CUDA_PATH)/lib/x64 -lcudart -lcufft

        cuda.name = cuda ${QMAKE_FILE_IN}
        cuda.input = CUDA_SOURCES
        cuda.output = ${QMAKE_FILE_BASE}_cuda.obj
        # ring_removal_polar_cuda.h (unlike fbp_cuda_backend.h) includes OpenCV directly (its API
        # takes/returns cv::Mat), so nvcc needs OpenCV's include path too, same as the g++ build
        # gets from OPENCV_DIR above.
        cuda.commands = nvcc -std=c++17 -O2 -arch=$$CUDA_ARCH -I$$PWD/include -I$$OPENCV_DIR/build/include -c ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
        cuda.variable_out = OBJECTS
        QMAKE_EXTRA_COMPILERS += cuda
    } else {
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
    }
}

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
