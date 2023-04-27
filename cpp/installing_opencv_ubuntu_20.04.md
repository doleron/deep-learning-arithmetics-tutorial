# How to install OpenCV from sources on Ubuntu 20.04

For C++, Python and Java. Worked also on Ubuntu 22.04

## Pre-flight steps

$ sudo apt update
$ sudo apt upgrade

## Installing dependencies

$ sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev v4l-utils

Obs: on Ubuntu 22.04 there is no libdc1394-22-dev lib. Just ommit it.

## Getting OpenCV

$ mkdir ~/opencv_build && cd ~/opencv_build

$ git clone https://github.com/opencv/opencv_contrib.git
$ git clone https://github.com/opencv/opencv.git

### Optional - To get specific version

If you want to install a specific version:

$ cd opencv_contrib
$ git checkout 4.5.4
$ cd ..

$ cd opencv
$ git checkout 4.5.4
$ cd ..

### Optional - Java support:

$ sudo apt install openjdk-11-jdk ant
$ ANT_HOME=/usr/share/ant
$ export ANT_HOME

## Preparing Building 

$ mkdir build && cd build

$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF \
    -D WITH_TBB=ON -D BUILD_TBB=ON -D WITH_EIGEN=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=OFF ..

### Optional - Remove support to gstreamer

Add the flag: -D WITH_GSTREAMER=OFF \

## Building    

$ make -j8
$ sudo make install
$ sudo ldconfig -v

## Checking installation

$ pkg-config --modversion opencv4
$ python3 -c "import cv2; print(cv2.__version__)"
