#!/bin/bash

BUILD_DIR="build"
INC_DIR=$BUILD_DIR"/include"
LIB_DIR=$BUILD_DIR"/lib"

function echosp {
    echo "---------------------"
    echo -e $1
}

function error {
    echosp "Error: $1 failed (see above)"
    exit -1
} 

# Build GLAD
cd modules/glad
echo "Building GLAD..."
make rebuild || error "GLAD make"
echosp "GLAD build successful\n"

# Build GLFW
cd ../glfw
echo "Building GLFW..."
cmake -DCMAKE_INSTALL_PREFIX=./build . || error "GLFW cmake"
make || error "GLFW make"
make install || error "GLFW make install"
echosp "GLFW build successful\n"

# Create the build folder and copy the libraries
cd ..
echo "Creating the build folder..."
mkdir -p $INC_DIR $LIB_DIR 2> /dev/null
echo "Creating links for the GLM headers..."
ln -s `pwd`/glm/glm $INC_DIR/glm 2> /dev/null
echo "Creating links for the GLAD headers and libraries..."
ln -s `pwd`/glad/build/include/* $INC_DIR 2> /dev/null
ln -s `pwd`/glad/build/lib64/*   $LIB_DIR 2> /dev/null
echo "Creating links for the GLFW headers and libraries..."
ln -s `pwd`/glfw/build/include/* $INC_DIR 2> /dev/null
ln -s `pwd`/glfw/build/lib/*   $LIB_DIR 2> /dev/null

echosp "All the libraries has been successfully built!"

