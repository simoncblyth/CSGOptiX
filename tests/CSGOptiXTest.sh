#!/bin/bash -l 

source ../env.sh 

fmt="%20s : %s\n"
printf "$fmt" NAME $NAME
printf "$fmt" PREFIX $PREFIX
printf "$fmt" GLM_PREFIX $GLM_PREFIX
printf "$fmt" CSG_PREFIX $CSG_PREFIX
printf "$fmt" OPTIX_PREFIX $OPTIX_PREFIX
printf "$fmt" CUDA_PREFIX $CUDA_PREFIX

usage(){ cat << EOU

CAUTION : this recreates a test executable created by the CMake machinery 

EOU
}


EXE=$PREFIX/lib/$NAME

gcc ${NAME}.cc \
    DemoGeo.cc \
    -std=c++11 \
    -I. \
    -I.. \
    -I${GLM_PREFIX} \
    -I${CUDA_PREFIX}/include \
    -I${OPTIX_PREFIX}/include \
    -I${CSG_PREFIX}/include \
     \
    -L${OPTIX_PREFIX}/lib64 -loptix \
    -L${CSG_PREFIX}/lib -lCSG \
    -L${PREFIX}/lib -lCSGOptiX \
    -lstdc++ \
    -o \
    $EXE

echo EXE $EXE

cd ..
./run.sh 

#which $NAME
#$NAME





