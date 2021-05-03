#!/bin/bash -l 

sdir=$(pwd)

source ./env.sh 

if [ -z "$PREFIX" -o ! -d "$PREFIX" ]; then 
   echo $0 PREFIX envvar not defined or directory $PREFIX does not exist 
   exit 1 
fi
if [ -z "$NAME" ]; then
   echo $0 NAME envvar not defined
   exit 2
fi


bdir=$PREFIX/build 
echo bdir $bdir NAME $NAME PREFIX $PREFIX

rm -rf $bdir && mkdir -p $bdir 
[ ! -d $bdir ] && exit 1
cd $bdir && pwd 

cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DOPTICKS_PREFIX=${OPTICKS_PREFIX} \
     -DCMAKE_MODULE_PATH=${OPTICKS_HOME}/cmake/Modules \
     -DCMAKE_INSTALL_PREFIX=$PREFIX 

#    -DOptiX_INSTALL_DIR=${OPTIX_PREFIX} \
#     -DCMAKE_MODULE_PATH=${OPTIX_PREFIX}/SDK/CMake \


rm -f $PREFIX/ptx/*.ptx
rm -f $PREFIX/bin/$NAME

make
[ $? -ne 0 ] && echo $0 : make FAIL && exit 1
make install   
[ $? -ne 0 ] && echo $0 : install FAIL && exit 2


exit 0

