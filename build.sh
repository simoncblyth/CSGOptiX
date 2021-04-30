#!/bin/bash -l 

sdir=$(pwd)
name=$(basename $sdir)

source ./env.sh 

if [ -z "$PREFIX" -o ! -d "$PREFIX" ]; then 
   echo $0 PREFIX envvar not defined or directory $PREFIX does not exist 
   exit 1 
fi

bdir=$PREFIX/build 
echo bdir $bdir name $name PREFIX $PREFIX

rm -rf $bdir && mkdir -p $bdir 
[ ! -d $bdir ] && exit 1
cd $bdir && pwd 


glm-version(){ echo 0.9.9.8 ; }
glm-name(){    echo glm-$(glm-version) ; }
glm-dir(){     echo $PREFIX/externals/glm/$(glm-name) ; }
glm-url(){     echo https://github.com/g-truc/glm/releases/download/$(glm-version)/$(glm-name).zip ; }
glm-dist(){    echo $(dirname $(glm-dir))/$(basename $(glm-url)) ; }
glm-get(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   local dir=$(dirname $(glm-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(glm-url)
   local zip=$(basename $url)
   local nam=$(glm-name)
   local opt=$( [ -n "${VERBOSE}" ] && echo "" || echo "-q" )

   local hpp=$nam/glm/glm/glm.hpp
   echo $msg nam $nam PWD $PWD hpp $hpp
   ## curiously directories under /tmp being emptied but directory structure
   ## remains, so have to check file rather than directory existance  

   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -f "$hpp" ] && unzip $opt $zip -d $nam
   ln -sfnv $(glm-name)/glm glm 
   echo symbolic link for access without version in path
   cd $iwd
}
glm-get

 
export GLM_PREFIX=$(glm-dir)/glm


cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DOptiX_INSTALL_DIR=${OPTIX_PREFIX} \
     -DCMAKE_MODULE_PATH=${OPTIX_PREFIX}/SDK/CMake \
     -DCMAKE_INSTALL_PREFIX=$PREFIX

rm -f $PREFIX/ptx/*.ptx
rm -f $PREFIX/bin/$name

make
[ $? -ne 0 ] && echo $0 : make FAIL && exit 1
make install   
[ $? -ne 0 ] && echo $0 : install FAIL && exit 2


exit 0

