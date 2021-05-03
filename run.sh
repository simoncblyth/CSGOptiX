#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
spec=$1

source ./env.sh 
[ $? -ne 0 ] && echo $0 FAIL from ./env.sh && exit 1 

echo RM OUTDIR $OUTDIR
rm -rf $OUTDIR
mkdir -p $OUTDIR
mkdir -p $OUTDIR/foundry

echo $msg BINARY $BINARY spec $spec

if [ -n "$DEBUG" ]; then 
    if [ "$(uname)" == "Linux" ]; then
       gdb -ex r --args $BINARY $spec
    elif [ "$(uname)" == "Darwin" ]; then
       lldb_ $BINARY $spec
    fi
else
    $BINARY $spec
fi 

[ $? -ne 0 ] && echo $0 : run  FAIL && exit 3

jpg=$OUTDIR/pixels.jpg
npy=$OUTDIR/posi.npy

echo BINARY : $BINARY 
echo OUTDIR : $OUTDIR
echo spec : $spec
echo jpg  : $jpg

if [ "$(uname)" == "Linux" ]; then 
   dig=$(cat $jpg | md5sum)
else
   dig=$(cat $jpg | md5)
fi 
echo md5  : $dig
echo npy  : $npy
ls -l $jpg $npy 

if [ "$(uname)" == "Darwin" ]; then
    open $jpg
fi
exit 0

