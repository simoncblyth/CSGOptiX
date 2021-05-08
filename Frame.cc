#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>
#include "CUDA_CHECK.h"

#include "PLOG.hh"
#include "NP.hh"
#include "Frame.h"

//#define STTF_IMPLEMENTATION 1 
//#include "STTF.hh"

#define SIMG_IMPLEMENTATION 1 
#include "SIMG.hh"

Frame::Frame(int width_, int height_)
    :
    width(width_),
    height(height_),
    depth(1),
    channels(4),
    img(new SIMG(width, height, channels,  nullptr )) 
    //ttf(PLOG::instance->ttf)
{
    init();
}

/**
Frame::init
-------------

Allocates pixels and isect on device. 

**/

void Frame::init()
{
    init_pixels(); 
    init_isect(); 
}


void Frame::init_pixels()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_pixels ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_pixels ),
                width*height*sizeof(uchar4)
                ) );
}

uchar4* Frame::getDevicePixels() const 
{
    return d_pixels ; 
}


void Frame::init_isect()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_isect ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_isect ),
                width*height*sizeof(float4)
                ) );
}
float4* Frame::getDeviceIsect() const 
{
    return d_isect ; 
}


void Frame::download()
{
    LOG(info) ; 
    download_pixels();  
    download_isect();  
}

void Frame::download_pixels()
{
    LOG(info) << "d_pixels " << d_pixels << std::endl ; 

    pixels.resize(width*height);  
    CUDA_CHECK( cudaMemcpy(
                static_cast<void*>( pixels.data() ),
                d_pixels,
                width*height*sizeof(uchar4),
                cudaMemcpyDeviceToHost
    ));
}

void Frame::download_isect()
{
    LOG(info) << "d_isect " << d_isect << std::endl ; 

    isect.resize(width*height);  
    CUDA_CHECK( cudaMemcpy(
                static_cast<void*>( isect.data() ),
                d_isect,
                width*height*sizeof(float4),
                cudaMemcpyDeviceToHost
    ));
}



unsigned char* Frame::getPixelsData() const
{
    unsigned char* data = (unsigned char*)pixels.data();  
    return data ; 
}


void Frame::annotate( const char* bottom_line, const char* top_line, int line_height )
{
    img->setData( getPixelsData() ); 
    img->annotate( bottom_line, top_line, line_height ); 

/*
    if(!ttf->valid || line_height > int(height)) return ; 
    unsigned char* ptr = (unsigned char*)pixels.data() ; 

    if( top_line )
        ttf->annotate( ptr, int(channels), int(width), int(height), line_height, top_line, false ); 

    if( bottom_line )
        ttf->annotate( ptr, int(channels), int(width), int(height), line_height, bottom_line, true ); 
*/

}



void Frame::write(const char* outdir, int jpg_quality) const 
{
    img->annotate();   // sets the image data

    LOG(info) << outdir << std::endl ; 
    writePNG(outdir, "pixels.png");  
    writeJPG(outdir, "pixels.jpg", jpg_quality);  
    writeNP(  outdir, "posi.npy" );
}


void Frame::writePNG(const char* dir, const char* name) const 
{
    //SIMG img(width, height, channels,  getPixelsData() ); 
    img->writePNG(dir, name); 
}
void Frame::writePNG(const char* path) const 
{
    //SIMG img(width, height, channels,  getPixelsData() ); 
    img->writePNG(path); 
}

void Frame::writeJPG(const char* dir, const char* name, int quality) const 
{
    //SIMG img(width, height, channels,  getPixelsData() ); 
    img->writeJPG(dir, name, quality); 
}
void Frame::writeJPG(const char* path, int quality) const 
{
    //SIMG img(width, height, channels,  getPixelsData() ); 
    img->writeJPG(path, quality); 
}




void Frame::writeNP( const char* dir, const char* name) const 
{
    std::cout << "Frame::writeNP " << dir << "/" << name << std::endl ; 
    NP::Write(dir, name, getIntersectData(), height, width, 4 );
}
float* Frame::getIntersectData() const
{
    return (float*)isect.data();
}

