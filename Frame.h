#pragma once

#include <vector>
#include <vector_types.h>


struct STTF ; 

struct Frame
{
    int width ; 
    int height ; 
    int depth ; 
    int channels ; 

    STTF*    ttf ;

    std::vector<float4> isect ; 
    std::vector<uchar4> pixels ; 
    uchar4* d_pixels = nullptr ; 
    float4* d_isect = nullptr ; 

    Frame(int width_, int height_ );
 
    void init(); 
    void init_pixels();
    void init_isect();
 
    void download(); 
    void download_pixels();
    void download_isect();

    void annotate( const char* bottom_line, const char* top_line, int line_height=24  );

    void write(const char* outdir, int jpg_quality) const ;
    void writePNG(const char* dir, const char* name) const ;
    void writeJPG(const char* dir, const char* name, int quality) const ;

    void writePNG(const char* path) const ;
    void writeJPG(const char* path, int quality) const ;

    void writeNP( const char* dir, const char* name) const ;

    float* getIntersectData() const ;
    uchar4* getDevicePixels() const ; 
    float4* getDeviceIsect() const ; 

}; 


