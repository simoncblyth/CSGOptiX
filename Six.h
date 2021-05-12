#pragma once

#include <map>
#include <vector>
#include <optixu/optixpp_namespace.h>
#include "qat4.h"

class Opticks ; 

struct CSGFoundry ; 
struct CSGSolid ; 

struct Params ; 

struct Six
{
    const Opticks*     ok ; 
    int                one_gas_ias ; 
    unsigned long long emm ; 

    optix::Context     context ;
    optix::Material    material ;
    optix::Buffer      pixels_buffer ; 
    optix::Buffer      posi_buffer ; 

    Params*           params ; 
    const char*       ptx_path ; 
    const char*       geo_ptx_path ; 
    unsigned          entry_point_index ; 
    unsigned          optix_device_ordinal ; 
    const CSGFoundry* foundry ; 

    //std::vector<optix::Geometry> solids ; 
    std::map<unsigned, optix::Geometry> solids ; 

    std::vector<optix::Group>    groups ; 

    Six(const Opticks* ok, const char* ptx_path, const char* geo_ptx_path, Params* params_);  
    void initContext();
    void initFrame();   // hookup pixels and isect buffers

    void updateContext();  // for changed params such as viewpoint 
    void initPipeline();
    void setFoundry(const CSGFoundry* foundry);

    void create();
    void createContextBuffers();
    void createGAS();
    void createIAS();

    template<typename T> void createContextBuffer( T* d_ptr, unsigned num_item, const char* name ); 
    optix::Group              createIAS(unsigned ias_idx);
    optix::Group              createIAS(const std::vector<qat4>& inst );
    optix::Group              createOneGASIAS(unsigned ias_idx, unsigned one_gas_ias) ;

    optix::GeometryInstance   createGeometryInstance(unsigned solid_idx, unsigned identity);
    optix::Geometry           createGeometry(unsigned solid_idx);
    optix::Geometry           getGeometry(unsigned solid_idx) const ;  
 
    void setTop(const char* spec);
    void launch();

    void snap(const char* path, const char* bottom_line, const char* top_line, unsigned line_height) ;

};
