#pragma once

#include <vector>
#include <optixu/optixpp_namespace.h>

struct Geo ; 
struct CSGFoundry ; 
struct Grid ; 
struct Params ; 
struct CSGSolid ; 

struct Six
{
    optix::Context     context ;
    optix::Material    material ;
    optix::Buffer      pixels_buffer ; 
    optix::Buffer      posi_buffer ; 

    const Params*     params ; 
    const char*       ptx_path ; 
    unsigned          entry_point_index ; 
    unsigned          optix_device_ordinal ; 
    const Geo*        geo ; 
    const CSGFoundry* foundry ; 

    std::vector<optix::Geometry> solids ; 
    std::vector<optix::Group>    assemblies ; 


    Six(const char* ptx_path, const Params* params_);  

    void initContext();
    void initPipeline();
    void setGeo(const Geo* geo);
    void convert();

    template<typename T> void createContextBuffer( T* d_ptr, unsigned num_item, const char* name ); 

    optix::GeometryInstance createGeometryInstance(unsigned solid_idx, unsigned identity);
    optix::Geometry         createGeometry(unsigned solid_idx);
    
    void convertSolids();
    void convertGroups();  // pre-7 IAS 
    optix::Group createAssembly(unsigned ias_idx);

    void launch();
    void save(const char* outdir) ; 


};
