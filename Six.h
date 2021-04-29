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
    optix::Context context ;
    optix::Material material ;
    optix::Buffer pixels_buffer ; 
    optix::Buffer posi_buffer ; 
    std::vector<optix::Geometry> solids ; 
    std::vector<optix::Group>    assemblies ; 

    const Params* params ; 
    const char*   ptx_path ; 
    unsigned    entry_point_index ; 
    unsigned    optix_device_ordinal ; 

    Six(const char* ptx_path, const Params* params_);  

    void initContext();
    void initPipeline();
    void setGeo(const Geo* geo);

    template<typename T> void createContextBuffer( T* d_ptr, unsigned num_item, const char* name ); 

    optix::GeometryInstance createGeometryInstance(unsigned solid_idx, unsigned identity);
    optix::Geometry         createSolidGeometry(const CSGFoundry* foundry, unsigned solid_idx);
    optix::GeometryGroup    createSimple(const Geo* geo);
    
    void createSolids(const CSGFoundry* foundry);
    void createGrids(const Geo* geo);
    optix::Group convertGrid(const Grid* gr);

    void launch();

    void save(const char* outdir) ; 


};
