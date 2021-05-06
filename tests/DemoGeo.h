#pragma once
#include <array>
#include <vector>
#include <string>

struct CSGSolid ; 
struct CSGPrim ; 
struct CSGFoundry ; 

/**
DemoGeo is for high level definition of specific examples of geometry, 
it is fine as a test of the CSG model but it definitely does not belong 
in the model package.
**/

struct DemoGeo
{
    DemoGeo(CSGFoundry* foundry_);

    void init();
    void init_sphere_containing_grid_of_spheres(unsigned layers);
    void init_parade();
    void init_layered(const char* name, unsigned layers);
    void init_clustered(const char* name);
    void init(const char* name);

    std::string desc() const ;
    void        write(const char* prefix) const ; 
    void        setCenterExtent(const float4& center_extent); 
    float4      getCenterExtent() const ; 

    float4       center_extent = {0.f, 0.f, 0.f, 100.f} ; 
    CSGFoundry*  foundry ; 
    const char*  top ;  
};


