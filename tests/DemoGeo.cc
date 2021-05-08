#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector_types.h>

#include "NP.hh"
#include "sutil_vec_math.h"

#include "CSGPrim.h"
#include "CSGFoundry.h"

#include "CXUtil.h"
#include "DemoGeo.h"
#include "DemoGrid.h"

#include "PLOG.hh"


DemoGeo::DemoGeo(CSGFoundry* foundry_)
    :
    foundry(foundry_),
    top("i0")
{
    init();
}

void DemoGeo::init()
{
    LOG(info) << "[" ; 

    std::string geometry = CXUtil::GetEValue<std::string>("GEOMETRY", "parade" ); 
    unsigned layers = CXUtil::GetEValue<unsigned>("LAYERS", 1) ; 

    LOG(info) 
        << " geometry " << geometry
        << " layers " << layers 
        ;    

    if(strcmp(geometry.c_str(), "sphere_containing_grid_of_spheres") == 0)
    {
        init_sphere_containing_grid_of_spheres(layers );
    }
    else if(strcmp(geometry.c_str(), "parade") == 0)
    {
        init_parade();
    }
    else if(CXUtil::StartsWith(geometry.c_str(), "clustered_"))
    {
        init_clustered( geometry.c_str() + strlen("clustered_")); 
    }
    else if(strcmp(geometry.c_str(), "layered_sphere") == 0 )
    {
        init_layered("sphere", layers);
    }
    else if(strcmp(geometry.c_str(), "layered_zsphere") == 0 )
    {
        init_layered("zsphere", layers);
    }
    else
    {
        init(geometry.c_str()); 
    }

    const float4 ce  = getCenterExtent(); 
    LOG(info) 
        << " top_extent " << ce.w  
        ; 


    LOG(info) << "]" ; 
}

/**
DemoGeo::init_sphere_containing_grid_of_spheres
---------------------------------------------

A cube of side 1 (halfside 0.5) has diagonal sqrt(3):1.7320508075688772 
that will fit inside a sphere of diameter sqrt(3) (radius sqrt(3)/2 : 0.86602540378443)
Container sphere "extent" needs to be sqrt(3) larger than the grid extent.

**/

void DemoGeo::init_sphere_containing_grid_of_spheres(unsigned layers )
{
    LOG(info) << "layers " << layers  ; 
    foundry->makeDemoSolids();  
    unsigned num_solid = foundry->getNumSolid() ; 

    unsigned ias_idx = 0 ; 
    const float4 ce = DemoGrid::AddInstances(foundry, ias_idx, num_solid) ; 

    float big_radius = float(ce.w)*sqrtf(3.f) ;
    LOG(info) << " big_radius " << big_radius ; 

    foundry->makeLayered("sphere", 0.7f, layers ); 
    foundry->makeLayered("sphere", 1.0f, layers ); 
    CSGSolid* so = foundry->makeLayered("sphere", big_radius, 1 ); 

    top = strdup("i0") ; 

    setCenterExtent(so->center_extent); 

}

void DemoGeo::init_parade()
{
    LOG(info) << "["  ;
 
    foundry->makeDemoSolids();  
    unsigned num_solid = foundry->getNumSolid() ; 

    unsigned ias_idx = 0 ; 
    const float4 ce = DemoGrid::AddInstances( foundry, ias_idx, num_solid ); 

    top = strdup("i0") ; 
    setCenterExtent(ce); 

    LOG(info) << "]"  ; 
}

/**
DemoGeo::init_clustered
--------------------

Aiming to test a GAS containing multiple spread (non-concentric) 
placements of the same type of single node Prim.  
Will need to assign appropriate node transforms and get those applied 
to the bbox at Prim+Node(?) level.

**/
void DemoGeo::init_clustered(const char* name)
{
    float unit = CXUtil::GetEValue<float>("CLUSTERUNIT", 200.f ); 
    std::string clusterspec = CXUtil::GetEValue<std::string>("CLUSTERSPEC","-1:2:1,-1:2:1,-1:2:1") ; 

    LOG(info) 
        << " name " << name 
        << " clusterspec " << clusterspec 
        << " unit " << unit 
        ; 

    bool inbox = false ; 
    std::array<int,9> cl ; 
    CXUtil::ParseGridSpec(cl, clusterspec.c_str()); // string parsed into array of 9 ints 
    CSGSolid* so = foundry->makeClustered(name, cl[0],cl[1],cl[2],cl[3],cl[4],cl[5],cl[6],cl[7],cl[8], unit, inbox ); 
    std::cout << "DemoGeo::init_layered" << name << " so.center_extent " << so->center_extent << std::endl ; 

    setCenterExtent(so->center_extent); 
    top = strdup("g0") ; 
}

void DemoGeo::init_layered(const char* name, unsigned layers)
{
    CSGSolid* so = foundry->makeLayered(name, 100.f, layers ); 
    LOG(info) << " name " << name << " so.center_extent " << so->center_extent ; 
    setCenterExtent(so->center_extent); 
    top = strdup("g0") ; 
}

void DemoGeo::init(const char* name)
{
    CSGSolid* so = foundry->make(name) ; 
    LOG(info) << " name " << name << " so.center_extent " << so->center_extent ; 
    setCenterExtent(so->center_extent); 
    top = strdup("g0") ; 
}


std::string DemoGeo::desc() const
{
    std::stringstream ss ; 
    ss << "DemoGeo " ;
    std::string s = ss.str(); 
    return s ; 
}

void DemoGeo::setCenterExtent(const float4&  center_extent_){ center_extent = center_extent_ ;  }
float4 DemoGeo::getCenterExtent() const  { return center_extent ;  }



void DemoGeo::write(const char* dir) const 
{
    LOG(info) << dir  ;  

    NP uspec("<u4", 5); 
    unsigned* u = uspec.values<unsigned>() ; 
    *(u+0) = foundry->getNumSolid(); 
    *(u+1) = 0 ; //getNumGrid(); 
    *(u+2) = 0 ; //InstanceId::ins_bits ; 
    *(u+3) = 0 ; //InstanceId::gas_bits ; 
    *(u+4) = CXUtil::Encode4(top) ; 
    uspec.save(dir, "uspec.npy"); 

    NP fspec("<f4", 4); 
    float* f = fspec.values<float>() ; 
    *(f+0) = center_extent.x ; 
    *(f+1) = center_extent.y ; 
    *(f+2) = center_extent.z ; 
    *(f+3) = center_extent.w ; 
    fspec.save(dir, "fspec.npy"); 

    // with foundry write everything at once, not individual solids/grids
    foundry->write(dir, "foundry"); 

}
