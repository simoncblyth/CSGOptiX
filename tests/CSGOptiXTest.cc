#include <string>
#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "sutil_vec_math.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "DemoGeo.h"


// TODO: eliminate this executable by 
//       enabling CSGOptiXRender to load/create demo geometries 


int main(int argc, char** argv)
{
    for(int i=0 ; i < argc ; i++ ) std::cout << i << ":" << argv[i] << std::endl ; 

    int meshIdx = SSys::getenvint("MIDX", -1 ); 
    int ordinal = SSys::getenvint("ORDINAL", 0 ); 

    OPTICKS_LOG(argc, argv); 
    Opticks ok(argc, argv); 
    ok.configure(); 

    const char* outdir = SSys::getenvvar("OUTDIR", "/tmp" );  
    std::string path = CSGOptiX::Path(outdir, "pixels.jpg"); 

    CSGFoundry foundry ;
    DemoGeo dg(&foundry) ;  
    //dg.write(outdir);  

    bool mesh_target = meshIdx > -1 ; 

    float4 ce = make_float4( 0, 0, 0, 1000.) ; 
    if(mesh_target)
    {
        std::vector<CSGPrim> prim ; 
        foundry.getMeshPrim(prim, meshIdx );  
        bool in_range = ordinal < prim.size() ; 

        LOG(info)  
            << " meshIdx " << meshIdx
            << " ordinal " << ordinal 
            << " prim.size " << prim.size()
            ;

        if(!in_range) return 1 ;   

        const CSGPrim& pr = prim[ordinal] ;  

        float4 pce = pr.ce() ; 
        ce.x = pce.x ; 
        ce.y = pce.y ; 
        ce.z = pce.z ; 
        ce.w = pce.w ; 
    }
    else
    {
        float4 dce = dg.getCenterExtent() ;   // defines the center-extent of the region to view
        ce.x = dce.x ; 
        ce.y = dce.y ; 
        ce.z = dce.z ; 
        ce.w = dce.w ; 
    }


    LOG(info)
        << " ce " << ce 
        << " ce.w " << ce.w 
        << " NB : no instance transforming : so this only makes sense for global prim " 
        ; 


    foundry.dump(); 
    foundry.upload();   // uploads nodes, planes, transforms

    CSGOptiX cx(&ok, &foundry, outdir); 
    cx.setTop(dg.top);    // sets geometry handle to be traced
    cx.setCE(ce);         // thru to composition

    double dt = cx.render(); 

    std::string top_line = "CSGOptiXTest" ; 
    std::string bottom_line = CSGOptiX::Annotation(dt); 
    cx.snap(path.c_str(), bottom_line.c_str(), top_line.c_str() );   

    return 0 ; 
}
