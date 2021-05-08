#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "sutil_vec_math.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Opticks ok(argc, argv); 
    ok.configure(); 

    int meshIdx = SSys::getenvint("MIDX", 130); 
    int ordinal = SSys::getenvint("ORDINAL", 0 ); 

    const char* outdir = SSys::getenvvar("OUTDIR", "/tmp" ); 
    const char* top    = SSys::getenvvar("TOP", "i0" ); 
    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" );


    CSGFoundry* foundry = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    foundry->upload(); 
    LOG(info) << "foundry " << foundry->desc() ; 
    foundry->summary(); 

    std::vector<CSGPrim> prim ; 
    foundry->getMeshPrim(prim, meshIdx );  
    bool in_range = ordinal < prim.size() ; 

    if(!in_range)
    {
        LOG(info)  
            << " meshIdx " << meshIdx
            << " ordinal " << ordinal 
            << " prim.size " << prim.size()
            ;
         return 1 ;   
    }

    const CSGPrim& pr = prim[ordinal] ;  

    const float4 ce = pr.ce() ; 
    LOG(info)
        << " top " << top
        << " ce " << ce 
        << " ce.w " << ce.w 
        << " NB : no instance transforming : so this only makes sense for global prim " 
        ; 

    CSGOptiX cx(&ok, foundry, outdir); 
    cx.setTop(top); 
    cx.setCE(ce);   // establish the coordinate system 

    double dt = cx.render();  

    std::string path = CSGOptiX::Path(outdir, "pixels.jpg" );  
    std::string bottom_line = CSGOptiX::Annotation(dt); 
    std::string top_line = "CSGOptiXRender" ; 

    cx.snap(path.c_str(), bottom_line.c_str(), top_line.c_str() );   

    return 0 ; 
}
