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

    int midx = SSys::getenvint("MIDX", 130);   // 130 is world box for default CFBASE
    int mord = SSys::getenvint("MORD",   0); 
    int iidx = SSys::getenvint("IIDX",   0);

    const char* outdir = SSys::getenvvar("OUTDIR", "/tmp" ); 
    const char* top    = SSys::getenvvar("TOP", "i0" ); 
    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" );

    CSGFoundry* foundry = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    foundry->upload(); 
    LOG(info) << "foundry " << foundry->desc() ; 
    foundry->summary(); 

    float4 ce = make_float4(0.f, 0.f, 0.f, 1000.f ); 
    int rc = foundry->getCenterExtent(ce, midx, mord, iidx) ;
    if(rc != 0) return 1 ; 


    CSGOptiX cx(&ok, foundry, outdir); 
    cx.setTop(top); 
    cx.setCE(ce);   // establish the coordinate system 


    bool flight = ok.hasArg("--flightconfig") ; 
    if(flight)
    {
        cx.render_flightpath() ; 
    }
    else
    {
        double dt = cx.render();  

        std::string path = CSGOptiX::Path(outdir, "pixels.jpg" );  
        std::string bottom_line = CSGOptiX::Annotation(dt); 
        std::string top_line = "CSGOptiXRender" ; 

        cx.snap(path.c_str(), bottom_line.c_str(), top_line.c_str() );   
    }
    return 0 ; 
}
