#include <string>
#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "sutil_vec_math.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "DemoGeo.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Opticks ok(argc, argv); 
    ok.configure(); 

    const char* outdir = SSys::getenvvar("OUTDIR", "/tmp" );  
    std::string path = CSGOptiX::Path(outdir, "pixels.jpg"); 

    CSGFoundry foundry ; 
    DemoGeo dg(&foundry) ;  
    dg.write(outdir);  
    float4 ce = dg.getCenterExtent() ;   // defines the center-extent of the region to view
    ce.w = ce.w*1.4 ; 

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
