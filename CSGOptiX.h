#pragma once

#include <optix.h>
#include <string>
#include <glm/fwd.hpp>
#include "plog/Severity.h"

struct CSGFoundry ; 
struct CSGView ; 

struct Params ; 

class Opticks ; 

#if OPTIX_VERSION < 70000
struct Six ; 
#else
struct Ctx ; 
struct PIP ; 
struct SBT ; 
#endif
struct Frame ; 

#include "SRenderer.hh"

struct CSGOptiX : public SRenderer 
{
    static const plog::Severity LEVEL ; 
    static const char* PTXNAME ; 
    static const char* GEO_PTXNAME ; 
    static const char* ENV(const char* key, const char* fallback);

    unsigned width = 1280u ; 
    unsigned height = 720u ; 
    unsigned depth = 1u ; 

    Opticks*          ok ;  
    Composition*      composition ; 
 
    const CSGFoundry* foundry ; 
    const char*       prefix ; 
    const char*       outdir ; 
    const char*       cmaketarget ; 
    const char*       ptxpath ; 
    const char*       geoptxpath ; 
    //unsigned          cameratype ; 
    float             tmin_model ; 
    int               jpg_quality ; 

    //glm::vec4         eye_model ; 
    //CSGView*          view ; 
    Params*           params  ; 
#if OPTIX_VERSION < 70000
    Six* six ;  
#else
    Ctx* ctx ; 
    PIP* pip ; 
    SBT* sbt ; 
#endif
    Frame* frame ; 


    CSGOptiX(Opticks* ok, const CSGFoundry* foundry, const char* outdir); 

    void init(); 
    void initGeometry();
    void initFrame();
 
    void setTop(const char* tspec); 
    void setCE(const float4& ce); 
    void setCE(const glm::vec4& ce); 

    void updateView(); 


    int  render_flightpath(); 

    // fulfil SRenderer protocol
    double render();    
    void snap(const char* path, const char* bottom_line, const char* top_line=nullptr, unsigned line_height=24); 

    static std::string Annotation( double dt ); 
    static std::string Path( const char* outdir, const char* name);

};



