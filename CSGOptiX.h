#pragma once

#include <optix.h>
#include <glm/fwd.hpp>
#include "plog/Severity.h"

struct CSGFoundry ; 
struct View ; 
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
    const CSGFoundry* foundry ; 
    const char*       prefix ; 
    const char*       outdir ; 
    const char*       cmaketarget ; 
    const char*       ptxpath ; 
    const char*       geoptxpath ; 
    unsigned          cameratype ; 
    int               jpg_quality ; 

    glm::vec4         eye_model ; 
    View*             view ; 
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
    void setCE(const glm::vec4& ce, float tmin_model, float tmax_model );
    void setTop(const char* tspec); 

    int  render_flightpath(); 

    // fulfil SRenderer protocol
    double render();    
    void snap(const char* path, const char* bottom_line, const char* top_line=nullptr, unsigned line_height=24); 



};



