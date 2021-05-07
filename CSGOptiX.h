#pragma once

#include <optix.h>
#include <glm/fwd.hpp>

struct CSGFoundry ; 
struct View ; 
struct Params ; 

#if OPTIX_VERSION < 70000
struct Six ; 
#else
struct Ctx ; 
struct PIP ; 
struct SBT ; 
#endif
struct Frame ; 

struct CSGOptiX
{
    static const char* PTXNAME ; 
    static const char* GEO_PTXNAME ; 
    static const char* ENV(const char* key, const char* fallback);

    unsigned width = 1280u ; 
    unsigned height = 720u ; 
    unsigned depth = 1u ; 

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


    CSGOptiX(const CSGFoundry* foundry); 
    void init(); 
    void setCE(const glm::vec4& ce, float tmin_model, float tmax_model );
    void render(const char* tspec); 

    void render_flightpath(); 

};



