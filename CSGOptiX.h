#pragma once

struct CSGFoundry ; 
struct View ; 
struct Params ; 

#include <glm/fwd.hpp>

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

    CSGOptiX(const CSGFoundry* foundry); 

    void init(); 
    void setCE(const glm::vec4& ce, float tmin, float tmax );
    void render(const char* tspec); 
};



