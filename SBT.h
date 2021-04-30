#pragma once

#include <vector>
#include <optix.h>

#include "Binding.h"
#include "GAS.h"
#include "IAS.h"

/**
SBT : RG,MS,HG program data preparation 
===========================================

Aim to minimize geometry specifics in here ...


**/
struct PIP ; 
struct CSGFoundry ; 
struct CSGPrim ; 

struct SBT 
{
    const PIP*    pip ; 
    Raygen*       raygen ;
    Miss*         miss ;
    HitGroup*     hitgroup ;
    HitGroup*     check ;

    const CSGFoundry*  foundry ; 

    bool          is_1NN ;  // 1NN:true is smallest bbox chopped 
    bool          is_11N ; 
 
    CUdeviceptr   d_raygen ;
    CUdeviceptr   d_miss ;
    CUdeviceptr   d_hitgroup ;

    OptixShaderBindingTable sbt = {};

    std::vector<GAS> vgas ; 
    std::vector<IAS> vias ; 
    AS*              top ; 


    SBT( const PIP* pip_ ); 

    AS* getAS(const char* spec) const ;
    void setTop(const char* spec) ;
    void setTop(AS* top_) ;
    AS* getTop() const ;


    void init();  
    void createRaygen();  
    void updateRaygen();  

    void createMiss();  
    void updateMiss();  

    void setFoundry(const CSGFoundry* foundry); 

    void createHitgroup();
    void checkHitgroup();

    void createIAS();
    void createIAS(unsigned ias_idx);

    void createGAS();

    void setPrimData( HitGroupData& data, const CSGPrim* prim);
    void dumpPrimData( const HitGroupData& data ) const ;
    void checkPrimData( HitGroupData& data, const CSGPrim* prim);

    const GAS& getGAS(unsigned gas_idx) const ;
    const IAS& getIAS(unsigned ias_idx) const ;

    unsigned getOffset(unsigned shape_idx_ , unsigned layer_idx_ ) const ; 
    unsigned _getOffset(unsigned shape_idx_ , unsigned layer_idx_ ) const ;

    unsigned getTotalRec() const ;



};

