CSGOptiX : expt with OptiX 7 geometry and rendering 
======================================================

tests/CSGOptiXRender.cc
    main that loads and uploads CSGFoundry geometry and creates 
    one or more renders and saves them to jpg   

CSGOptiX.h
    top level struct using either OptiX pre-7 OR 7 

Params.h
    workhorse for communicating between CPU and GPU 

Frame.h
    render pixels holder  

BI.h
    wrapper for OptixBuildInput 
AS.h
    common acceleration structure base struct for GAS and IAS
GAS.h
    bis vector of BI build inputs 
IAS.h
    vector of transforms and d_instances 

GAS_Builder.h
    building OptiX geometry acceleration structure 

IAS_Builder.h
    building OptiX instance acceleration structure 

Binding.h
    GPU/CPU types, including SbtRecord : RaygenData, MissData, HitGroupData (effectively Prim)

PIP.h
    OptiX render pipeline creation from ptx file

SBT.h
    brings together OptiX 7 geometry and render pipeline programs, nexus of control  

Ctx.h
    holder of OptixDeviceContext and Params and Properties instances

Properties.h
    holder of information gleaned from OptiX 7

InstanceId.h
    encode/decode identity info

OPTIX_CHECK.h
    error check macro for optix 7 calls

Six.h
    optix pre-7 rendering of CSGFoundary geometry


 

