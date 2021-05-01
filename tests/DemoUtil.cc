#include <sstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>

#include "sutil_vec_math.h"
#include "DemoUtil.h"
#include <glm/gtx/transform.hpp>


const char* DemoUtil::PTXPath( const char* install_prefix, const char* cmake_target, const char* cu_stem, const char* cu_ext ) // static
{
    std::stringstream ss ; 
    ss << install_prefix
       << "/ptx/"
       << cmake_target
       << "_generated_"
       << cu_stem
       << cu_ext
       << ".ptx" 
       ;
    std::string path = ss.str();
    return strdup(path.c_str()); 
}


template <typename T>
T DemoUtil::ato_( const char* a )   // static 
{
    std::string s(a);
    std::istringstream iss(s);
    T v ; 
    iss >> v ; 
    return v ; 
}

/**
DemoUtil::ParseGridSpec
----------------------

Parse a python style xyz gridspec eg "-10:11:2,-10:11:2,-10:11:2"
into a list of 9 ints.

**/

void DemoUtil::ParseGridSpec(  std::array<int,9>& grid, const char* spec)  // static 
{
    int idx = 0 ; 
    std::stringstream ss(spec); 
    std::string s;
    while (std::getline(ss, s, ',')) 
    {   
        std::stringstream tt(s); 
        std::string t;
        while (std::getline(tt, t, ':')) grid[idx++] = std::atoi(t.c_str()) ; 
    }   

    std::cout << "DemoUtil::ParseGridSpec " << spec << " : " ; 
    for(int i=0 ; i < 9 ; i++) std::cout << grid[i] << " " ; 
    std::cout << std::endl ; 
}



void DemoUtil::GridMinMax(const std::array<int,9>& grid, int3&mn, int3& mx)  // static 
{
    mn.x = grid[0] ; mx.x = grid[1] ; 
    mn.y = grid[3] ; mx.y = grid[4] ; 
    mn.z = grid[6] ; mx.z = grid[7] ; 
}

void DemoUtil::GridMinMax(const std::array<int,9>& grid, int&mn, int& mx)  // static 
{
    for(int a=0 ; a < 3 ; a++)
    for(int i=grid[a*3+0] ; i < grid[a*3+1] ; i+=grid[a*3+2] )
    {
        if( i > mx ) mx = i ; 
        if( i < mn ) mn = i ; 
    }
    std::cout << "DemoUtil::GridMinMax " << mn << " " << mx << std::endl ; 
}

template <typename T>
T DemoUtil::GetEValue(const char* key, T fallback) // static 
{
    const char* sval = getenv(key); 
    T val = sval ? ato_<T>(sval) : fallback ;
    return val ;  
}

template <typename T>
void DemoUtil::GetEVector(std::vector<T>& vec, const char* key, const char* fallback )
{
    const char* sval = getenv(key); 
    std::stringstream ss(sval ? sval : fallback); 
    std::string s ; 
    while(getline(ss, s, ',')) vec.push_back(ato_<T>(s.c_str()));   
}  

void DemoUtil::GetEVec(glm::vec3& v, const char* key, const char* fallback )
{
    std::vector<float> vec ; 
    DemoUtil::GetEVector<float>(vec, key, fallback); 
    std::cout << key << DemoUtil::Present(vec) << std::endl ; 
    assert( vec.size() == 3 ); 
    for(int i=0 ; i < 3 ; i++) v[i] = vec[i] ; 
}

void DemoUtil::GetEVec(glm::vec4& v, const char* key, const char* fallback )
{
    std::vector<float> vec ; 
    DemoUtil::GetEVector<float>(vec, key, fallback); 
    std::cout << key << DemoUtil::Present(vec) << std::endl ; 
    assert( vec.size() == 4 ); 
    for(int i=0 ; i < 4 ; i++) v[i] = vec[i] ; 
}



template <typename T>
std::string DemoUtil::Present(std::vector<T>& vec)
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < vec.size() ; i++) ss << vec[i] << " " ; 
    return ss.str();
}

bool DemoUtil::StartsWith( const char* s, const char* q)  // static
{
    return strlen(q) <= strlen(s) && strncmp(s, q, strlen(q)) == 0 ; 
}

void DemoUtil::DumpGrid(const std::array<int,9>& cl)
{
    int i0 = cl[0] ;
    int i1 = cl[1] ;
    int is = cl[2] ;
    int j0 = cl[3] ;
    int j1 = cl[4] ; 
    int js = cl[5] ;
    int k0 = cl[6] ;
    int k1 = cl[7] ;
    int ks = cl[8] ; 

    unsigned num = 0 ; 
    for(int i=i0 ; i < i1 ; i+=is ) 
    for(int j=j0 ; j < j1 ; j+=js ) 
    for(int k=k0 ; k < k1 ; k+=ks ) 
    {
        std::cout << std::setw(2) << num << " (i,j,k) " << "(" << i << "," << j << "," << k << ") " << std::endl ; 
        num += 1 ; 
    }
}

unsigned DemoUtil::Encode4(const char* s) // static 
{
    unsigned u4 = 0u ; 
    for(unsigned i=0 ; i < std::min(4ul, strlen(s)) ; i++ )
    {
        unsigned u = unsigned(s[i]) ; 
        u4 |= ( u << (i*8) ) ; 
    }
    return u4 ; 
}


template float       DemoUtil::GetEValue<float>(const char* key, float fallback); 
template int         DemoUtil::GetEValue<int>(const char* key,   int  fallback); 
template unsigned    DemoUtil::GetEValue<unsigned>(const char* key,   unsigned  fallback); 
template std::string DemoUtil::GetEValue<std::string>(const char* key,  std::string  fallback); 
template bool        DemoUtil::GetEValue<bool>(const char* key,  bool  fallback); 


template void  DemoUtil::GetEVector<unsigned>(std::vector<unsigned>& vec, const char* key, const char* fallback  ); 
template void  DemoUtil::GetEVector<float>(std::vector<float>& vec, const char* key, const char* fallback  ); 

template std::string DemoUtil::Present<float>(std::vector<float>& ); 
template std::string DemoUtil::Present<unsigned>(std::vector<unsigned>& ); 
template std::string DemoUtil::Present<int>(std::vector<int>& ); 


