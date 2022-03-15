#pragma once

#include <string>
#include <cuda_runtime.h>
#include "helper_cuda.h"

class ImageBuffer
{
public:
    
    int width, height, nchannel;
    unsigned char* data;


    ImageBuffer(int _w, int _h, int _n = 3);
    __device__ void write(int w, int h, float3 c);
    __host__ void save(const std::string& file);
};