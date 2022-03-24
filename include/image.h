#pragma once

#include <string>
#include <cuda_runtime.h>
#include "helper_cuda.h"

class ImageBuffer
{
public:
    int width, height, nchannel;
    unsigned char *data;

    ImageBuffer(int _w, int _h, int _n = 3);

    template<class Func>
    void render(Func paint);

    void save(const std::string &file);

    ~ImageBuffer();
};

#include "image.inl"