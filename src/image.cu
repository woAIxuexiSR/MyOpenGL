#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

ImageBuffer::ImageBuffer(int _w, int _h, int _n)
    : width(_w), height(_h), nchannel(_n)
{
    checkCudaErrors(cudaMallocManaged(&data, width * height * nchannel * sizeof(unsigned char)));
}

__device__ void ImageBuffer::write(int w, int h, float3 c)
{
    int idx = h * width * nchannel + w * nchannel;
    data[idx + 0] = static_cast<int>(c.x * 255.999f);
    data[idx + 1] = static_cast<int>(c.y * 255.999f);
    data[idx + 2] = static_cast<int>(c.z * 255.999f);
}

__host__ void ImageBuffer::save(const std::string& file)
{
    stbi_write_jpg(file.c_str(), width, height, nchannel, data, -1);
}