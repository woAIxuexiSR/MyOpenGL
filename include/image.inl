#include "image.h"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif

ImageBuffer::ImageBuffer(int _w, int _h, int _n) : width(_w), height(_h), nchannel(_n)
{
    checkCudaErrors(cudaMalloc(&data, width * height * nchannel * sizeof(unsigned char)));
}

template<class Func>
__global__ void kernel(unsigned char* data, int w, int h, int n, Func paint)
{
    int ww = w / gridDim.x, hh = h / blockDim.x;

    int x0 = blockIdx.x * ww, y0 = threadIdx.x * hh;
    for(int i = 0; i < ww; i++)
    {
        for(int j = 0; j < hh; j++)
        {
            int x = x0 + i, y = y0 + j;
            float xp = (float)x / w, yp = (float)y / h;
            float3 color = paint(xp, yp);
            
            int idx = y * w * n + x * n;
            data[idx + 0] = static_cast<unsigned char>(color.x * 255.999f);
            data[idx + 1] = static_cast<unsigned char>(color.y * 255.999f);
            data[idx + 2] = static_cast<unsigned char>(color.z * 255.999f);
        }
    }
}

template<class Func>
void ImageBuffer::render(Func paint)
{
    kernel<<<64, 64>>>(data, width, height, nchannel, paint);
    checkCudaErrors(cudaDeviceSynchronize());
}

void ImageBuffer::save(const std::string &file)
{
    int size = width * height * nchannel;
    unsigned char* dst = new unsigned char[size];
    checkCudaErrors(cudaMemcpy(dst, data, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    stbi_write_jpg(file.c_str(), width, height, nchannel, dst, 0);
}

ImageBuffer::~ImageBuffer()
{
    checkCudaErrors(cudaFree(data));
}