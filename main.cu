#include <iostream>
#include <string>
#include <chrono>
#include <tuple>

#include <cuda_runtime.h>
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


class Image
{
public:
    int width, height, nchannel;
    unsigned char* data;

    Image(int _w, int _h, int _n = 3) : width(_w), height(_h), nchannel(_n)
    {
        checkCudaErrors(cudaMallocManaged(&data, width * height * nchannel * sizeof(unsigned char)));
    }

    __host__ __device__ void set(int y, int x, float r, float g, float b)
    {
        int idx = y * width * nchannel + x * nchannel;
        data[idx] = static_cast<unsigned char>(r * 255.999);
        data[idx + 1] = static_cast<unsigned char>(g * 255.999);
        data[idx + 2] = static_cast<unsigned char>(b * 255.999);
    }

    void save(const std::string& filename)
    {
        stbi_write_jpg(filename.c_str(), width, height, nchannel, data, 0);
    }

};


__global__ void kernel(int w, int h, Image img, int frame)
{
    int ww = w / gridDim.x, hh = h / blockDim.x;

    int x0 = blockIdx.x * ww, y0 = threadIdx.x * hh;
    for(int i = 0; i < ww; i++)
    {
        for(int j = 0; j < hh; j++)
        {
            int x = x0 + i, y = y0 + j;
            float xp = (float)x / w, yp = (float)y / h;
            float r = (frame / 60.0f) * (1 - xp) + (1.0f - frame / 60.0f) * xp;
            float g = (frame / 60.0f) * (1 - yp) + (1.0f - frame / 60.0f) * yp;
            img.set(y, x, r, g, 0.0f);
        }
    }
}


int main()
{
    auto start = std::chrono::steady_clock::now();

    int w = 2048, h = 1024;
    Image img(w, h);

    for(int i = 0; i < 60; i++)
    {
        kernel<<<256, 256>>>(w, h, img, i);
        cudaDeviceSynchronize();

        img.save("image/test" + std::to_string(i) + ".jpg");
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count() << "s" << std::endl;
    return 0;
}