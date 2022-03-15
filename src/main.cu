#include <iostream>
#include <cuda_runtime.h>
#include "image.h"


template<class Func>
__global__ void kernel(int w, int h, Func func)
{
    int ww = w / gridDim.x, hh = h / blockDim.x;
    int wid = blockIdx.x * ww, hid = threadIdx.x * hh;

    for(int i = 0; i < ww; i++)
    {
        for(int j = 0; j < hh; j++)
        {
            int pw = wid + i, ph = hid + j;
            float r = pw / (float)w, g = ph / (float)h, b = 0.0f;
            func(pw, ph, float3{r, g, b});
        }
    }
}

template<class Func>
__global__ void test(int w, int h, Func func)
{
    for(int i = 0; i < w; i++)
    {
        for(int j = 0; j < h; j++)
        {
            float r = i / (float)w, g = j / (float)h, b = 0.0f;
            func(i, j, float3{r, g, b});
        }
    }
}

int main()
{
    std::cout << 1 << std::endl;

    ImageBuffer ib(800, 600);

    // kernel<<<40, 20>>>(800, 600, [data=ib.data] __device__ (int w, int h, float3 c) {
    //     int idx = h * 800 * 3 + w * 3;
    //     data[idx + 0] = static_cast<int>(c.x * 255.999f);
    //     data[idx + 1] = static_cast<int>(c.y * 255.999f);
    //     data[idx + 2] = static_cast<int>(c.z * 255.999f);
    // });
    test<<<1, 1>>>(800, 600, [data=ib.data] __device__ (int w, int h, float3 c) {
        int idx = h * 800 * 3 + w * 3;
        data[idx + 0] = static_cast<int>(c.x * 255.999f);
        data[idx + 1] = static_cast<int>(c.y * 255.999f);
        data[idx + 2] = static_cast<int>(c.z * 255.999f);
    });
    
    checkCudaErrors(cudaDeviceSynchronize());

    ib.save("test.jpg");
    std::cout << "hahaha" << std::endl;
    return 0;
}