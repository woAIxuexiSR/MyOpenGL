#include <iostream>
#include <cuda_runtime.h>
#include <functional>
#include "helper_cuda.h"
#include "image.h"

int main()
{
    int w = 2048, h = 1024;

    ImageBuffer img(w, h);
    img.render([] __device__(float xp, float yp) -> float3 {
        return make_float3(xp, yp, 0.0f);
    });

    img.save("test.jpg");

    return 0;
}