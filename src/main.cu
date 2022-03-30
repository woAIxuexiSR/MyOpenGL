#include <iostream>
#include "helper_cuda.h"
#include "ticktock.h"
#include "image.h"
#include "buffer.h"

int main()
{
    int a[10];
    for(int i = 0; i < 10; i++)
        a[i] = i;
    
    for(int i = 0; i < 10; i++)
        std::cout << a[i] << std::endl;

    return 0;
}