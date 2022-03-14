#include <iostream>
#include <string>
#include <functional>
#include <vector>
#include "test.h"

void run(std::function<void(int)> func)
{
    func(4);
}

int main()
{
    run([] (int a) {
        std::cout << a << std::endl;
    });

    func(5);
    return 0;
}