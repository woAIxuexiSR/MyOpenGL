#pragma once

#include <chrono>
#include <iostream>

#define TICK(x) auto start_##x = std::chrono::steady_clock::now()
#define TOCK(x) auto end_##x = std::chrono::steady_clock::now();            \
                std::cout << #x " : " << std::chrono::duration_cast<std::chrono::duration<double> >(end_##x - start_##x).count() << "s" << std::endl

