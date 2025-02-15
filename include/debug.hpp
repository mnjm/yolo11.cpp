#pragma once

#include <iostream>

#ifdef DEBUG_MODE
#define DEBUG_PRINT(x) (std::cout << x << std::endl)
#define DEBUG_PRINT_MAT_SHAPE(mat)                                       \
    do                                                                   \
    {                                                                    \
        std::cout << "Mat: " #mat << " dims=" << mat.dims << ", size=["; \
        for (int i = 0; i < mat.dims; i++)                               \
        {                                                                \
            std::cout << mat.size[i] << (i < mat.dims - 1 ? " x " : ""); \
        }                                                                \
        std::cout << "], channels=" << mat.channels() << std::endl;      \
    } while (0)
#else
#define DEBUG_PRINT(x)
#define DEBUG_PRINT_MAT_SHAPE(mat)
#endif