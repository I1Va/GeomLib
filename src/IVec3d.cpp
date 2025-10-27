#include <iostream>

#include "GmUtilities.hpp"


namespace gm
{

std::ostream &operator<<(std::ostream &stream, const __m256d &v) {
    stream << "[";
    for (std::size_t i = 0; i < 3; i++)
        stream << mm_256_get_elem(v, i) << ", ";
    stream << mm_256_get_elem(v, 3) << "]";

    return stream;
}

std::ostream &operator<<(std::ostream &stream, const __m128d &v) {
    stream << "[";
    for (std::size_t i = 0; i < 1; i++)
        stream << mm_128_get_elem(v, i) << ", ";
    stream << mm_128_get_elem(v, 1) << "]";

    return stream;
}

}