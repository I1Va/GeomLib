#pragma once

#include <cmath>
#include <random>
#include <cstdio>
#include <immintrin.h>
#include <omp.h>

namespace gm
{

class GlobalMPRandomGenerator {
    std::vector<std::minstd_rand> threadsGenerators;
public:
    // one thread default case
    GlobalMPRandomGenerator() : threadsGenerators(0) {}

    void setThreadsNum(const size_t threadsNum) {
        threadsGenerators.resize(threadsNum);
    }
    void setThreadSeed(const size_t tid, const int seed) {
        threadsGenerators[tid].seed(seed);
    }
    std::minstd_rand &getThreadGenerator(const size_t tid) {
        return threadsGenerators[tid];
    }
};

void setThreadsNum(const size_t threadsNum);
void setThreadSeed(const int seed);

double randomDouble();
float randomFloat();

void   mm_256d_set_elem(__m256d &v, std::size_t n, double value);
double mm_256d_get_elem(const __m256d &v, std::size_t n);

double mm_128d_get_elem(const __m128d &v, std::size_t n);
void   mm_128d_set_elem(__m128d &v, std::size_t, double value);

float  mm_128_get_elem(const __m128 &v, std::size_t n);
void   mm_128_set_elem(__m128 &v, std::size_t n, float value);


std::ostream &operator<<(std::ostream &stream, const __m256d &v);
std::ostream &operator<<(std::ostream &stream, const __m128d &v);

inline double degrees_to_radians(double degrees) {
    return degrees * M_PI / 180.0;
}

inline double randomDouble(double min, double max) {
    return min + (max-min)*randomDouble();
}

inline double randomFloat(double min, double max) {
    return min + (max-min)*randomFloat();
}
}