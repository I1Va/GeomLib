#pragma once

#include <cmath>
#include <random>
#include <cstdio>
#include <immintrin.h>
#include <omp.h>

namespace gm
{

class GlobalMPRandomGenerator {
    std::vector<std::mt19937> generators;
    std::vector<size_t> threadGeneratorId;
public:
    GlobalMPRandomGenerator() {
        // one thread default case
        generators.resize(1);
        threadGeneratorId.resize(1);
        threadGeneratorId[0] = 0;
    }

    void resetRandomGenerator(const size_t generatorsNum, const size_t theadsNum) {
        generators.resize(generatorsNum);
        threadGeneratorId.resize(theadsNum);
        for (size_t i = 0; i < generators.size(); i++) {
            generators[i].seed(i);
        }
    }

    void setThreadGeneratorId(const size_t tid, const size_t generatorId) {
        threadGeneratorId[tid] = generatorId;
    }

    std::mt19937 &getThreadGenerator(const size_t tid) {
        return generators[threadGeneratorId[tid]];
    }
};

void resetRandomGenerator(const size_t generatorsNum, const size_t theadsNum);
void setThreadGeneratorId(const size_t generatorId);

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