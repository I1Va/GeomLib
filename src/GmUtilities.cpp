#include <random>
#include <cassert>
#include <iostream>
#include <omp.h>
#include "GmUtilities.hpp"

namespace gm
{

GlobalMPRandomGenerator globalGenerator;

void setThreadSeed(const int seed) {
    globalGenerator.setThreadSeed(omp_get_thread_num(), seed);
}

void setThreadsNum(const size_t threadsNum) {
    globalGenerator.setThreadsNum(threadsNum);
}

double randomDouble() {
    thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(globalGenerator.getThreadGenerator(omp_get_thread_num()));
}

float randomFloat() {
    thread_local std::uniform_real_distribution<float> distribution(0.0, 1.0);
    return distribution(globalGenerator.getThreadGenerator(omp_get_thread_num()));
}

void mm_256d_set_elem(__m256d &v, std::size_t n, double value) {
    assert(n < 4);
    switch (n) {
        case 0: {
            __m128d lo = _mm256_castpd256_pd128(v);
            __m128d hi = _mm256_extractf128_pd(v, 1);

            lo = _mm_move_sd(lo, _mm_set_sd(value)); 
            v = _mm256_set_m128d(hi, lo);
            break;
        }
        case 1: {
            __m128d lo = _mm256_castpd256_pd128(v);
            double low = _mm_cvtsd_f64(lo);      
            lo = _mm_set_pd(value, low);              
            v = _mm256_insertf128_pd(v, lo, 0);
            break;
        }
        case 2: {
            __m128d hi = _mm256_extractf128_pd(v, 1);

            hi = _mm_move_sd(hi, _mm_set_sd(value));
            v = _mm256_insertf128_pd(v, hi, 1);
            break;
        }
        case 3: {
            __m128d hi = _mm256_extractf128_pd(v, 1);
            double lowhi = _mm_cvtsd_f64(hi);          
            hi = _mm_set_pd(value, lowhi);            
            v = _mm256_insertf128_pd(v, hi, 1);
            break;
        }
    }
}

double mm_256d_get_elem(const __m256d &v, std::size_t n) {
    switch (n) {
        case 0: 
            return _mm256_cvtsd_f64(v);
        
        case 1: {
            __m128d lo = _mm256_castpd256_pd128(v);
            return _mm_cvtsd_f64(_mm_shuffle_pd(lo, lo, 0b01));
        }
        
        case 2: {
            __m128d hi = _mm256_extractf128_pd(v, 1);
            return _mm_cvtsd_f64(hi);
        }

        case 3: {
            __m128d hi = _mm256_extractf128_pd(v, 1);
            return _mm_cvtsd_f64(_mm_shuffle_pd(hi, hi, 0b01));
        }
    
        default:
            assert(0 && "n > 3");
            return 0.0;
    }
}

double mm_128d_get_elem(const __m128d &v, std::size_t n) {
    assert(n == 0 || n == 1);
    if (n == 0) return _mm_cvtsd_f64(v);
    // n == 1
    __m128d hi = _mm_shuffle_pd(v, v, 0b01); // move upper 64 -> lower 64
    return _mm_cvtsd_f64(hi);
}

void mm_128d_set_elem(__m128d &v, std::size_t n, double value) {
    assert(n == 0 || n == 1);

    double a, b;
    if (n == 0) {
        b = mm_128d_get_elem(v, 1);
        v = _mm_set_pd(b, value);
    } else {
        a = mm_128d_get_elem(v, 0);
        v = _mm_set_pd(value, a);
    }
}

float mm_128_get_elem(const __m128 &v, std::size_t n) {
    assert(n < 4);

    switch (n) {
        case 0: return _mm_cvtss_f32(v);

        case 1: {
            __m128 sh = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
            return _mm_cvtss_f32(sh);
        }
        case 2: {
            __m128 sh = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
            return _mm_cvtss_f32(sh);
        }
        case 3: {
            __m128 sh = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));
            return _mm_cvtss_f32(sh);
        }
    }

    return 0;
}

void mm_128_set_elem(__m128 &v, std::size_t n, float value) {
    assert(n >= 0 && n < 4);
    __m128 val = _mm_set1_ps(value);

    const int m3 = (n == 3) ? -1 : 0;
    const int m2 = (n == 2) ? -1 : 0;
    const int m1 = (n == 1) ? -1 : 0;
    const int m0 = (n == 0) ? -1 : 0;
    __m128 mask = _mm_castsi128_ps(_mm_set_epi32(m3, m2, m1, m0));

    v = _mm_or_ps(_mm_and_ps(mask, val), _mm_andnot_ps(mask, v));
}

std::ostream &operator<<(std::ostream &stream, const __m256d &v) {
    stream << "[";
    for (std::size_t i = 0; i < 3; i++)
        stream << mm_256d_get_elem(v, i) << ", ";
    stream << mm_256d_get_elem(v, 3) << "]";

    return stream;
}

std::ostream &operator<<(std::ostream &stream, const __m128d &v) {
    stream << "[";
    for (std::size_t i = 0; i < 1; i++)
        stream << mm_128d_get_elem(v, i) << ", ";
    stream << mm_128d_get_elem(v, 1) << "]";

    return stream;
}

}