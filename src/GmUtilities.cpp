#include <random>
#include <cassert>

#include "GmUtilities.hpp"


namespace gm
{

double randomDouble() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

float randomFloat() {
    static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

void mm_256_set_elem(__m256d &v, std::size_t n, double value) {
    switch (n) {
        case 0: {
            __m128d lo = _mm256_castpd256_pd128(v);
            lo = _mm_move_sd(lo, _mm_set_sd(value)); 
            v = _mm256_insertf128_pd(v, lo, 0);
            break;
        }
        case 1: {
            __m128d lo = _mm256_castpd256_pd128(v);
            __m128d tmp = _mm_set_sd(value);
            lo = _mm_shuffle_pd(lo, tmp, 0b01); 
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
            __m128d tmp = _mm_set_sd(value);
            hi = _mm_shuffle_pd(hi, tmp, 0b01); 
            v = _mm256_insertf128_pd(v, hi, 1);
            break;
        }
        default:
            assert(0 && "n > 3");
    }
}

double mm_256_get_elem(const __m256d &v, std::size_t n) {
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

double mm_128_get_elem(const __m128d &v, int n) {
    assert(n == 0 || n == 1);
    if (n == 0) return _mm_cvtsd_f64(v);
    // n == 1
    __m128d hi = _mm_shuffle_pd(v, v, 0b01); // move upper 64 -> lower 64
    return _mm_cvtsd_f64(hi);
}

void mm_128_set_elem(__m128d &v, int n, double value) {
    assert(n == 0 || n == 1);
    double a, b;
    if (n == 0) {
        // replace low lane, keep high lane
        b = mm_128_get_elem(v, 1);
        v = _mm_set_pd(b, value); // [high, low]
    } else {
        // replace high lane, keep low lane
        a = mm_128_get_elem(v, 0);
        v = _mm_set_pd(value, a); // [high, low]
    }
}

}