#include <gtest/gtest.h>
#include <immintrin.h>
#include <cstddef>
#include "GmUtilities.hpp"

using namespace gm;


TEST(MM128D_GetSet, GetElements) {
    // _mm_set_pd(high, low)
    __m128d v = _mm_set_pd(3.5, 1.25); // lanes: [high=3.5, low=1.25]
    EXPECT_DOUBLE_EQ(mm_128d_get_elem(v, 0), 1.25); // low
    EXPECT_DOUBLE_EQ(mm_128d_get_elem(v, 1), 3.5);  // high
}

TEST(MM128D_GetSet, SetElements) {
    __m128d v = _mm_set_pd(3.5, 1.25);

    // replace low (index 0)
    mm_128d_set_elem(v, 0, 7.0);
    EXPECT_DOUBLE_EQ(mm_128d_get_elem(v, 0), 7.0);
    EXPECT_DOUBLE_EQ(mm_128d_get_elem(v, 1), 3.5);

    // replace high (index 1)
    mm_128d_set_elem(v, 1, -2.5);
    EXPECT_DOUBLE_EQ(mm_128d_get_elem(v, 0), 7.0);
    EXPECT_DOUBLE_EQ(mm_128d_get_elem(v, 1), -2.5);

    // overwrite again
    mm_128d_set_elem(v, 0, 0.0);
    mm_128d_set_elem(v, 1, 42.42);
    EXPECT_DOUBLE_EQ(mm_128d_get_elem(v, 0), 0.0);
    EXPECT_DOUBLE_EQ(mm_128d_get_elem(v, 1), 42.42);
}

TEST(MM128_GetSet, GetElements) {
    // _mm_set_ps(w,z,y,x) -> lanes [3=w,2=z,1=y,0=x]
    __m128 v = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);
    EXPECT_FLOAT_EQ(mm_128_get_elem(v, 0), 1.0f);
    EXPECT_FLOAT_EQ(mm_128_get_elem(v, 1), 2.0f);
    EXPECT_FLOAT_EQ(mm_128_get_elem(v, 2), 3.0f);
    EXPECT_FLOAT_EQ(mm_128_get_elem(v, 3), 4.0f);
}

TEST(MM128_GetSet, SetElements) {
    __m128 v = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);

    // replace lane 0 (x)
    mm_128_set_elem(v, 0, 9.5f);
    EXPECT_FLOAT_EQ(mm_128_get_elem(v, 0), 9.5f);
    EXPECT_FLOAT_EQ(mm_128_get_elem(v, 1), 2.0f);
    EXPECT_FLOAT_EQ(mm_128_get_elem(v, 2), 3.0f);
    EXPECT_FLOAT_EQ(mm_128_get_elem(v, 3), 4.0f);

    // replace lane 3 (w)
    mm_128_set_elem(v, 3, -1.25f);
    EXPECT_FLOAT_EQ(mm_128_get_elem(v, 3), -1.25f);

    // replace middle lanes
    mm_128_set_elem(v, 1, 0.125f);
    mm_128_set_elem(v, 2, 7.75f);
    EXPECT_FLOAT_EQ(mm_128_get_elem(v, 0), 9.5f);
    EXPECT_FLOAT_EQ(mm_128_get_elem(v, 1), 0.125f);
    EXPECT_FLOAT_EQ(mm_128_get_elem(v, 2), 7.75f);
    EXPECT_FLOAT_EQ(mm_128_get_elem(v, 3), -1.25f);
}
