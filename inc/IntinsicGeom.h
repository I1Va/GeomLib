#ifndef GEOM_H
#define GEOM_H

#include <immintrin.h>
#include <concepts>
#include <cmath>
#include <cassert>
#include <limits>
#include <algorithm>
#include <ostream>
#include <istream>
#include <random>


namespace gm 
{

#ifdef GM_DEBUG
    #define GM_ASSERT_VEC3(v) assert( (v).isFinite() )
    #define GM_ASSERT_VEC2(v) assert( (v).isFinite() )
    #define GM_ASSERT(cond) assert(cond)
#else
    #define GM_ASSERT_VEC3(v) ((void)0)
    #define GM_ASSERT_VEC2(v) ((void)0)
    #define GM_ASSERT(cond) ((void)0)
#endif


inline double degrees_to_radians(double degrees) {
    return degrees * M_PI / 180.0;
}

inline double randomDouble() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline double randomDouble(double min, double max) {
    return min + (max-min)*randomDouble();
}

inline void mm_256_set_elem(__m256d &v, size_t n, double value) {
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

inline double mm_256_get_elem(const __m256d &v, size_t n) {
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


// ------------------------------------ 3D specialization ------------------------------------
class IVec3 {
    __m256d cords_;
    mutable double len2Cache_;
    mutable bool len2Dirty_;

public:
    constexpr IVec3() noexcept: cords_(), len2Cache_(), len2Dirty_(true) { _mm256_setzero_pd(); }
    constexpr IVec3(double x, double y, double z) noexcept: cords_(), len2Cache_(), len2Dirty_(true) { _mm256_set_pd(0, z, y, x); }
    constexpr explicit IVec3(__m256d cords) noexcept: cords_(cords), len2Cache_(), len2Dirty_(true) { }
    constexpr explicit IVec3(double val) noexcept: cords_(), len2Cache_(), len2Dirty_(true) { _mm256_set1_pd(val); }

    constexpr IVec3(const IVec3 &other) noexcept
        : cords_(other.cords_),
          len2Cache_(), len2Dirty_(true) {}

    // ------------------------------------ Operators ------------------------------------
    constexpr IVec3 operator+(const IVec3 &other) const {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        __m256d reCords = _mm256_add_pd(cords_, other.cords_);

        return IVec3(reCords);
    }


    constexpr IVec3 operator-(const IVec3 &other) const {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        __m256d reCords = _mm256_sub_pd(cords_, other.cords_);

        return IVec3(reCords);
    }

    constexpr IVec3 operator*(const IVec3 &other) const {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        __m256d reCords = _mm256_mul_pd(cords_, other.cords_);

        return IVec3(reCords);
    }

    constexpr IVec3 operator*(const double scalar) const {
        GM_ASSERT_VEC3(*this);

        __m256d scalarVec = _mm256_set1_pd(scalar);
        __m256d reCords = _mm256_mul_pd(cords_, scalarVec);

        return IVec3(reCords);
    }

    constexpr IVec3 operator/(const double scalar) const {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT(scalar != 0);

        __m256d scalarVec = _mm256_set1_pd(scalar);
        __m256d reCords = _mm256_div_pd(cords_, scalarVec);

        return IVec3(reCords);
    }

    constexpr IVec3 operator+=(const IVec3 &other) {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        
        cords_ = _mm256_add_pd(cords_, other.cords_);
        len2Dirty_ = true;
        
        return *this;
    }

    constexpr IVec3 operator-=(const IVec3 &other) {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        cords_ = _mm256_sub_pd(cords_, other.cords_);
        len2Dirty_ = true;
        
        return *this;
    }

    constexpr IVec3 operator*=(const IVec3 &other) {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        cords_ = _mm256_mul_pd(cords_, other.cords_);
        len2Dirty_ = true;
        
        return *this;
    }

    constexpr IVec3 operator/=(const double scalar) {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT(scalar != 0);

        __m256d scalarVec = _mm256_set1_pd(scalar);
        cords_ = _mm256_div_pd(cords_, scalarVec);
        len2Dirty_ = true;
        
        return *this;
    }

    IVec3 &operator=(const IVec3 &other) = default;

    // --------------------------------------- Math -----------------------------------------------
    [[nodiscard]] constexpr IVec3 clamped(const double min, const double max) const {
        GM_ASSERT_VEC3(*this);

        __m256d minVec = _mm256_set1_pd(min);
        __m256d maxVec = _mm256_set1_pd(max);

        __m256d clamped = _mm256_max_pd(cords_, minVec);
                clamped = _mm256_min_pd(cords_, maxVec); 

        return IVec3(clamped);
    }

    void setAngle(const IVec3 &axis, const double radians) {
        IVec3 k = axis.normalized();
        double ux = k.x(), uy = k.y(), uz = k.z();

        double c = std::cos(radians);
        double s = std::sin(radians);
        double oneC = 1 - c;

        cords_ = _mm256_set_pd
        (
            0,                                                                                   // w
            x() * (uz*ux*oneC - uy*s)  + y() * (uz*uy*oneC + ux*s)  + z() * (c + uz*uz*oneC),    // z
            x() * (uy*ux*oneC + uz*s)  + y() * (c + uy*uy*oneC)     + z() * (uy*uz*oneC - ux*s), // y
            x() * (c + ux*ux*oneC)     + y() * (ux*uy*oneC - uz*s)  + z() * (ux*uz*oneC + uy*s)  // x
        ); // Rodrigues’ rotation formula
    }

    void rotate(const IVec3 &axis, const double deltaRadians) {
        IVec3 k = axis.normalized();

        double ux = k.x(), uy = k.y(), uz = k.z();
        double c = std::cos(deltaRadians);
        double s = std::sin(deltaRadians);
        double oneC = 1 - c;

        cords_ = _mm256_set_pd
        (
            0,                                                                                   // w
            x() * (uz*ux*oneC - uy*s)  + y() * (uz*uy*oneC + ux*s)  + z() * (c + uz*uz*oneC),    // z
            x() * (uy*ux*oneC + uz*s)  + y() * (c + uy*uy*oneC)     + z() * (uy*uz*oneC - ux*s), // y
            x() * (c + ux*ux*oneC)     + y() * (ux*uy*oneC - uz*s)  + z() * (ux*uz*oneC + uy*s)  // x
        ); // Rodrigues’ inremental rotation formula
       
    }

    constexpr bool isFinite() const {
        return std::isfinite(x()) && std::isfinite(y()) && std::isfinite(z());
    }

    inline bool nearZero() const {
        const double eps = std::numeric_limits<double>::epsilon();
        return (fabs(x()) < eps) && (fabs(y()) < eps) && (fabs(z()) < eps);
    }

    [[nodiscard]] constexpr IVec3 normalized() const {
        double len = length();
        return len > double(0) ? *this / len : IVec3();
    }

    // --------------------------------------- Getters --------------------------------------------
    constexpr double x() const { return mm_256_get_elem(cords_, 0); }
    constexpr double y() const { return mm_256_get_elem(cords_, 1); }    
    constexpr double z() const { return mm_256_get_elem(cords_, 2); }

    constexpr double length() const { return std::sqrt(length2()); }
    constexpr double length2() const 
    {
        if (len2Dirty_) {
           
            __m256d v2 = _mm256_mul_pd(cords_, cords_);
            __m256d sum1 = _mm256_hadd_pd(v2, v2);     // [y^2 + x^2, z^2 + 0, y^2 + x^2, z^2 + 0]
            __m128d lo = _mm256_castpd256_pd128(sum1); // [y^2 + x^2, z^2 + 0]
            len2Cache_ = _mm_cvtsd_f64(lo) + _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo));

            len2Dirty_ = false;
        }
        return len2Cache_;
    }

    static IVec3 random() {
        return IVec3(randomDouble(), randomDouble(), randomDouble());
    }

    static IVec3 random(double min, double max) {
        return IVec3(randomDouble(min,max), randomDouble(min,max), randomDouble(min,max));
    }

    // --------------------------------------- Setters --------------------------------------------
    void setX(const double scalar) noexcept
    {   
        mm_256_set_elem(cords_, 0, scalar);
        len2Dirty_ = true;
    }

    void setY(const double scalar) noexcept
    {
        mm_256_set_elem(cords_, 1, scalar);
        len2Dirty_ = true;
    }
    
    void setZ(const double scalar) noexcept
    {
        mm_256_set_elem(cords_, 2, scalar);
        len2Dirty_ = true;
    }

    // ---------------- Stream operators ----------------------------------------------------------
    friend std::ostream& operator<<(std::ostream& os, const IVec3& v) {
        return os << "vec3{" << v.x() << ", " << v.y() << ", " << v.z() << "}";
    }

    friend std::istream& operator>>(std::istream& is, IVec3& v) {
        double x, y, z;
        is >> x >> y >> z;
        if (is) {
            v.setX(x);
            v.setY(y);
            v.setZ(z);
        
            v.len2Dirty_ = true;
        }
        return is;
    }
};

}

#endif // GEOM_H
