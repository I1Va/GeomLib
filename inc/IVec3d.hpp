#pragma once

#include <immintrin.h>
#include <iostream>
#include <cassert>

#include "GmUtilities.hpp"

#ifdef GM_DEBUG
    #define GM_ASSERT_VEC3(v) assert( (v).isFinite() )
    #define GM_ASSERT_VEC2(v) assert( (v).isFinite() )
    #define GM_ASSERT(cond) assert(cond)
#else
    #define GM_ASSERT_VEC3(v) ((void)0)
    #define GM_ASSERT_VEC2(v) ((void)0)
    #define GM_ASSERT(cond) ((void)0)
#endif

namespace gm 
{

std::ostream &operator<<(std::ostream &stream, const __m256d &v);
std::ostream &operator<<(std::ostream &stream, const __m128d &v);

/* -------------------- IVec3d -------------------- */
class IVec3d {
    __m256d cords_;
    mutable double len2Cache_;
    mutable bool len2Dirty_;

public:
    IVec3d() noexcept: cords_(_mm256_setzero_pd()), len2Cache_(), len2Dirty_(true) {}
    IVec3d(double x, double y, double z) noexcept: cords_(_mm256_set_pd(0, z, y, x)), len2Cache_(), len2Dirty_(true) {}
    explicit IVec3d(__m256d cords) noexcept: cords_(cords), len2Cache_(), len2Dirty_(true) {}
    explicit IVec3d(double val) noexcept: cords_(_mm256_set1_pd(val)), len2Cache_(), len2Dirty_(true) {}

    IVec3d(const IVec3d &other) noexcept
        : cords_(other.cords_),
          len2Cache_(), len2Dirty_(true) {}

    // ------------------------------------ Operators ------------------------------------
    IVec3d operator+(const IVec3d &other) const {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        __m256d reCords = _mm256_add_pd(cords_, other.cords_);

        return IVec3d(reCords);
    }


    IVec3d operator-(const IVec3d &other) const {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        __m256d reCords = _mm256_sub_pd(cords_, other.cords_);

        return IVec3d(reCords);
    }

    IVec3d operator*(const IVec3d &other) const {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        __m256d reCords = _mm256_mul_pd(cords_, other.cords_);

        return IVec3d(reCords);
    }

    IVec3d operator*(const double scalar) const {
        GM_ASSERT_VEC3(*this);

        __m256d scalarVec = _mm256_set1_pd(scalar);
        __m256d reCords = _mm256_mul_pd(cords_, scalarVec);

        return IVec3d(reCords);
    }

    IVec3d operator/(const double scalar) const {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT(scalar != 0);

        __m256d scalarVec = _mm256_set1_pd(scalar);
        __m256d reCords = _mm256_div_pd(cords_, scalarVec);

        return IVec3d(reCords);
    }

    IVec3d operator+=(const IVec3d &other) {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        
        cords_ = _mm256_add_pd(cords_, other.cords_);
        len2Dirty_ = true;
        
        return *this;
    }

    IVec3d operator-=(const IVec3d &other) {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        cords_ = _mm256_sub_pd(cords_, other.cords_);
        len2Dirty_ = true;
        
        return *this;
    }

    IVec3d operator*=(const IVec3d &other) {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        cords_ = _mm256_mul_pd(cords_, other.cords_);
        len2Dirty_ = true;
        
        return *this;
    }

    IVec3d operator/=(const double scalar) {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT(scalar != 0);

        __m256d scalarVec = _mm256_set1_pd(scalar);
        cords_ = _mm256_div_pd(cords_, scalarVec);
        len2Dirty_ = true;
        
        return *this;
    }

    IVec3d &operator=(const IVec3d &other) = default;

    // --------------------------------------- Math -----------------------------------------------
    [[nodiscard]] IVec3d clamped(const double min, const double max) const {
        GM_ASSERT_VEC3(*this);

        __m256d minVec = _mm256_set1_pd(min);
        __m256d maxVec = _mm256_set1_pd(max);

        __m256d clamped = _mm256_max_pd(cords_, minVec);
                clamped = _mm256_min_pd(clamped, maxVec); 

        return IVec3d(clamped);
    }

    void setAngle(const IVec3d &axis, const double radians) {
        IVec3d k = axis.normalized();
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

    void rotate(const IVec3d &axis, const double deltaRadians) {
        IVec3d k = axis.normalized();

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

    bool isFinite() const {
        return std::isfinite(x()) && std::isfinite(y()) && std::isfinite(z());
    }

    inline bool nearZero() const {
        const double eps = std::numeric_limits<double>::epsilon();
        return (std::fabs(x()) < eps) && (std::fabs(y()) < eps) && (std::fabs(z()) < eps);
    }

    [[nodiscard]] IVec3d normalized() const {
        double len = length();
        return len > double(0) ? *this / len : IVec3d();
    }

    // --------------------------------------- Getters --------------------------------------------
    double x() const { return mm_256_get_elem(cords_, 0); }
    double y() const { return mm_256_get_elem(cords_, 1); }    
    double z() const { return mm_256_get_elem(cords_, 2); }
    __m256d cords() const { return cords_; }

    double length() const { return std::sqrt(length2()); }
    double length2() const 
    {
        if (len2Dirty_) {
           
            __m256d v2 = _mm256_mul_pd(cords_, cords_);
            __m256d sum = _mm256_hadd_pd(v2, v2);      // [w^2 + z^2, w^2 + z^2, y^2 + x^2, y^2 + x^2]
            len2Cache_ = mm_256_get_elem(sum, 0) + mm_256_get_elem(sum, 3);
        
            len2Dirty_ = false;
        }
        return len2Cache_;
    }

    static IVec3d random() {
        return IVec3d(randomDouble(), randomDouble(), randomDouble());
    }

    static IVec3d random(double min, double max) {
        return IVec3d(randomDouble(min,max), randomDouble(min,max), randomDouble(min,max));
    }

    static IVec3d randomUnit() {
        IVec3d result(randomDouble(), randomDouble(), randomDouble());
        return result.normalized();
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
    friend std::ostream& operator<<(std::ostream& os, const IVec3d& v) {
        return os << "vec3{" << v.x() << ", " << v.y() << ", " << v.z() << "}";
    }

    friend std::istream& operator>>(std::istream& is, IVec3d& v) {
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



// --------------------- Free functions -------------------------------------------------------
inline double dot(const IVec3d &a, const IVec3d &b) noexcept {
    GM_ASSERT_VEC3(a);
    GM_ASSERT_VEC3(b);

     __m256d mul = _mm256_mul_pd(a.cords(), b.cords());

    // low 128: [e0, e1], high 128: [e2, e3]
    __m128d lo = _mm256_castpd256_pd128(mul);
    __m128d hi = _mm256_extractf128_pd(mul, 1);

    // sum e0 + e1 using horizontal add on the low half
    __m128d sum_lo = _mm_hadd_pd(lo, lo);         // [e0+e1, e0+e1]
    double e01 = _mm_cvtsd_f64(sum_lo);           // extract scalar e0+e1

    // extract e2 (low element of hi) and add
    double e2 = _mm_cvtsd_f64(hi);

    return e01 + e2;
}

inline IVec3d getOrtogonal(const IVec3d &a, const IVec3d &b) {
    double bLen2 = b.length2();
    double c = dot(a, b) / bLen2;
    return a - b * c;
}

inline IVec3d cross(const IVec3d &a, const IVec3d &b) noexcept {
    GM_ASSERT_VEC3(a);
    GM_ASSERT_VEC3(b);
    return IVec3d{
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x()
    };
}

inline IVec3d cordPow(const IVec3d &base, const IVec3d &exp) {
    GM_ASSERT_VEC3(base);
    GM_ASSERT_VEC3(exp);
    GM_ASSERT(base.x() >= 0 && base.y() >= 0 && base.z() >= 0);

    return IVec3d{
        std::pow(static_cast<double>(base.x()), exp.x()),
        std::pow(static_cast<double>(base.y()), exp.y()),
        std::pow(static_cast<double>(base.z()), exp.z())
    };
}


class IPoint3 {
    __m256d cords_; // [w, z, y, x] when created with _mm256_set_pd

public:
    IPoint3() noexcept : cords_(_mm256_setzero_pd()) {}
    IPoint3(double x, double y, double z) noexcept : cords_(_mm256_set_pd(0.0, z, y, x)) {}
    explicit IPoint3(__m256d cords) noexcept : cords_(cords) {}
    explicit IPoint3(double v) noexcept : cords_(_mm256_set1_pd(v)) {}

    // copy / assign
    IPoint3(const IPoint3& o) noexcept = default;
    IPoint3& operator=(const IPoint3& o) noexcept = default;

    // Point - Point = Vector (requires IVec3d(__m256d))
    IVec3d operator-(const IPoint3 &other) const noexcept {
        return IVec3d(_mm256_sub_pd(cords_, other.cords_));
    }

    // Point + Vector = Point
    IPoint3 operator+(const IVec3d &vec) const noexcept {
        return IPoint3(_mm256_add_pd(cords_, vec.cords()));
    }

    // Point - Vector = Point
    IPoint3 operator-(const IVec3d &vec) const noexcept {
        return IPoint3(_mm256_sub_pd(cords_, vec.cords()));
    }

    // Scalar ops
    IPoint3 operator*(double s) const noexcept {
        __m256d sv = _mm256_set1_pd(s);
        return IPoint3(_mm256_mul_pd(cords_, sv));
    }
    IPoint3 operator/(double s) const noexcept {
        assert(s != 0.0);
        __m256d sv = _mm256_set1_pd(s);
        return IPoint3(_mm256_div_pd(cords_, sv));
    }

    // Accessors (no memory stores)
    double x() const noexcept { return _mm256_cvtsd_f64(cords_); }

    double y() const noexcept {
        __m128d lo = _mm256_castpd256_pd128(cords_);               // [y, x]
        __m128d sh = _mm_shuffle_pd(lo, lo, 0b01);                 // [x, y] -> move high to low
        return _mm_cvtsd_f64(sh);                                  // low of shuffled -> original y
    }

    double z() const noexcept {
        __m128d hi = _mm256_extractf128_pd(cords_, 1);             // [w, z] in hi: low= z
        return _mm_cvtsd_f64(hi);
    }

    // Setters (modify in-place)
    void setX(const double scalar) { mm_256_set_elem(cords_, 0, scalar); }
    void setY(const double scalar) { mm_256_set_elem(cords_, 1, scalar); }
    void setZ(const double scalar) { mm_256_set_elem(cords_, 2, scalar); }

    // stream operators
    friend std::ostream& operator<<(std::ostream& os, const IPoint3& p) {
        return os << "point3{" << p.x() << ", " << p.y() << ", " << p.z() << "}";
    }
    friend std::istream& operator>>(std::istream& is, IPoint3& p) {
        double a, b, c;
        is >> a >> b >> c;
        if (is) { p.setX(a); p.setY(b); p.setZ(c); }
        return is;
    }
};


/* -------------------- IVec2 -------------------- */
class IVec2 {
    __m128d cords_;            // layout: _mm_set_pd(y, x) => cords_[0]=x, cords_[1]=y
    mutable double len2Cache_;
    mutable bool len2Dirty_;

public:
    IVec2() noexcept : cords_(_mm_setzero_pd()), len2Cache_(0.0), len2Dirty_(true) {}
    IVec2(double x, double y) noexcept : cords_(_mm_set_pd(y, x)), len2Cache_(0.0), len2Dirty_(true) {}
    explicit IVec2(__m128d cords) noexcept : cords_(cords), len2Cache_(0.0), len2Dirty_(true) {}
    explicit IVec2(double val) noexcept : cords_(_mm_set1_pd(val)), len2Cache_(0.0), len2Dirty_(true) {}

    // copy / assign default
    IVec2(const IVec2& o) noexcept = default;
    IVec2& operator=(const IVec2& o) noexcept = default;

    // arithmetic
    IVec2 operator+(const IVec2 &o) const noexcept { return IVec2(_mm_add_pd(cords_, o.cords_)); }
    IVec2 operator-(const IVec2 &o) const noexcept { return IVec2(_mm_sub_pd(cords_, o.cords_)); }
    IVec2 operator*(double s) const noexcept {
        __m128d sv = _mm_set1_pd(s);
        return IVec2(_mm_mul_pd(cords_, sv));
    }
    IVec2 operator/(double s) const noexcept {
        assert(s != 0.0);
        __m128d sv = _mm_set1_pd(s);
        return IVec2(_mm_div_pd(cords_, sv));
    }

    IVec2& operator+=(const IVec2 &o) noexcept { cords_ = _mm_add_pd(cords_, o.cords_); len2Dirty_ = true; return *this; }
    IVec2& operator-=(const IVec2 &o) noexcept { cords_ = _mm_sub_pd(cords_, o.cords_); len2Dirty_ = true; return *this; }
    IVec2& operator*=(double s) noexcept { __m128d sv = _mm_set1_pd(s); cords_ = _mm_mul_pd(cords_, sv); len2Dirty_ = true; return *this; }
    IVec2& operator/=(double s) noexcept { assert(s != 0.0); __m128d sv = _mm_set1_pd(s); cords_ = _mm_div_pd(cords_, sv); len2Dirty_ = true; return *this; }

    // accessors
    double x() const noexcept { return mm_128_get_elem(cords_, 0); }
    double y() const noexcept { return mm_128_get_elem(cords_, 1); }
    __m128d cords() const { return cords_; }

    void setX(double v) noexcept { mm_128_set_elem(cords_, 0, v); len2Dirty_ = true; }
    void setY(double v) noexcept { mm_128_set_elem(cords_, 1, v); len2Dirty_ = true; }

    // math
    double dot(const IVec2 &o) const noexcept {
        __m128d m = _mm_mul_pd(cords_, o.cords_);          // [y1*y2, x1*x2]
        __m128d h = _mm_hadd_pd(m, m);                    // [x1*x2 + y1*y2, ...]
        return _mm_cvtsd_f64(h);
    }

    double length2() const noexcept {
        if (len2Dirty_) {
            __m128d m = _mm_mul_pd(cords_, cords_);
            __m128d h = _mm_hadd_pd(m, m);
            len2Cache_ = _mm_cvtsd_f64(h);
            len2Dirty_ = false;
        }
        return len2Cache_;
    }

    double length() const noexcept { return std::sqrt(length2()); }

    IVec2 normalized() const noexcept {
        double len = length();
        if (len == 0.0) return IVec2(0.0, 0.0);
        return *this / len;
    }

    bool isFinite() const noexcept { return std::isfinite(x()) && std::isfinite(y()); }

    bool nearZero() const noexcept {
        const double eps = std::numeric_limits<double>::epsilon();
        return (std::fabs(x()) < eps) && (std::fabs(y()) < eps);
    }

    // stream
    friend std::ostream& operator<<(std::ostream& os, const IVec2& v) {
        return os << "ivec2{" << v.x() << ", " << v.y() << "}";
    }
    friend std::istream& operator>>(std::istream& is, IVec2& v) {
        double a, b;
        is >> a >> b;
        if (is) {
            v.setX(a);
            v.setY(b);
            v.len2Dirty_ = true;
        }
        return is;
    }
};

/* -------------------- IPoint2 -------------------- */
class IPoint2 {
    __m128d cords_; // same layout: _mm_set_pd(y,x)

public:
    IPoint2() noexcept : cords_(_mm_setzero_pd()) {}
    IPoint2(double x, double y) noexcept : cords_(_mm_set_pd(y, x)) {}
    explicit IPoint2(__m128d cords) noexcept : cords_(cords) {}
    explicit IPoint2(double v) noexcept : cords_(_mm_set1_pd(v)) {}

    // copy / assign default
    IPoint2(const IPoint2& o) noexcept = default;
    IPoint2& operator=(const IPoint2& o) noexcept = default;

    // point - point = vector
    IVec2 operator-(const IPoint2 &o) const noexcept {
        return IVec2(_mm_sub_pd(cords_, o.cords_));
    }

    // point + vector = point
    IPoint2 operator+(const IVec2 &vec) const noexcept {
        return IPoint2(_mm_add_pd(cords_, vec.cords()));
    }

    // point - vector = point
    IPoint2 operator-(const IVec2 &vec) const noexcept {
        return IPoint2(_mm_sub_pd(cords_, vec.cords()));
    }

    // scalar ops
    IPoint2 operator*(double s) const noexcept {
        __m128d sv = _mm_set1_pd(s);
        return IPoint2(_mm_mul_pd(cords_, sv));
    }
    IPoint2 operator/(double s) const noexcept {
        assert(s != 0.0);
        __m128d sv = _mm_set1_pd(s);
        return IPoint2(_mm_div_pd(cords_, sv));
    }

    // accessors
    double x() const noexcept { return mm_128_get_elem(cords_, 0); }
    double y() const noexcept { return mm_128_get_elem(cords_, 1); }

    void setX(double v) noexcept { mm_128_set_elem(cords_, 0, v); }
    void setY(double v) noexcept { mm_128_set_elem(cords_, 1, v); }

    // stream
    friend std::ostream& operator<<(std::ostream& os, const IPoint2& p) {
        return os << "point2{" << p.x() << ", " << p.y() << "}";
    }
    friend std::istream& operator>>(std::istream& is, IPoint2& p) {
        double a, b;
        is >> a >> b;
        if (is) { p.setX(a); p.setY(b); }
        return is;
    }
};

}
