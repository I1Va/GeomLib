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

/* -------------------- IVec3f -------------------- */
class IVec3f {
    __m128 cords_;               // layout: _mm_set_ps(0.0f, z, y, x)
    mutable float len2Cache_;
    mutable bool len2Dirty_;

public:
    IVec3f() noexcept: cords_(_mm_setzero_ps()), len2Cache_(), len2Dirty_(true) {}
    IVec3f(float x, float y, float z) noexcept: cords_(_mm_set_ps(0.0f, z, y, x)), len2Cache_(), len2Dirty_(true) {}
    explicit IVec3f(__m128 cords) noexcept: cords_(cords), len2Cache_(), len2Dirty_(true) {}
    explicit IVec3f(float val) noexcept: cords_(_mm_set1_ps(val)), len2Cache_(), len2Dirty_(true) {}

    IVec3f(const IVec3f &other) noexcept
        : cords_(other.cords_),
          len2Cache_(), len2Dirty_(true) {}

    // ------------------------------------ Operators ------------------------------------
    IVec3f operator+(const IVec3f &other) const noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        __m128 reCords = _mm_add_ps(cords_, other.cords_);
        return IVec3f(reCords);
    }


    IVec3f operator-(const IVec3f &other) const noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        __m128 reCords = _mm_sub_ps(cords_, other.cords_);
        return IVec3f(reCords);
    }

    IVec3f operator*(const IVec3f &other) const noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        __m128 reCords = _mm_mul_ps(cords_, other.cords_);
        return IVec3f(reCords);
    }

    IVec3f operator*(const float scalar) const noexcept {
        GM_ASSERT_VEC3(*this);

        __m128 scalarVec = _mm_set1_ps(scalar);
        __m128 reCords = _mm_mul_ps(cords_, scalarVec);
        return IVec3f(reCords);
    }

    IVec3f operator/(const float scalar) const noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT(scalar != 0.0f);

        __m128 scalarVec = _mm_set1_ps(scalar);
        __m128 reCords = _mm_div_ps(cords_, scalarVec);
        return IVec3f(reCords);
    }

    IVec3f& operator+=(const IVec3f &other) noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        cords_ = _mm_add_ps(cords_, other.cords_);
        len2Dirty_ = true;
        return *this;
    }

    IVec3f& operator-=(const IVec3f &other) noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        cords_ = _mm_sub_ps(cords_, other.cords_);
        len2Dirty_ = true;
        return *this;
    }

    IVec3f& operator*=(const IVec3f &other) noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        cords_ = _mm_mul_ps(cords_, other.cords_);
        len2Dirty_ = true;
        return *this;
    }

    IVec3f& operator/=(const float scalar) noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT(scalar != 0.0f);

        __m128 scalarVec = _mm_set1_ps(scalar);
        cords_ = _mm_div_ps(cords_, scalarVec);
        len2Dirty_ = true;
        return *this;
    }

    IVec3f &operator=(const IVec3f &other) = default;

    // --------------------------------------- Math -----------------------------------------------
    [[nodiscard]] IVec3f clamped(const float min, const float max) const noexcept {
        GM_ASSERT_VEC3(*this);

        __m128 minVec = _mm_set1_ps(min);
        __m128 maxVec = _mm_set1_ps(max);

        __m128 clamped = _mm_max_ps(cords_, minVec);
                clamped = _mm_min_ps(clamped, maxVec);
        return IVec3f(clamped);
    }

    void setAngle(const IVec3f &axis, const float radians) noexcept {
        IVec3f k = axis.normalized();
        float ux = k.x(), uy = k.y(), uz = k.z();

        float c = std::cosf(radians);
        float s = std::sinf(radians);
        float oneC = 1.0f - c;

        cords_ = _mm_set_ps(
            0.0f,
            x() * (uz*ux*oneC - uy*s)  + y() * (uz*uy*oneC + ux*s)  + z() * (c + uz*uz*oneC),
            x() * (uy*ux*oneC + uz*s)  + y() * (c + uy*uy*oneC)     + z() * (uy*uz*oneC - ux*s),
            x() * (c + ux*ux*oneC)     + y() * (ux*uy*oneC - uz*s)  + z() * (ux*uz*oneC + uy*s)
        );
    }

    void rotate(const IVec3f &axis, const float deltaRadians) noexcept {
        IVec3f k = axis.normalized();

        float ux = k.x(), uy = k.y(), uz = k.z();
        float c = std::cosf(deltaRadians);
        float s = std::sinf(deltaRadians);
        float oneC = 1.0f - c;

        cords_ = _mm_set_ps(
            0.0f,
            x() * (uz*ux*oneC - uy*s)  + y() * (uz*uy*oneC + ux*s)  + z() * (c + uz*uz*oneC),
            x() * (uy*ux*oneC + uz*s)  + y() * (c + uy*uy*oneC)     + z() * (uy*uz*oneC - ux*s),
            x() * (c + ux*ux*oneC)     + y() * (ux*uy*oneC - uz*s)  + z() * (ux*uz*oneC + uy*s)
        );
    }

    bool isFinite() const noexcept {
        return std::isfinite(x()) && std::isfinite(y()) && std::isfinite(z());
    }

    inline bool nearZero() const noexcept {
        const float eps = std::numeric_limits<float>::epsilon();
        return (std::fabsf(x()) < eps) && (std::fabsf(y()) < eps) && (std::fabsf(z()) < eps);
    }

    [[nodiscard]] IVec3f normalized() const noexcept {
        float len = length();
        return len > 0.0f ? *this / len : IVec3f();
    }

    // --------------------------------------- Getters --------------------------------------------
    float x() const noexcept { return mm_128_get_elem(cords_, 0); }
    float y() const noexcept { return mm_128_get_elem(cords_, 1); }
    float z() const noexcept { return mm_128_get_elem(cords_, 2); }

    __m128 cords() const noexcept { return cords_; }

    float length() const noexcept { return std::sqrtf(length2()); }
    float length2() const noexcept
    {
        if (len2Dirty_) {
            __m128 v2 = _mm_mul_ps(cords_, cords_);
            __m128 sum = _mm_hadd_ps(v2, v2);      // [x+y, z+0, x+y, z+0]
            float e01 = _mm_cvtss_f32(sum);
            float e2 = _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1,1,1,1)));
            len2Cache_ = e01 + e2;

            len2Dirty_ = false;
        }
        return len2Cache_;
    }

    static IVec3f random() {
        return IVec3f(randomFloat(), randomFloat(), randomFloat());
    }

    static IVec3f random(float min, float max) {
        return IVec3f(randomFloat(min,max), randomFloat(min,max), randomFloat(min,max));
    }

    static IVec3f randomUnit() {
        IVec3f result(randomFloat(), randomFloat(), randomFloat());
        return result.normalized();
    }

    // --------------------------------------- Setters --------------------------------------------
    void setX(const float scalar) noexcept
    {
        mm_128_set_elem(cords_, 0, scalar);
        len2Dirty_ = true;
    }

    void setY(const float scalar) noexcept
    {
        mm_128_set_elem(cords_, 1, scalar);
        len2Dirty_ = true;
    }
    
    void setZ(const float scalar) noexcept
    {
        mm_128_set_elem(cords_, 2, scalar);
        len2Dirty_ = true;
    }

    // ---------------- Stream operators ----------------------------------------------------------
    friend std::ostream& operator<<(std::ostream& os, const IVec3f& v) {
        return os << "vec3{" << v.x() << ", " << v.y() << ", " << v.z() << "}";
    }

    friend std::istream& operator>>(std::istream& is, IVec3f& v) {
        float x, y, z;
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
inline float dot(const IVec3f &a, const IVec3f &b) noexcept {
    GM_ASSERT_VEC3(a);
    GM_ASSERT_VEC3(b);

    __m128 mul = _mm_mul_ps(a.cords(), b.cords());
    __m128 sum = _mm_hadd_ps(mul, mul); // [x*y + y*y, z*y + 0, ...]
    float e01 = _mm_cvtss_f32(sum);
    float e2 = _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1,1,1,1)));
    return e01 + e2;
}

inline IVec3f getOrtogonal(const IVec3f &a, const IVec3f &b) {
    float bLen2 = b.length2();
    float c = dot(a, b) / bLen2;
    return a - b * c;
}

inline IVec3f cross(const IVec3f &a, const IVec3f &b) noexcept {
    GM_ASSERT_VEC3(a);
    GM_ASSERT_VEC3(b);
    return IVec3f{
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x()
    };
}

inline IVec3f cordPow(const IVec3f &base, const IVec3f &exp) {
    GM_ASSERT_VEC3(base);
    GM_ASSERT_VEC3(exp);
    GM_ASSERT(base.x() >= 0 && base.y() >= 0 && base.z() >= 0);

    return IVec3f{
        std::powf(base.x(), exp.x()),
        std::powf(base.y(), exp.y()),
        std::powf(base.z(), exp.z())
    };
}


class IPoint3 {
    __m128 cords_; // layout: _mm_set_ps(0.0f, z, y, x)

public:
    IPoint3() noexcept : cords_(_mm_setzero_ps()) {}
    IPoint3(float x, float y, float z) noexcept : cords_(_mm_set_ps(0.0f, z, y, x)) {}
    explicit IPoint3(__m128 cords) noexcept : cords_(cords) {}
    explicit IPoint3(float v) noexcept : cords_(_mm_set1_ps(v)) {}

    // copy / assign
    IPoint3(const IPoint3& o) noexcept = default;
    IPoint3& operator=(const IPoint3& o) noexcept = default;

    // Point - Point = Vector (requires IVec3f(__m128))
    IVec3f operator-(const IPoint3 &other) const noexcept {
        return IVec3f(_mm_sub_ps(cords_, other.cords_));
    }

    // Point + Vector = Point
    IPoint3 operator+(const IVec3f &vec) const noexcept {
        return IPoint3(_mm_add_ps(cords_, vec.cords()));
    }

    // Point - Vector = Point
    IPoint3 operator-(const IVec3f &vec) const noexcept {
        return IPoint3(_mm_sub_ps(cords_, vec.cords()));
    }

    // Scalar ops
    IPoint3 operator*(float s) const noexcept {
        __m128 sv = _mm_set1_ps(s);
        return IPoint3(_mm_mul_ps(cords_, sv));
    }
    IPoint3 operator/(float s) const noexcept {
        assert(s != 0.0f);
        __m128 sv = _mm_set1_ps(s);
        return IPoint3(_mm_div_ps(cords_, sv));
    }

    // Accessors (no memory stores)
    float x() const noexcept { return _mm_cvtss_f32(cords_); }

    float y() const noexcept {
        __m128 sh = _mm_shuffle_ps(cords_, cords_, _MM_SHUFFLE(1,1,1,1));
        return _mm_cvtss_f32(sh);
    }

    float z() const noexcept {
        __m128 sh = _mm_shuffle_ps(cords_, cords_, _MM_SHUFFLE(2,2,2,2));
        return _mm_cvtss_f32(sh);
    }

    // Setters (modify in-place)
    void setX(const float scalar) { mm_128_set_elem(cords_, 0, scalar); }
    void setY(const float scalar) { mm_128_set_elem(cords_, 1, scalar); }
    void setZ(const float scalar) { mm_128_set_elem(cords_, 2, scalar); }

    // stream operators
    friend std::ostream& operator<<(std::ostream& os, const IPoint3& p) {
        return os << "point3{" << p.x() << ", " << p.y() << ", " << p.z() << "}";
    }
    friend std::istream& operator>>(std::istream& is, IPoint3& p) {
        float a, b, c;
        is >> a >> b >> c;
        if (is) { p.setX(a); p.setY(b); p.setZ(c); }
        return is;
    }
};


/* -------------------- IVec2f -------------------- */
class IVec2f {
    __m128 cords_;            // layout: _mm_set_ps(0.0f, 0.0f, y, x) or _mm_set_ps(0,y,x,0) but we use low 2 lanes
    mutable float len2Cache_;
    mutable bool len2Dirty_;

public:
    IVec2f() noexcept : cords_(_mm_setzero_ps()), len2Cache_(0.0f), len2Dirty_(true) {}
    IVec2f(float x, float y) noexcept : cords_(_mm_set_ps(0.0f, 0.0f, y, x)), len2Cache_(0.0f), len2Dirty_(true) {}
    explicit IVec2f(__m128 cords) noexcept : cords_(cords), len2Cache_(0.0f), len2Dirty_(true) {}
    explicit IVec2f(float val) noexcept : cords_(_mm_set1_ps(val)), len2Cache_(0.0f), len2Dirty_(true) {}

    // copy / assign default
    IVec2f(const IVec2f& o) noexcept = default;
    IVec2f& operator=(const IVec2f& o) noexcept = default;

    // arithmetic
    IVec2f operator+(const IVec2f &o) const noexcept { return IVec2f(_mm_add_ps(cords_, o.cords_)); }
    IVec2f operator-(const IVec2f &o) const noexcept { return IVec2f(_mm_sub_ps(cords_, o.cords_)); }
    IVec2f operator*(float s) const noexcept {
        __m128 sv = _mm_set1_ps(s);
        return IVec2f(_mm_mul_ps(cords_, sv));
    }
    IVec2f operator/(float s) const noexcept {
        assert(s != 0.0f);
        __m128 sv = _mm_set1_ps(s);
        return IVec2f(_mm_div_ps(cords_, sv));
    }

    IVec2f& operator+=(const IVec2f &o) noexcept { cords_ = _mm_add_ps(cords_, o.cords_); len2Dirty_ = true; return *this; }
    IVec2f& operator-=(const IVec2f &o) noexcept { cords_ = _mm_sub_ps(cords_, o.cords_); len2Dirty_ = true; return *this; }
    IVec2f& operator*=(float s) noexcept { __m128 sv = _mm_set1_ps(s); cords_ = _mm_mul_ps(cords_, sv); len2Dirty_ = true; return *this; }
    IVec2f& operator/=(float s) noexcept { assert(s != 0.0f); __m128 sv = _mm_set1_ps(s); cords_ = _mm_div_ps(cords_, sv); len2Dirty_ = true; return *this; }

    // accessors
    float x() const noexcept { return mm_128_get_elem(cords_, 0); }
    float y() const noexcept { return mm_128_get_elem(cords_, 1); }
    __m128 cords() const noexcept { return cords_; }

    void setX(float v) noexcept { mm_128_set_elem(cords_, 0, v); len2Dirty_ = true; }
    void setY(float v) noexcept { mm_128_set_elem(cords_, 1, v); len2Dirty_ = true; }

    // math
    float dot(const IVec2f &o) const noexcept {
        __m128 m = _mm_mul_ps(cords_, o.cords_);          // [x1*x2, y1*y2, ...]
        __m128 h = _mm_hadd_ps(m, m);                    // [x1*x2 + y1*y2, ...]
        return _mm_cvtss_f32(h);
    }

    float length2() const noexcept {
        if (len2Dirty_) {
            __m128 m = _mm_mul_ps(cords_, cords_);
            __m128 h = _mm_hadd_ps(m, m);
            len2Cache_ = _mm_cvtss_f32(h);
            len2Dirty_ = false;
        }
        return len2Cache_;
    }

    float length() const noexcept { return std::sqrtf(length2()); }

    IVec2f normalized() const noexcept {
        float len = length();
        if (len == 0.0f) return IVec2f(0.0f, 0.0f);
        return *this / len;
    }

    bool isFinite() const noexcept { return std::isfinite(x()) && std::isfinite(y()); }

    bool nearZero() const noexcept {
        const float eps = std::numeric_limits<float>::epsilon();
        return (std::fabsf(x()) < eps) && (std::fabsf(y()) < eps);
    }

    // stream
    friend std::ostream& operator<<(std::ostream& os, const IVec2f& v) {
        return os << "IVec2f{" << v.x() << ", " << v.y() << "}";
    }
    friend std::istream& operator>>(std::istream& is, IVec2f& v) {
        float a, b;
        is >> a >> b;
        if (is) {
            v.setX(a);
            v.setY(b);
            v.len2Dirty_ = true;
        }
        return is;
    }
};

/* -------------------- IPoint2f -------------------- */
class IPoint2f {
    __m128 cords_; // same layout: low lanes contain x,y

public:
    IPoint2f() noexcept : cords_(_mm_setzero_ps()) {}
    IPoint2f(float x, float y) noexcept : cords_(_mm_set_ps(0.0f, 0.0f, y, x)) {}
    explicit IPoint2f(__m128 cords) noexcept : cords_(cords) {}
    explicit IPoint2f(float v) noexcept : cords_(_mm_set1_ps(v)) {}

    // copy / assign default
    IPoint2f(const IPoint2f& o) noexcept = default;
    IPoint2f& operator=(const IPoint2f& o) noexcept = default;

    // point - point = vector
    IVec2f operator-(const IPoint2f &o) const noexcept {
        return IVec2f(_mm_sub_ps(cords_, o.cords_));
    }

    // point + vector = point
    IPoint2f operator+(const IVec2f &vec) const noexcept {
        return IPoint2f(_mm_add_ps(cords_, vec.cords()));
    }

    // point - vector = point
    IPoint2f operator-(const IVec2f &vec) const noexcept {
        return IPoint2f(_mm_sub_ps(cords_, vec.cords()));
    }

    // scalar ops
    IPoint2f operator*(float s) const noexcept {
        __m128 sv = _mm_set1_ps(s);
        return IPoint2f(_mm_mul_ps(cords_, sv));
    }
    IPoint2f operator/(float s) const noexcept {
        assert(s != 0.0f);
        __m128 sv = _mm_set1_ps(s);
        return IPoint2f(_mm_div_ps(cords_, sv));
    }

    // accessors
    float x() const noexcept { return mm_128_get_elem(cords_, 0); }
    float y() const noexcept { return mm_128_get_elem(cords_, 1); }

    void setX(float v) noexcept { mm_128_set_elem(cords_, 0, v); }
    void setY(float v) noexcept { mm_128_set_elem(cords_, 1, v); }

    // stream
    friend std::ostream& operator<<(std::ostream& os, const IPoint2f& p) {
        return os << "point2{" << p.x() << ", " << p.y() << "}";
    }
    friend std::istream& operator>>(std::istream& is, IPoint2f& p) {
        float a, b;
        is >> a >> b;
        if (is) { p.setX(a); p.setY(b); }
        return is;
    }
};

}
