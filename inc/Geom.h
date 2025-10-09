#ifndef GEOM_H
#define GEOM_H

#include <concepts>
#include <cmath>
#include <cassert>
#include <limits>
#include <algorithm>
#include <ostream>
#include <istream>

#ifdef GM_DEBUG
    #define GM_ASSERT_VEC3(v) assert( (v).isFinite() )
    #define GM_ASSERT_VEC2(v) assert( (v).isFinite() )
    #define GM_ASSERT(cond) assert(cond)
#else
    #define GM_ASSERT_VEC3(v) ((void)0)
    #define GM_ASSERT_VEC2(v) ((void)0)
    #define GM_ASSERT(cond) ((void)0)
#endif




template<std::floating_point T, std::size_t N>
struct GmPoint;
template<std::floating_point T, std::size_t N>
class GmVec;



// ------------------------------------ 3D specialization ------------------------------------
template<std::floating_point T>
class GmVec<T,3> {
    T x_{}, y_{}, z_{};
    mutable T len2Cache_;
    mutable bool len2Dirty_;

public:
    constexpr GmVec() noexcept : x_(), y_(), z_(), len2Cache_(), len2Dirty_(true) {}
    constexpr GmVec(T x_, T y_, T z_) noexcept : x_(x_), y_(y_), z_(z_), len2Cache_(), len2Dirty_(true) {}
    constexpr explicit GmVec(T val) noexcept : x_(val), y_(val), z_(val), len2Cache_(), len2Dirty_(true) {}
    template<std::floating_point U>
    constexpr GmVec(const GmVec<U,3>& other) noexcept
        : x_(static_cast<T>(other.x())), y_(static_cast<T>(other.y())), z_(static_cast<T>(other.z())),
          len2Cache_(), len2Dirty_(true) {}

    // ------------------------------------ Operators ------------------------------------
    constexpr GmVec<T, 3> operator+(const GmVec<T, 3> &other) const noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        return GmVec<T, 3>(x_ + other.x(), y_ + other.y(), z_ + other.z());
    }
    constexpr GmVec<T, 3> operator-(const GmVec<T, 3> &other) const noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        return GmVec<T, 3>(x_ - other.x(), y_ - other.y(), z_ - other.z());
    }
    constexpr GmVec<T, 3> operator*(const GmVec<T, 3> &other) const noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        return GmVec<T, 3>(x_ * other.x(), y_ * other.y(), z_ * other.z());
    }
    constexpr GmVec<T, 3> operator*(const T scalar) const noexcept {
        GM_ASSERT_VEC3(*this);

        return GmVec<T, 3>(x_ * scalar, y_ * scalar, z_ * scalar);
    }
    constexpr GmVec<T, 3> operator/(const T scalar) const {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT(scalar != 0);

        return GmVec<T, 3>(x_ / scalar, y_ / scalar, z_ / scalar);
    }
    constexpr GmVec<T, 3> operator+=(const GmVec<T, 3> &other) noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        x_ += other.x();
        y_ += other.y();
        z_ += other.z();
        len2Dirty_ = true;
        
        return *this;
    }
    constexpr GmVec<T, 3> operator-=(const GmVec<T, 3> &other) noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        x_ -= other.x();
        y_ -= other.y();
        z_ -= other.z();
        len2Dirty_ = true;
        
        return *this;
    }
    constexpr GmVec<T, 3> operator*=(const GmVec<T, 3> &other) noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);

        x_ *= other.x();
        y_ *= other.y();
        z_ *= other.z();
        len2Dirty_ = true;
        
        return *this;
    }
    constexpr GmVec<T, 3> operator/=(const T scalar) noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT(scalar != 0);

        x_ /= scalar;
        y_ /= scalar;
        z_ /= scalar;
        len2Dirty_ = true;
        
        return *this;
    }
    GmVec<T, 3> &operator=(const GmVec<T, 3> &other) noexcept = default;
    constexpr bool operator==(const GmVec<T, 3> &other) const noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);
    
        return x_ == other.x() && y_ == other.y() && z_ == other.z();
    }
    constexpr bool operator!=(const GmVec<T, 3> &other) const noexcept {
        GM_ASSERT_VEC3(*this);
        GM_ASSERT_VEC3(other);
    
        return !(*this == other);
    }
    // --------------------------------------- Math -----------------------------------------------
    [[nodiscard]] constexpr GmVec<T, 3> clamped(const T min, const T max) const noexcept {
        GM_ASSERT_VEC3(*this);

        return GmVec<T, 3>
        (
            std::clamp(x_, min, max),
            std::clamp(y_, min, max),
            std::clamp(z_, min, max)
        );
    }
    [[nodiscard]] constexpr bool isFinite() const noexcept {
        return std::isfinite(x_) && std::isfinite(y_) && std::isfinite(z_);
    }
    [[nodiscard]] constexpr GmVec<T, 3> normalized() const noexcept {
        T len = length();
        return len > T(0) ? *this / len : GmVec<T, 3>();
    }

    // --------------------------------------- Getters --------------------------------------------
    [[nodiscard]] constexpr T x() const noexcept { return x_; }
    [[nodiscard]] constexpr T y() const noexcept { return y_; }
    [[nodiscard]] constexpr T z() const noexcept { return z_; }

    [[nodiscard]] constexpr T length() const noexcept { return std::sqrt(length2()); }
    [[nodiscard]] constexpr T length2() const noexcept 
    {
        if (len2Dirty_) {
            len2Cache_ = x_*x_ + y_*y_ + z_*z_;
            len2Dirty_ = false;
        }
        return len2Cache_;
    }

    // --------------------------------------- Setters --------------------------------------------
    void setX(const T scalar) noexcept
    {
        x_ = scalar;
        len2Dirty_ = true;
    }
    void setY(const T scalar) noexcept
    {
        y_ = scalar;
        len2Dirty_ = true;
    }
    void setZ(const T scalar) noexcept
    {
        z_ = scalar;
        len2Dirty_ = true;
    }

    // ---------------- Stream operators ----------------------------------------------------------
    friend std::ostream& operator<<(std::ostream& os, const GmVec<T,3>& v) {
        return os << "vec3{" << v.x_ << ", " << v.y_ << ", " << v.z_ << "}";
    }

    friend std::istream& operator>>(std::istream& is, GmVec<T,3>& v) {
        T x, y, z;
        is >> x >> y >> z;
        if (is) {
            v.x_ = x; 
            v.y_ = y;
            v.z_ = z;
            v.len2Dirty_ = true;
        }
        return is;
    }
};

// --------------------- Free functions -------------------------------------------------------
template<std::floating_point T>
[[nodiscard]] constexpr T dot(const GmVec<T,3> &a, const GmVec<T,3> &b) noexcept {
    GM_ASSERT_VEC3(a);
    GM_ASSERT_VEC3(b);
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

template<std::floating_point T>
[[nodiscard]] constexpr GmVec<T,3> cross(const GmVec<T,3> &a, const GmVec<T,3> &b) noexcept {
    GM_ASSERT_VEC3(a);
    GM_ASSERT_VEC3(b);
    return GmVec<T,3>{
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x()
    };
}

template<std::floating_point T>
[[nodiscard]] GmVec<T,3> cordPow(const GmVec<T,3> &base, const GmVec<T,3> &exp) {
    GM_ASSERT_VEC3(base);
    GM_ASSERT_VEC3(exp);
    GM_ASSERT(base.x() >= T(0) && base.y() >= T(0) && base.z() >= T(0));

    return GmVec<T,3>{
        static_cast<T>(std::pow(static_cast<double>(base.x()), static_cast<double>(exp.x()))),
        static_cast<T>(std::pow(static_cast<double>(base.y()), static_cast<double>(exp.y()))),
        static_cast<T>(std::pow(static_cast<double>(base.z()), static_cast<double>(exp.z())))
    };
}


template<std::floating_point T>
struct GmPoint<T, 3> {
    T x{}, y{}, z{};

    constexpr GmPoint() noexcept = default;
    constexpr GmPoint(T x_, T y_, T z_) noexcept : x(x_), y(y_), z(z_) {}
    constexpr explicit GmPoint(T val) noexcept : x(val), y(val), z(val) {}
    template<std::floating_point U>
    constexpr GmPoint(const GmPoint<U,3>& other) noexcept
        : x(static_cast<T>(other.x)), y(static_cast<T>(other.y)), z(static_cast<T>(other.z)) {}


    // ------------------------------- Point–Point operators ----------------------------------
    [[nodiscard]] constexpr GmVec<T, 3> operator-(const GmPoint<T, 3> &other) const noexcept {
        return GmVec<T, 3>(x - other.x, y - other.y, z - other.z);
    }
    [[nodiscard]] constexpr GmPoint<T, 3> operator+(const GmPoint<T, 3> &other) const noexcept {
        return {x + other.x, y + other.y, z + other.z};
    }
    [[nodiscard]] constexpr GmPoint<T, 3> operator*(const GmPoint<T, 3> &other) const noexcept {
        return {x * other.x, y * other.y, z * other.z};
    }
    [[nodiscard]] constexpr GmPoint<T, 3> operator/(const GmPoint<T, 3> &other) const noexcept {
        GM_ASSERT(other.x != 0 && other.y != 0 && other.z != 0);
        return {x / other.x, y / other.y, z / other.z};
    }

    // ------------------------------- Point–Scalar operators ---------------------------------
    [[nodiscard]] constexpr GmPoint<T, 3> operator*(T scalar) const noexcept {
        return {x * scalar, y * scalar, z * scalar};
    }
    [[nodiscard]] constexpr GmPoint<T, 3> operator/(T scalar) const noexcept {
        GM_ASSERT(scalar != 0);
        return {x / scalar, y / scalar, z / scalar};
    }

    // --------------------------------------- Comparison -------------------------------------
    [[nodiscard]] constexpr bool operator==(const GmPoint<T, 3> &other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }
    [[nodiscard]] constexpr bool operator!=(const GmPoint<T, 3> &other) const noexcept {
        return !(*this == other);
    }

    // ---------------- Stream operators ------------------------------------------------------
    friend std::ostream& operator<<(std::ostream& os, const GmPoint<T, 3> &p) {
        return os << "point3{" << p.x << ", " << p.y << ", " << p.z << "}";
    }
    friend std::istream& operator>>(std::istream& is, GmPoint<T, 3> &p) {
        return is >> p.x >> p.y >> p.z;
    }
};


#endif // GEOM_H
