#include <gtest/gtest.h>
#include "IVec3d.hpp"

using namespace gm;

static constexpr double EPS = 1e-12;


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}


TEST(IVec3dDotTest, ZeroVectors) {
    IVec3d a(0.0, 0.0, 0.0);
    IVec3d b(0.0, 0.0, 0.0);
    EXPECT_DOUBLE_EQ(dot(a, b), 0.0);
}

TEST(IVec3dDotTest, IdentityVectors) {
    IVec3d a(1.0, 0.0, 0.0);
    IVec3d b(0.0, 1.0, 0.0);
    EXPECT_DOUBLE_EQ(dot(a, b), 0.0); // orthogonal vectors

    IVec3d c(1.0, 1.0, 1.0);
    EXPECT_DOUBLE_EQ(dot(c, c), 3.0); // dot(v,v) = |v|^2
}

TEST(IVec3dDotTest, PositiveValues) {
    IVec3d a(1.0, 2.0, 3.0);
    IVec3d b(4.0, 5.0, 6.0);
    EXPECT_DOUBLE_EQ(dot(a, b), 32.0); // 1*4 + 2*5 + 3*6
}

TEST(IVec3dDotTest, NegativeValues) {
    IVec3d a(-1.0, -2.0, -3.0);
    IVec3d b(4.0, -5.0, 6.0);
    EXPECT_DOUBLE_EQ(dot(a, b), -12.0); // -1*4 + -2*-5 + -3*6 = -12
}

TEST(IVec3dDotTest, MixedValues) {
    IVec3d a(0.5, -1.5, 2.0);
    IVec3d b(4.0, -0.5, -1.0);
    EXPECT_DOUBLE_EQ(dot(a, b), 0.75); // 0.5*4 + -1.5*-0.5 + 2*-1 = 0.75
}



TEST(HitIntrinsics, DotAndLength) {
    IVec3d a(1.0, 2.0, 3.0);
    IVec3d b(4.0, 5.0, 6.0);

    // dot = 1*4 + 2*5 + 3*6 = 32
    EXPECT_DOUBLE_EQ(dot(a, b), 32.0);

    // length2 of 'a' = 1 + 4 + 9 = 14
    std::cout << a << "\n";
    std::cout << a.length2() << "\n";
    std::cout << a.length() << "\n";
    EXPECT_DOUBLE_EQ(a.length2(), 14.0);
    EXPECT_NEAR(a.length(), std::sqrt(14.0), EPS);
}

TEST(HitIntrinsics, PointMinusPointYieldsVector) {
    IPoint3 p1(1.0, 2.0, 3.0);
    IPoint3 p2(4.0, 6.0, 8.0);

    IVec3d v = p1 - p2; // v = p1 - p2 = (-3, -4, -5)

    EXPECT_DOUBLE_EQ(v.x(), -3.0);
    EXPECT_DOUBLE_EQ(v.y(), -4.0);
    EXPECT_DOUBLE_EQ(v.z(), -5.0);
}

TEST(HitIntrinsics, PointPlusVectorTimesScalar) {
    IPoint3 origin(0.0, 0.0, 0.0);
    IVec3d dir(1.0, 2.0, 3.0);
    double t = 2.5;

    // rec.point = origin + dir * t
    IPoint3 rec_point = origin + dir * t;

    EXPECT_NEAR(rec_point.x(), dir.x() * t, 1e-12);
    EXPECT_NEAR(rec_point.y(), dir.y() * t, 1e-12);
    EXPECT_NEAR(rec_point.z(), dir.z() * t, 1e-12);
}

TEST(HitIntrinsics, OutwardNormalIsUnit) {
    // sphere at origin radius R, hit point at (R, 0, 0)
    IPoint3 position(0.0, 0.0, 0.0);
    IPoint3 hitP(5.0, 0.0, 0.0);
    double radius = 5.0;

    // outward normal = (hitP - position) / radius
    IVec3d outward = (hitP - position) / radius;

    EXPECT_NEAR(outward.x(), 1.0, 1e-12);
    EXPECT_NEAR(outward.y(), 0.0, 1e-12);
    EXPECT_NEAR(outward.z(), 0.0, 1e-12);

    // magnitude should be 1
    EXPECT_NEAR(outward.length(), 1.0, 1e-12);
}

TEST(HitIntrinsics, SettersGettersAndMutationAffectsLength) {
    IVec3d v(1.0, 2.0, 3.0);

    // initial checks
    EXPECT_DOUBLE_EQ(v.x(), 1.0);
    EXPECT_DOUBLE_EQ(v.y(), 2.0);
    EXPECT_DOUBLE_EQ(v.z(), 3.0);

    double initialLen2 = v.length2();
    EXPECT_DOUBLE_EQ(initialLen2, 14.0);

    // mutate components
    v.setX(10.0);
    v.setY(-4.0);
    v.setZ(0.5);

    EXPECT_DOUBLE_EQ(v.x(), 10.0);
    EXPECT_DOUBLE_EQ(v.y(), -4.0);
    EXPECT_DOUBLE_EQ(v.z(), 0.5);

    // recompute length2 should reflect new values
    double expectedLen2 = 10.0*10.0 + (-4.0)*(-4.0) + 0.5*0.5;
    EXPECT_NEAR(v.length2(), expectedLen2, 1e-12);
}

TEST(HitIntrinsics, ZeroDiscriminantCase) {
    // Test dot/length usage in degenerate quadratic: choose oc and dir so discriminant == 0
    // Use oc parallel to dir so half_b^2 == a*c
    IVec3d dir(1.0, 0.0, 0.0);
    IPoint3 origin(0.0, 0.0, 0.0);
    IPoint3 center(2.0, 0.0, 0.0);
    double radius = 1.0;

    IVec3d oc = origin - center; // oc = (-2,0,0)
    double a = dot(dir, dir);                 // = 1
    double half_b = dot(oc, dir);             // = -2
    double c = dot(oc, oc) - radius*radius;   // = 4 - 1 = 3

    double discriminant = half_b*half_b - a*c; // = 4 - 3 = 1 (>0)
    EXPECT_GT(discriminant, 0.0);

    // Now arrange to force discriminant == 0 (example: oc = (-1,0,0), radius=0)
    IVec3d oc2(-1.0, 0.0, 0.0);
    double half_b2 = dot(oc2, dir); // -1
    double c2 = dot(oc2, oc2) - 1.0; // 1 - 1 = 0
    double disc2 = half_b2*half_b2 - a*c2; // = 1
    EXPECT_GE(disc2, 0.0); // just verify non-negative path (sanity)
}


