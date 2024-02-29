#include "dual.hh"

#include <complex>
#include <iostream>
#include <type_traits>

#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

// Tests are given a tag "[base]" do not used nested Duals; tests given the tag
// "[template]" use nested Duals.

TEMPLATE_TEST_CASE("default constructed base duals are floating-point zeros",
                   "[base]",
                   float,
                   double,
                   std::complex<float>,
                   std::complex<double>)
{
  using dual_type = jac::Dual<TestType, 2>;
  static_assert(std::is_same_v<typename dual_type::inner_type, TestType>);

  // A default-constructed dual should have the right size, and all components
  // should be zero.
  dual_type x;
  static_assert(x.size() == 3);
  // The cast below is needed specifically to support the testing of
  // std::complex<float>.
  REQUIRE(x[0] == static_cast<TestType>(0.0));
  REQUIRE(x[1] == static_cast<TestType>(0.0));
  REQUIRE(x[2] == static_cast<TestType>(0.0));
}

TEMPLATE_TEST_CASE("default constructed duals are zeros",
                   "[template]",
                   float,
                   double,
                   std::complex<float>,
                   std::complex<double>,
                   (jac::Dual<double, 2>),
                   (jac::Dual<std::complex<double>, 2>),
                   (jac::Dual < jac::Dual<double, 3>, 4) >)
{
  using dual_type = jac::Dual<TestType, 2>;
  static_assert(std::is_same_v<typename dual_type::inner_type, TestType>);
  dual_type x;
  static_assert(x.size() == 3);

  for (std::size_t i = 0; i != x.size(); ++i) {
    REQUIRE(x[i] == dual_type::inner_type_zero);
  }
}

TEMPLATE_TEST_CASE("create dual number from nonzero value of scalar type",
                   "[base]",
                   float,
                   double,
                   std::complex<float>,
                   std::complex<double>)
{
  using dual_type = jac::Dual<TestType, 2>;
  dual_type x{2.5};
  // The cast below is needed specifically to support the testing of
  // std::complex<float>.
  REQUIRE(x[0] == static_cast<TestType>(2.5));
  REQUIRE(x[1] == static_cast<TestType>(0.0));
  REQUIRE(x[2] == static_cast<TestType>(0.0));
}

TEST_CASE("create dual number for nonzero value of inner type", "[template]")
{
  jac::Dual<double, 2> inner{3.5};
  jac::Dual<double, 2> inner_zero;
  jac::Dual<jac::Dual<double, 2>, 3> x{inner};
  REQUIRE(x[0] == inner);
  REQUIRE(x[1] == inner_zero);
  REQUIRE(x[2] == inner_zero);
  REQUIRE(x[3] == inner_zero);
}

TEST_CASE("create dual number using for_deriv factory function", "[base]")
{
  auto x = jac::Dual<double, 2>::for_deriv(3.5, 1);
  REQUIRE(x[0] == 3.5);
  REQUIRE(x[1] == 1.0);
  REQUIRE(x[2] == 0.0);

  auto y = jac::Dual<double, 2>::for_deriv(2.5, 2);
  REQUIRE(y[0] == 2.5);
  REQUIRE(y[1] == 0.0);
  REQUIRE(y[2] == 1.0);
}

TEMPLATE_TEST_CASE("create dual number using for_deriv factor function",
                   "[template]",
                   float,
                   double,
                   std::complex<float>,
                   std::complex<double>,
                   (jac::Dual<double, 2>),
                   (jac::Dual<std::complex<float>, 2>),
                   (jac::Dual<jac::Dual<double, 3>, 4>))
{
  using inner_t = TestType;
  inner_t a = {1.5};
  inner_t b = {2.5};
  inner_t c = {3.5};
  using OuterType = jac::Dual<inner_t, 3>;
  auto x = OuterType::for_deriv(a, 1);
  auto y = OuterType::for_deriv(b, 2);
  auto z = OuterType::for_deriv(c, 3);

  REQUIRE(x[0] == a);
  REQUIRE(x[1] == OuterType::inner_type_one);
  REQUIRE(x[2] == OuterType::inner_type_zero);
  REQUIRE(x[3] == OuterType::inner_type_zero);

  REQUIRE(y[0] == b);
  REQUIRE(y[1] == OuterType::inner_type_zero);
  REQUIRE(y[2] == OuterType::inner_type_one);
  REQUIRE(y[3] == OuterType::inner_type_zero);

  REQUIRE(z[0] == c);
  REQUIRE(z[1] == OuterType::inner_type_zero);
  REQUIRE(z[2] == OuterType::inner_type_zero);
  REQUIRE(z[3] == OuterType::inner_type_one);
}

TEST_CASE("indexing into nested Dual", "[template]")
{
  jac::Dual<jac::Dual<jac::Dual<double, 2>, 3>, 4> x{2.5};
  REQUIRE(x.size() == 5);
  auto a = x[0];
  REQUIRE(a.size() == 4);
  auto b = a[0];
  REQUIRE(b.size() == 3);
  auto c = b[0];
  static_assert(std::is_same_v<decltype(c), double>);
  REQUIRE(c == 2.5);
}

TEST_CASE("inplace addition", "[base]")
{
  jac::Dual<double, 2> x = {1.0, 2.0, 3.0};
  jac::Dual<double, 2> y = {1.5, 2.5, 3.5};
  x += y;
  jac::Dual<double, 2> expected = {2.5, 4.5, 6.5};
  REQUIRE(x == expected);
}

TEST_CASE("addition", "[base]")
{
  jac::Dual<double, 2> x = {1.0, 2.0, 3.0};
  jac::Dual<double, 2> y = {1.5, 2.5, 3.5};
  auto z = x + y;
  jac::Dual<double, 2> expected = {2.5, 4.5, 6.5};
  REQUIRE(z == expected);
}

TEST_CASE("addition", "[template]")
{
  jac::Dual<jac::Dual<double, 2>, 2> x;
  x[0] = {1., 2., 3.};
  x[1] = {4., 5., 6.};
  x[2] = {7., 8., 9.};
  auto y(x); // copy x
  REQUIRE(x == y);
  auto z = x + y;
  REQUIRE(z[0] == jac::Dual<double, 2>{2., 4., 6.});
  REQUIRE(z[1] == jac::Dual<double, 2>{8., 10., 12.});
  REQUIRE(z[2] == jac::Dual<double, 2>{14., 16., 18.});
}

TEST_CASE("inplace subtraction", "[base]")
{
  jac::Dual<double, 2> x = {1.0, 2.0, 3.0};
  jac::Dual<double, 2> y = {1.5, 2.5, 3.5};
  x -= y;
  jac::Dual<double, 2> expected = {-0.5, -0.5, -0.5};
  REQUIRE(x == expected);
}

TEST_CASE("subtraction", "[base]")
{
  jac::Dual<double, 2> x = {1.0, 2.0, 3.0};
  jac::Dual<double, 2> y = {1.5, 2.5, 3.5};
  auto z = x - y;
  jac::Dual<double, 2> expected = {-0.5, -0.5, -0.5};
  REQUIRE(z == expected);
}

TEST_CASE("subtraction", "[template]")
{
  jac::Dual<jac::Dual<double, 2>, 2> x;
  x[0] = {1., 2., 3.};
  x[1] = {4., 5., 6.};
  x[2] = {7., 8., 9.};
  auto y(x); // copy x
  REQUIRE(x == y);
  auto z = x - y;
  REQUIRE(z == jac::Dual<jac::Dual<double, 2>, 2>{});
}

TEST_CASE("multiplication", "[base]")
{
  auto f = [](jac::Dual<double, 2> const& x, jac::Dual<double, 2> const& y) {
    return x * y;
  };
  auto x = jac::Dual<double, 2>::for_deriv(4.0, 1);
  auto y = jac::Dual<double, 2>::for_deriv(2.0, 2);
  jac::Dual<double, 2> z = f(x, y);
  REQUIRE(z[0] == 8.0);
  REQUIRE(z[1] == 2.0);
  REQUIRE(z[2] == 4.0);

  auto u = x * y;
  REQUIRE(z == u);
}

// TODO: Add testing for multiplication of nested Duals.

TEST_CASE("division", "[base]")
{
  auto f = [](jac::Dual<double, 2> const& x, jac::Dual<double, 2> const& y) {
    return x / y;
  };
  auto x = jac::Dual<double, 2>::for_deriv(4.0, 1);
  auto y = jac::Dual<double, 2>::for_deriv(2.0, 2);
  jac::Dual<double, 2> z = f(x, y);
  REQUIRE(z[0] == 2.0);
  REQUIRE(z[1] == 0.5);
  REQUIRE(z[2] == -1.0);
}

// TODO: add testing for division of nested duals.

TEST_CASE("aggregate initialization", "[base]")
{
  jac::Dual<double, 3> x{1., 2., 3., 4.};
  auto y = jac::Dual<double, 3>::from_init({1., 2., 3., 4.});
  REQUIRE(x == y);

  std::initializer_list<double> vals{2., 4., 6., 8.};
  auto z = jac::Dual<double, 3>::from_init(vals);
  auto t = x + x;
  REQUIRE(t == z);
}

TEST_CASE("aggregate initialization", "[template]")
{
  auto x = jac::Dual<jac::Dual<double, 2>, 3>::from_init(
    {{1., 2., 3.}, {3., 4., 5.}, {5., 6., 7.}, {7., 8., 9.}});
  REQUIRE(x.size() == 4);
  for (std::size_t i = 0; i != 4; ++i) {
    REQUIRE(x[i].size() == 3);
  }
  REQUIRE(x[0] == jac::Dual<double, 2>::from_init({1., 2., 3.}));

  // TODO: Try to get the following to work for initializing a
  // nested Dual.
  //
  // jac::Dual<jac::Dual<double, 2>, 3> a{
  //   {1., 2., 3.}, {3., 4., 5.}, {5., 6., 7.}, {7., 8., 9.}};
  // REQUIRE(a == x);
}

template <typename T>
T
f(T x, T y)
{
  return x * x + y * y;
}

TEST_CASE("testing x*x + y*y") {
  using d1 = jac::Dual<double, 2>;

  d1 x{2.0, 1.0, 0.0};
  d1 y{3.0, 0.0, 1.0};

  auto res = f(x, y);
  REQUIRE(res[0] == 13.0);
  REQUIRE(res[1] == 4.0);
  REQUIRE(res[2] == 6.0);
}

TEST_CASE("second deriv") {
  using d1 = jac::Dual<double, 2>;
  using d2 = jac::Dual<d1, 2>;

  d2 x{d1{2.0, 1.0, 0.0},d1{1.0, 0.0, 0.0}, d1{0.0, 0.0, 0.0}};
  d2 y{d1{3.0, 0.0, 1.0},d1{0.0, 0.0, 0.0}, d1{1.0, 0.0, 0.0}};
  auto res = f(x, y);
  REQUIRE(res[0] == d1{13.0, 4.0, 6.0});
  REQUIRE(res[1] == d1{4.0, 2.0, 0.0});
  REQUIRE(res[2] == d1{6.0, 0.0, 2.0});
}

