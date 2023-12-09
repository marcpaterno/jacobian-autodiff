#include "dual.hh"

#include <complex>
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

  for (int i = 0; i != x.size(); ++i) {
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
  using inner_type = typename dual_type::inner_type;

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