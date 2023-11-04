#include "dual.hh"

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

using jac::Dual;

// NOTE: These tests include a good deal of floating point math.
// When using transcendental function, we use the Catch2 facilities for testing
// approximate equality (because the transcendental functions in the math
// library are approximate). When using arithmetic functions, we try to use
// numerical values that are exactly representable in IEEE floating point, so
// that we can do exact tests of the results.

TEST_CASE("default construction")
{
  Dual<1> x1;
  CHECK(std::size(x1) == 2);
  for (int i = 0; i != std::size(x1); ++i) {
    CHECK(x1[i] == 0.0);
  }

  Dual<3> x3;
  CHECK(std::size(x3) == 4);
  for (int i = 0; i != std::size(x3); ++i) {
    CHECK(x3[i] == 0.0);
  }
}

TEST_CASE("implicit conversion from double")
{
  Dual<2> x = 4.0;
  CHECK(x[0] == 4.0);
  CHECK(x[1] == 0.0);
}

TEST_CASE("assignment from double")
{
  Dual<2> x;
  x = 4.0;
  CHECK(x[0] == 4.0);
  CHECK(x[1] == 0.0);
}

TEST_CASE("aggregate initialization")
{
  Dual<1> x(std::array{3., 2.});
  CHECK(x[0] == 3.0);
  CHECK(x[1] == 2.0);
}

TEST_CASE("sin function")
{
  Dual<1> x(0.0, 1);
  auto y = sin(x);
  CHECK(y[0] == 0.0);
  CHECK_THAT(y[1], Catch::Matchers::WithinAbs(1.0, 1e-6));
}

TEST_CASE("cos function")
{
  Dual<1> x(0.0, 1);
  auto y = cos(x);
  CHECK(y[0] == 1.0);
  CHECK_THAT(y[1], Catch::Matchers::WithinAbs(0.0, 1e-15));
}

TEST_CASE("addition")
{
  auto f = [](Dual<2> const& x, Dual<2> const& y) { return 2 * x + y; };
  Dual<2> x(2.0, 1);
  Dual<2> y(5.0, 2);
  Dual<2> z = f(x, y);
  CHECK(z[0] == 9.0);
  CHECK(z[1] == 2.0);
  CHECK(z[2] == 1.0);
}

TEST_CASE("subtraction")
{
  auto f = [](Dual<2> const& x, Dual<2> const& y) { return 3 * x - 2 * y; };
  Dual<2> x(2.0, 1);
  Dual<2> y(5.0, 2);
  Dual<2> z = f(x, y);
  CHECK(z[0] == -4.0);
  CHECK(z[1] == 3.0);
  CHECK(z[2] == -2.0);
}

TEST_CASE("multiplication")
{
  auto f = [](Dual<2> const& x, Dual<2> const& y) { return x * y; };
  Dual<2> x(4.0, 1);
  Dual<2> y(2.0, 2);
  Dual<2> z = f(x, y);
  CHECK(z[0] == 8.0);
  CHECK(z[1] == 2.0);
  CHECK(z[2] == 4.0);
}

TEST_CASE("division")
{
  auto f = [](Dual<2> const& x, Dual<2> const& y) { return x / y; };
  Dual<2> x(4.0, 1);
  Dual<2> y(2.0, 2);
  Dual<2> z = f(x, y);
  CHECK(z[0] == 2.0);
  CHECK(z[1] == 0.5);
  CHECK(z[2] == -1.0);
}
