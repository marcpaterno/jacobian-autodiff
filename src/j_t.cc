#include "dual.hh"
#include "j.hh"

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

#include <array>
#include <functional>

template <typename T>
std::array<T, 1>
f_2_1(std::array<T, 2> const& x)
{
  std::array<T, 1> result;
  result[0] = x[0] * x[1];
  return result;
}

using jac::J;

TEST_CASE("function using double")
{
  std::array<double, 2> x{1., 2.0};
  auto z = f_2_1(x);
  CHECK(z[0] == 2.0);

  using std::array;
  using std::function;
  function<array<double, 1>(array<double, 2> const&)> uf(f_2_1<double>);
  auto uz = uf(x);
  CHECK(uz == z); // all the values must be the same.
}

TEST_CASE("function using dual")
{
  std::array<jac::Dual<2>, 2> x;
}