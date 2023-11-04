#include "dual.hh"
#include "jacobian.hh"

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

using jac::Dual;
using jac::dual_array;
using jac::make_jacobian;

// f1 :: dual_array<1> -> Dual<1>
// f1(x) = x^2 + 2x

// f1' :: dual_array<1> -> Dual<1>
// f1'(x) = 2 x + 2
inline Dual<1>
f1(dual_array<1> const& args)
{
  auto const x = args[0];
  return x * (x + 2.0);
}

TEST_CASE("function of one variable")
{
  dual_array<1> arg{Dual<1>(1.5)};
  Dual<1> res = f1(arg);
  CHECK(res[0] == 5.25);

  jac::Jacobian<decltype(f1), 1> f1deriv(f1);
  Dual<1> gradient = f1deriv(arg);
  CHECK(gradient[0] == 5.25);
  CHECK(gradient[1] == 5.0);
}

inline Dual<2>
f2(dual_array<2> const& args)
{
  auto const x = args[0];
  auto const y = args[1];
  return 2 * x - y*y*y;
}

TEST_CASE("function of two variables")
{
  dual_array<2> arg{Dual<2>(2.0), Dual<2>(3.0)};
  jac::Jacobian<decltype(f2), 2> f2deriv(f2);
  Dual<2> gradient = f2deriv(arg);
  CHECK(gradient[0] == -23.0);
  CHECK(gradient[1] == 2.0);
  CHECK(gradient[2] == -27.0);
}
